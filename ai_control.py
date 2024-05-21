#!/usr/bin/env python

import glob
import os
import sys
from collections import deque
import math
import numpy as np

try:
    sys.path.append(
        glob.glob(
            "**/*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla
import ai_knowledge as data
from ai_knowledge import Status
import numpy as np


# Executor is responsible for moving the vehicle around
# In this implementation it only needs to match the steering and speed so that we arrive at provided waypoints
# BONUS TODO: implement different speed limits so that planner would also provide speed target speed in addition to direction

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, delta_time):
        self.integral += error * delta_time
        derivative = (error - self.prev_error) / delta_time
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class Executor(object):
    def __init__(self, knowledge, vehicle):
        self.vehicle = vehicle
        self.knowledge = knowledge
        self.target_pos = knowledge.get_location()

        # PID controllers for longitudinal and lateral control
        self.longitudinal_pid = PIDController(0.14, 0.00002, 0.0)
        self.lateral_pid = PIDController(0.14, 0.0002, 0.0)

    # Update the executor at some intervals to steer the car in the desired direction
    def update(self, time_elapsed):
        status = self.knowledge.get_status()
        if status == Status.DRIVING or status == Status.HEALING:
            dest = self.knowledge.get_current_destination()
            self.update_control(dest, time_elapsed)
        if status == Status.CRASHED:
            self.handle_crash()

    def handle_crash(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self.vehicle.apply_control(control)

    def update_control(self, destination, delta_time):
        self.vehicle.get_world().debug.draw_string(
            destination,
            "*",
            draw_shadow=True,
            color=carla.Color(r=0, g=255, b=0),
            life_time=600.0,
            persistent_lines=True,
        )

        # Get vehicle's current transform and location
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation

        # Convert vehicle's current location and destination into numpy arrays
        vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])
        destination_pos = np.array([destination.x, destination.y])

        # Calculate the vector from the vehicle to the destination
        vector_to_destination = destination_pos - vehicle_pos
        distance_to_destination = np.linalg.norm(vector_to_destination)
        vector_to_destination_normalized = vector_to_destination / distance_to_destination

        # Get vehicle's forward vector
        forward_vector = np.array([
            np.cos(np.radians(vehicle_rotation.yaw)),
            np.sin(np.radians(vehicle_rotation.yaw))
        ])

        # Dot product and cross product to find the angle to the destination
        dot_product = np.dot(forward_vector, vector_to_destination_normalized)
        cross_product = np.cross(forward_vector, vector_to_destination_normalized)

        # Calculate steering angle (angle between vehicle's forward direction and destination direction)
        angle_to_destination = np.arccos(np.clip(dot_product, -1.0, 1.0))
        steer_direction = np.sign(cross_product)

        # Calculate speed deviation (longitudinal control)
        current_speed = np.linalg.norm([
            self.vehicle.get_velocity().x,
            self.vehicle.get_velocity().y
        ])
        target_speed = self.knowledge.get_target_speed()
        speed_error = target_speed - current_speed

        throttle_brake = self.longitudinal_pid.compute(speed_error, delta_time)
        throttle = max(0.0, throttle_brake)
        brake = max(0.0, -throttle_brake)

        # Calculate steering deviation (lateral control)
        angular_error = steer_direction * angle_to_destination
        steer = self.lateral_pid.compute(angular_error, delta_time)

        # Create vehicle control object
        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.steer = np.clip(steer, -1.0, 1.0)
        control.hand_brake = False

        if self.knowledge.get_status() == Status.HEALING:
            control.throttle = min(control.throttle, 0.1)  # Reduce speed when healing

        self.vehicle.apply_control(control)

# Planner is responsible for creating a plan for moving around
# In our case it creates a list of waypoints to follow so that vehicle arrives at destination
# Alternatively this can also provide a list of waypoints to try avoid crashing or 'uncrash' itself
class Planner(object):
    def __init__(self, knowledge, vehicle):
        self.knowledge = knowledge
        self.vehicle = vehicle
        self.path = deque([])

    # Create a map of waypoints to follow to the destination and save it
    def make_plan(self, source, destination):
        self.path = self.build_path(source, destination)
        self.update_plan()
        self.knowledge.update_destination(self.get_current_destination())

    # Function that is called at time intervals to update ai-state
    def update(self, time_elapsed):
        self.update_plan()
        self.knowledge.update_destination(self.get_current_destination())
        obstacles = self.knowledge.get_obstacles()
        if obstacles is None:
            obstacles = []

    # Update internal state to make sure that there are waypoints to follow and that we have not arrived yet
    def update_plan(self):
        if len(self.path) == 0:
            return

        if self.knowledge.arrived_at(self.path[0]):
            self.path.popleft()

        if len(self.path) == 0:
            self.knowledge.update_status(Status.ARRIVED)

    def is_space_available(self, location):
        # Implement logic to check if the location is free from obstacles
        for obstacle in self.knowledge.get_obstacles():
            if location.distance(obstacle) < 3.0:  # Adjust the distance threshold
                return False
        return True

    def calculate_detour(self, vehicle_location, obstacle_location):

        DETOUR_THRESHOLD = 0.8
        # Calculate the direction vector from vehicle to obstacle
        direction_to_obstacle = obstacle_location - vehicle_location
        distance_to_obstacle = direction_to_obstacle.length()

        # Normalize the direction vector
        direction_to_obstacle /= distance_to_obstacle

        # Perpendicular vectors for left and right directions
        left_direction = carla.Location(
            -direction_to_obstacle.y, direction_to_obstacle.x, 0
        )
        right_direction = carla.Location(
            direction_to_obstacle.y, -direction_to_obstacle.x, 0
        )

        # Check space on the left
        left_detour = (
            vehicle_location + left_direction * DETOUR_THRESHOLD
        )  # Adjust the detour distance
        if self.is_space_available(left_detour):
            return left_detour

        # Check space on the right
        right_detour = (
            vehicle_location + right_direction * DETOUR_THRESHOLD
        )  # Adjust the detour distance
        if self.is_space_available(right_detour):
            return right_detour

        # If obstacle is directly in front, try going around it
        front_left_detour = (
            vehicle_location
            + direction_to_obstacle * DETOUR_THRESHOLD
            + left_direction * DETOUR_THRESHOLD
        )
        if self.is_space_available(front_left_detour):
            return front_left_detour

        front_right_detour = (
            vehicle_location
            + direction_to_obstacle * DETOUR_THRESHOLD
            + right_direction * DETOUR_THRESHOLD
        )
        if self.is_space_available(front_right_detour):
            return front_right_detour

        # If no detour is possible, return None
        return None

    # get current destination
    def get_current_destination(self):
        status = self.knowledge.get_status()
        # if we are driving, then the current destination is next waypoint
        if status == Status.DRIVING:
            # n_distance = self.path[0].distance(self.knowledge.get_location())
            # print("Distance To: ", n_distance)
            # TODO: Take into account traffic lights and other cars
            self.knowledge.update_data("target_speed", 8)
            if self.path is None or len(self.path) == 0:
                return self.knowledge.get_location()
            return self.path[0]
        if status == Status.ARRIVED:
            self.knowledge.update_data("target_speed", 0)
            return self.knowledge.get_location()
        if status == Status.HEALING:
            self.knowledge.update_data("target_speed", 0.5)
            # Add new destinations if new obstacles are detected
            obstacles = self.knowledge.get_obstacles()
            for obstacle_location in obstacles:
                vehicle_location = self.knowledge.get_location()
                # print(obstacle)
                if (
                    vehicle_location.distance(obstacle_location) < 3.0
                ):  # Check for nearby obstacles
                    detour_destination = self.calculate_detour(
                        vehicle_location, obstacle_location
                    )
                    if detour_destination:
                        self.path.appendleft(detour_destination)

                    else:
                        return self.knowledge.get_location()

            # TODO: Implement crash handling. Probably needs to be done by following waypoint list to exit the crash site.
            # Afterwards needs to remake the path.
            # self.knowledge.update_status(Status.DRIVING
            if self.path is None or len(self.path) == 0:
                return self.knowledge.get_location()
            return self.path[0]
        if status == Status.CRASHED:
            # TODO: implement function for crash handling, should provide map of wayoints to move towards to for exiting crash state.
            # You should use separate waypoint list for that, to not mess with the original path.
            return self.knowledge.get_location()
        # otherwise destination is same as current position
        return self.knowledge.get_location()

    # TODO: Implementation
    # TODO: create path of waypoints from source to destination

    def build_path(self, source, destination):
        self.path = deque([])

        world = self.vehicle.get_world()
        world_map = world.get_map()

        # Get Waypoints from source to destination using Carla's map API
        source_waypoint = world_map.get_waypoint(source.location)
        destination_waypoint = world_map.get_waypoint(destination)

        # Generating Waypoints with less than 5 meters interval
        current_waypoint = source_waypoint
        count = 0
        PATH_THRESHOLD = source.location.distance(destination) / 5 + 10

        while current_waypoint.transform.location.distance(destination) > 5.01:
            next_waypoints = current_waypoint.next(5.0)

            if len(next_waypoints) == 0:
                break

            next_waypoint = next_waypoints[0]

            """

            # Check if lane change is needed
            if next_waypoint.lane_change == carla.LaneChange.Right:
                possible_waypoint = next_waypoint.get_right_lane()
                if possible_waypoint and possible_waypoint.lane_type == carla.LaneType.Driving:
                    next_waypoint = possible_waypoint
            """

            """
            elif next_waypoint.lane_change == carla.LaneChange.Left:
                possible_waypoint = next_waypoint.get_left_lane()
                if possible_waypoint and possible_waypoint.lane_type == carla.LaneType.Driving:
                    next_waypoint = possible_waypoint
            
            """

            self.path.append(next_waypoint.transform.location)
            world.debug.draw_string(
                next_waypoint.transform.location,
                "^",
                draw_shadow=True,
                color=carla.Color(r=255, g=0, b=0),
                life_time=600.0,
                persistent_lines=True,
            )

            current_waypoint = next_waypoint
            count += 1
            if count > PATH_THRESHOLD:
                break

        self.path.append(destination)
        return self.path
