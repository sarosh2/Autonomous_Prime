#!/usr/bin/env python

import glob
import os
import sys
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

import weakref
import carla
import ai_knowledge as data
import numpy as np
import math


# Monitor is responsible for reading the data from the sensors and telling it to the knowledge
# TODO: Implement other sensors (lidar and depth sensors mainly)
# TODO: Use carla API to read whether car is at traffic lights and their status, update it into knowledge
class Monitor(object):
    def __init__(self, knowledge, vehicle):
        self.vehicle = vehicle
        self.knowledge = knowledge
        weak_self = weakref.ref(self)

        self.knowledge.update_data("location", self.vehicle.get_transform().location)
        self.knowledge.update_data("rotation", self.vehicle.get_transform().rotation)

        world = self.vehicle.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.lane_detector = world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_detector.listen(lambda event: Monitor._on_invasion(weak_self, event))
        # Adding a Collision Detector
        self.collision_sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(),
            attach_to=self.vehicle,
        )
        self.collision_sensor.listen(
            lambda event: self.knowledge.update_status(data.Status.CRASHED)
        )

        # create LIDAR sensor
        self.setup_lidar(world)

        # Checking traffic light state
        closest_tl = self.get_nearby_traffic_light(vehicle, world, 50000)
        # TODO check below line code implementation
        self.knowledge.update_data("traffic_light_value", self.get_traffic_light_state(closest_tl))
        # self.knowledge.memory["traffic_light_value"] = self.get_traffic_light_state(closest_tl)

        # create depth sensor

    # convert lidar information into an np array and send it to knowledge
    def lidar_callback(self, point_cloud):
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype("f4")))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.knowledge.update_data("lidar_data", data)

    def setup_lidar(self, world):

        # setup lidar blueprints and attributes
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(30))
        lidar_bp.set_attribute("noise_stddev", str(0.1))
        lidar_bp.set_attribute("upper_fov", str(15.0))
        lidar_bp.set_attribute("lower_fov", str(-25.0))
        lidar_bp.set_attribute("channels", str(64.0))
        lidar_bp.set_attribute("points_per_second", str(50000))
        lidar_bp.set_attribute("rotation_frequency", str(20.0))
        lidar_transform = carla.Transform(carla.Location(z=2))

        # create lidar sensor
        self.lidar_sensor = world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        self.lidar_sensor.listen(self.lidar_callback)

    def get_nearby_traffic_light(self, vehicle, world, distance_threshold=50):
        # Get the location and forward vector of the vehicle
        vehicle_location = vehicle.get_location()
        vehicle_transform = vehicle.get_transform()
        vehicle_forward_vector = vehicle_transform.get_forward_vector()


        # Get the world the vehicle is in
        world = vehicle.get_world()

        # Get all traffic lights in the world
        traffic_lights = world.get_actors().filter("traffic.traffic_light")

        # Find the nearest traffic light within the distance threshold and in front of the vehicle
        closest_traffic_light = None
        min_distance = distance_threshold

        for traffic_light in traffic_lights:
            # Get the location of the traffic light
            traffic_light_location = traffic_light.get_location()

            # Calculate the distance from the vehicle to the traffic light
            distance = vehicle_location.distance(traffic_light_location)

            if distance < min_distance:
                # Calculate the direction vector from the vehicle to the traffic light
                direction_vector = traffic_light_location - vehicle_location
                direction_vector = direction_vector.make_unit_vector()

                # Calculate the dot product to check if the traffic light is in front of the vehicle
                dot_product = (
                    direction_vector.x * vehicle_forward_vector.x
                    + direction_vector.y * vehicle_forward_vector.y
                    + direction_vector.z * vehicle_forward_vector.z
                )

                if dot_product > 0:  # Traffic light is in front of the vehicle
                    closest_traffic_light = traffic_light
                    min_distance = distance
        world.debug.draw_string(
                traffic_light_location,
                "^",
                draw_shadow=True,
                color=carla.Color(r=255, g=0, b=255),
                life_time=600.0,
                persistent_lines=True,
            )
        print("traffic light direction: ",self.get_facing_direction(closest_traffic_light))
        print("vehicle direction: ", self.get_facing_direction(vehicle))
        return closest_traffic_light

    def get_traffic_light_state(self, traffic_light):
        if traffic_light:
            return traffic_light.get_state()
        return None

    def get_facing_direction(self, actor):
      # Get the transform of the actor (traffic light or vehicle)
      transform = actor.get_transform()
      
      # Extract the rotation (pitch, yaw, roll)
      rotation = transform.rotation
      
      # Calculate the facing direction based on the yaw angle
      yaw = rotation.yaw
      facing_direction = carla.Vector3D(
          x=math.cos(math.radians(yaw)),
          y=math.sin(math.radians(yaw)),
          z=0
      )
      return facing_direction

    def are_directions_aligned(self, direction1, direction2, threshold=0.95):
        # Calculate the dot product of the two direction vectors
        dot_product = (direction1.x * direction2.x + 
                      direction1.y * direction2.y + 
                      direction1.z * direction2.z)
        
        # Normalize the dot product (it should be between -1 and 1)
        magnitude1 = math.sqrt(direction1.x**2 + direction1.y**2 + direction1.z**2)
        magnitude2 = math.sqrt(direction2.x**2 + direction2.y**2 + direction2.z**2)
        normalized_dot_product = dot_product / (magnitude1 * magnitude2)
        
        # Check if the directions are aligned within the given threshold
        return normalized_dot_product > threshold

    # Function that is called at time intervals to update ai-state
    def update(self, time_elapsed):
        # Update the position of vehicle into knowledge
        self.knowledge.update_data("location", self.vehicle.get_transform().location)
        self.knowledge.update_data("rotation", self.vehicle.get_transform().rotation)

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.knowledge.update_data("lane_invasion", event.crossed_lane_markings)


# Analyser is responsible for parsing all the data that the knowledge has received from Monitor and turning it into something usable
# TODO: During the update step parse the data inside knowledge into information that could be used by planner to plan the route
class Analyser(object):
    def __init__(self, knowledge, vehicle):
        self.knowledge = knowledge
        self.vehicle = vehicle
        self.is_lidar_below_threshold = False
        self.obstacle_threshold = 1.0
        self.vehicle_threshold = 10.0

    def detect_obstacle(self, data):
        distance = self.get_distance(data)
        if distance < self.obstacle_threshold:  # Example threshold for obstacles
            # print('Obstacle detected. : ', data)
            obstacle_location = carla.Location(float(data[0]), float(data[1]), float(data[2]))
            return obstacle_location
        else:
            return None
    
    def get_distance(self, pdata):
        return np.sqrt(pdata[0] ** 2 + pdata[1] ** 2)

    def is_vehicle_obstacle(self, pdata):
        world = self.vehicle.get_world()
        obstacle_location = carla.Location(float(pdata[0]), float(pdata[1]), float(pdata[2]))
        vehicles = world.get_actors().filter("vehicle.*")

        for vehicle in vehicles:
            if vehicle.id != self.vehicle.id:
              vehicle_location = vehicle.get_transform().location
              distance = obstacle_location.distance(vehicle_location)
              if distance < 2.0:
                  return True
        return False

    def analyse_lidar(self):
        lidar_data = self.knowledge.get_lidar_data()

        if lidar_data is None:
            print("Lidar data is None")
            return

        obstacles = []
        is_vehicle = False

        for pdata in lidar_data:
          
            if self.get_distance(pdata) < self.vehicle_threshold:
                #clear
                #print(" Checking for vehicle")
                if self.is_vehicle_obstacle(pdata):
                  is_vehicle = True
                  print("Vehicle detected")
                  obstacles.append(carla.Location(float(pdata[0]), float(pdata[1]), float(pdata[2])))
                  break
            obstacle = self.detect_obstacle(pdata)
            if obstacle is not None:
                obstacles.append(obstacle)

        if len(obstacles) == 0:
            self.knowledge.update_status(data.Status.DRIVING)
        else:
            self.knowledge.update_data("is_vehicle", is_vehicle)
            self.knowledge.update_data("obstacles", obstacles)
            self.knowledge.update_status(data.Status.HEALING)

    def analyze_obstacles(self):
        obstacles = self.knowledge.get_obstacles()
        if obstacles is None:
            return
        for obstacle in obstacles:
            world = self.vehicle.get_world()
            world.debug.draw_string(
                obstacle,
                "O",
                draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0),
                life_time=0.1,
                persistent_lines=True,
            )

    # Function that is called at time intervals to update ai-state
    def update(self, time_elapsed):
        if self.knowledge.get_status() == data.Status.CRASHED:
            return
        self.analyse_lidar()
        #self.analyze_obstacles()
        print("Lidar Data from Knowledge: ", self.knowledge.get_status())
        return