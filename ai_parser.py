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

        # create depth sensor

    # convert lidar information into an np array and send it to knowledge
    def lidar_callback(self, point_cloud):
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype("f4")))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        #world = self.vehicle.get_world()
        #for data_point in data:
            #location = carla.Location(float(-data_point[0]), float(-data_point[1]), float(data_point[2])) + self.vehicle.get_transform().location
            #world.debug.draw_string(location,".",draw_shadow=False,color=carla.Color(r=255, g=0, b=255),life_time=1,persistent_lines=True)

        self.knowledge.update_data("lidar_data", data)

    def setup_lidar(self, world):

        # setup lidar blueprints and attributes
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(30))
        lidar_bp.set_attribute("noise_stddev", str(0.1))
        lidar_bp.set_attribute("upper_fov", str(25.0))
        lidar_bp.set_attribute("lower_fov", str(-21.4))
        lidar_bp.set_attribute("channels", str(32.0))
        lidar_bp.set_attribute("points_per_second", str(20000))
        lidar_bp.set_attribute("rotation_frequency", str(20.0))
        lidar_transform = carla.Transform(carla.Location(z=1.95))

        # create lidar sensor
        self.lidar_sensor = world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        self.lidar_sensor.listen(self.lidar_callback)

    def check_vehicle_traffic_light(self):
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                return True
        return False

    # Function that is called at time intervals to update ai-state
    def update(self, time_elapsed):
        # Update the position of vehicle into knowledge
        self.knowledge.update_data("location", self.vehicle.get_transform().location)
        self.knowledge.update_data("rotation", self.vehicle.get_transform().rotation)

        #update the traffic state
        self.knowledge.update_data("is_at_traffic_light", self.check_vehicle_traffic_light())

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
        self.obstacle_high_threshold = 4.7
        self.obstacle_low_threshold = 1.5

    def detect_obstacle(self, data):
        distance = self.get_distance(data)
        if self.obstacle_low_threshold < distance < self.obstacle_high_threshold:  # Example threshold for obstacles
            # print('Obstacle detected. : ', data)
            obstacle_location = carla.Location(float(-data[0]), float(-data[1]), float(data[2]))
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
    
    def analyze_obstacles(self):
        obstacles = self.knowledge.get_obstacles()
        if obstacles is None:
            return
        for obstacle in obstacles:
            world = self.vehicle.get_world()
            true_location = self.vehicle.get_transform().location + obstacle
            world.debug.draw_string(
                true_location,
                "O",
                draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255),
                life_time=0.3,
                persistent_lines=True,
            )
        return 
    
    # Analyse Traffic Light
    def analyse_traffic_light(self):
        return self.knowledge.get_traffic_light_state()


    def analyse_lidar(self):
        lidar_data = self.knowledge.get_lidar_data()

        if lidar_data is None:
            print("Lidar data is None")
            return

        obstacles = []
        for pdata in lidar_data:
            obstacle = self.detect_obstacle(pdata)
            if obstacle is not None:
                obstacles.append(obstacle)

        if len(obstacles) == 0:
            self.knowledge.update_status(data.Status.DRIVING)
        else:
            self.knowledge.update_data("obstacles", obstacles)
            self.knowledge.update_status(data.Status.HEALING)

    # Function that is called at time intervals to update ai-state
    def update(self, time_elapsed):
        if self.knowledge.get_status() == data.Status.CRASHED:
            return
        if self.analyse_traffic_light():
            self.knowledge.update_status(data.Status.REDLIGHT)
            print("Traffic Light Detected, current Status: ", self.knowledge.get_status())
            return
        self.analyse_lidar()
        #self.analyze_obstacles()
        print("Analysis from Monitored Knowledge, Current Status: ", self.knowledge.get_status())
        return
