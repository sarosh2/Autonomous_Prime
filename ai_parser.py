#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
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
    
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)

    world = self.vehicle.get_world()
    bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    self.lane_detector = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
    self.lane_detector.listen(lambda event: Monitor._on_invasion(weak_self, event))

    #create LIDAR sensor
    self.setup_lidar(world)
    
    #create depth sensor

  #convert lidar information into an np array and send it to knowledge
  def lidar_callback(self, point_cloud):
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    self.knowledge.update_data('lidar_data', data)


  def setup_lidar(self, world):

    #setup lidar blueprints and attributes
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range',str(100))
    lidar_bp.set_attribute('noise_stddev',str(0.1))
    lidar_bp.set_attribute('upper_fov',str(15.0))
    lidar_bp.set_attribute('lower_fov',str(-25.0))
    lidar_bp.set_attribute('channels',str(64.0))
    lidar_bp.set_attribute('points_per_second',str(500000))
    lidar_bp.set_attribute('rotation_frequency',str(20.0))
    lidar_transform = carla.Transform(carla.Location(z=2))

    #create lidar sensor
    self.lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
    self.lidar_sensor.listen(self.lidar_callback)

  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    # Update the position of vehicle into knowledge
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)

  @staticmethod
  def _on_invasion(weak_self, event):
    self = weak_self()
    if not self:
      return
    self.knowledge.update_data('lane_invasion',event.crossed_lane_markings)

# Analyser is responsible for parsing all the data that the knowledge has received from Monitor and turning it into something usable
# TODO: During the update step parse the data inside knowledge into information that could be used by planner to plan the route
class Analyser(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge
    self.is_lidar_below_threshold = False
    self.collision_threshold = 1.0
    self.healing_threshold = 20.0

  def detect_collision(self, data):
    # Implement collision detection logic
    if data[0]**2 + data[1]**2 + data[2]**2 < self.collision_threshold:  #threshold
      return True
    return False

  def detect_obstacle(self, data):
    distance = data[0]**2 + data[1]**2 + data[2]**2
    if distance < self.healing_threshold:  # Example threshold for obstacles
      obstacle_location = carla.Location(x=data[0], y=data[1], z=data[2])
      return obstacle_location
    else:
      return None

  def analyse_lidar(self):
    lidar_data = self.knowledge.get_lidar_data()

    if lidar_data is None:
      print('Lidar data is None')
      return

    obstacles = []
    
    for pdata in lidar_data:
      if self.detect_collision(pdata):
        self.knowledge.update_status(data.Status.CRASHED)
        return
      else:
        obstacle = self.detect_obstacle(pdata)
        if obstacle is not None:
          self.knowledge.update_status(data.Status.HEALING)
          obstacles.append(obstacle)
          
    if len(obstacles) == 0:
      self.knowledge.update_status(data.Status.DRIVING)
    else:
      self.knowledge.update_data('obstacles', obstacles)


  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    print('Analyser update called')
    self.analyse_lidar()
    print('Lidar Data from Knowledge: ', self.knowledge.get_status())
    return
