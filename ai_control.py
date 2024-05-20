#!/usr/bin/env python

import glob
import os
import sys
from collections import deque
import math
import numpy as np

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import ai_knowledge as data
from ai_knowledge import Status

# Executor is responsible for moving the vehicle around
# In this implementation it only needs to match the steering and speed so that we arrive at provided waypoints
# BONUS TODO: implement different speed limits so that planner would also provide speed target speed in addition to direction
class Executor(object):
  def __init__(self, knowledge, vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    self.target_pos = knowledge.get_location()
    
  #Update the executor at some intervals to steer the car in desired direction
  def update(self, time_elapsed):
    status = self.knowledge.get_status()
    #TODO: this needs to be able to handle
    if status == Status.DRIVING:
      dest = self.knowledge.get_current_destination()
      self.update_control(dest, [1], time_elapsed)

  # TODO: steer in the direction of destination and throttle or brake depending on how close we are to destination
  # TODO: Take into account that exiting the crash site could also be done in reverse, so there might need to be additional data passed between planner and executor, or there needs to be some way to tell this that it is ok to drive in reverse during HEALING and CRASHED states. An example is additional_vars, that could be a list with parameters that can tell us which things we can do (for example going in reverse)
  def update_control(self, destination, additional_vars, delta_time):
    
    #calculate throttle and heading
    target_speed = additional_vars[0] if additional_vars else 1.0
    # Get vehicle's current transform and location
    vehicle_transform = self.vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation

    # Convert vehicle's current location and destination into numpy arrays
    vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])
    destination_pos = np.array([destination.x, destination.y])

    # Calculate the vector from the vehicle to the destination
    vector_to_destination = destination_pos - vehicle_pos
    vector_to_destination_normalized = vector_to_destination / np.linalg.norm(vector_to_destination)

    # Get vehicle's forward vector
    forward_vector = np.array([np.cos(np.radians(vehicle_rotation.yaw)), np.sin(np.radians(vehicle_rotation.yaw))])

    # Dot product and cross product to find the angle to the destination
    dot_product = np.dot(forward_vector, vector_to_destination_normalized)
    cross_product = np.cross(forward_vector, vector_to_destination_normalized)

    # Calculate steering angle (angle between vehicle's forward direction and destination direction)
    angle_to_destination = np.arccos(np.clip(dot_product, -1.0, 1.0))
    steer_direction = np.sign(cross_product)

    # Create vehicle control object
    control = carla.VehicleControl()
    control.throttle = 0.7  # You might want to adjust this based on distance to destination and current speed
    control.steer = steer_direction * (angle_to_destination / np.pi)  # Normalize steering angle to [-1, 1]
    control.brake = 0.0
    control.hand_brake = False

    # Apply the control to the vehicle
    self.vehicle.apply_control(control)

# Planner is responsible for creating a plan for moving around
# In our case it creates a list of waypoints to follow so that vehicle arrives at destination
# Alternatively this can also provide a list of waypoints to try avoid crashing or 'uncrash' itself
class Planner(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge
    self.path = deque([])

  # Create a map of waypoints to follow to the destination and save it
  def make_plan(self, source, destination):
    self.path = self.build_path(source,destination)
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination())
  
  # Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination())
  
  #Update internal state to make sure that there are waypoints to follow and that we have not arrived yet
  def update_plan(self):
    if len(self.path) == 0:
      return
    
    if self.knowledge.arrived_at(self.path[0]):
      self.path.popleft()
    
    if len(self.path) == 0:
      self.knowledge.update_status(Status.ARRIVED)
    else:
      self.knowledge.update_status(Status.DRIVING)

  #get current destination 
  def get_current_destination(self):
    status = self.knowledge.get_status()
    #if we are driving, then the current destination is next waypoint
    if status == Status.DRIVING:
      #TODO: Take into account traffic lights and other cars
      return self.path[0]
    if status == Status.ARRIVED:
      return self.knowledge.get_location()
    if status == Status.HEALING:
      #TODO: Implement crash handling. Probably needs to be done by following waypoint list to exit the crash site.
      #Afterwards needs to remake the path.
      return self.knowledge.get_location()
    if status == Status.CRASHED:
      #TODO: implement function for crash handling, should provide map of wayoints to move towards to for exiting crash state. 
      #You should use separate waypoint list for that, to not mess with the original path. 
      return self.knowledge.get_location()
    #otherwise destination is same as current position
    return self.knowledge.get_location()

  #TODO: Implementation
  def build_path(self, source, destination):
    self.path = deque([])
    self.path.append(destination)
    #TODO: create path of waypoints from source to destination
    return self.path


