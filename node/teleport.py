#!/usr/bin/env python3
import math
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class Teleport: 
    def __init__(self, model_name="B1", reference_frame="world"):

        self.model_name = model_name
        self.reference_frame = reference_frame

    def teleport(self, x, y, z, yaw):
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = self.reference_frame

        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z

        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = math.sin(yaw * 0.5)
        state.pose.orientation.w = math.cos(yaw * 0.5)

        state.twist.linear.x  = 0.0
        state.twist.linear.y  = 0.0
        state.twist.linear.z  = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        try:
            resp = self.set_state(state)
            if not resp.success:
                rospy.logwarn("Teleport failed: %s", resp.status_message)
        except Exception as e:
            rospy.logerr("Failed to set model state: %s", e)