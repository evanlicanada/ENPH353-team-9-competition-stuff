#!/usr/bin/env python3
import math
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

"""
To find robot position: 
$ rosservice call /gazebo/get_model_state "{model_name: 'B1', relative_entity_name: 'world'}"

Destinations: 
1. Starting location:
rosservice call /gazebo/set_model_state "model_state:
  model_name: 'B1'
  reference_frame: 'world'
  pose:
    position: {x: 5.499995095757394, y: 2.4995765304828543, z: 0.040000462677662746}
    orientation: {x: 6.342845588236196e-07, y: -4.662894710339661e-07, z: -0.706898193114858, w: 0.7073153077449471}
  twist:
    linear:  {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}"

2. First pink line:
rosservice call /gazebo/set_model_state "model_state:
  model_name: 'B1'
  reference_frame: 'world'
  pose:
    position: {x: 0.628385129930686, y: 0.0006904961777512801, z: 0.040002067617634356}
    orientation: {x: -1.1011464673110303e-06, y: -7.734997000777081e-07, z: 0.4590277749340737, w: 0.8884219165673832}
  twist:
    linear:  {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}"

3. Second pink line: 
rosservice call /gazebo/set_model_state "model_state:
  model_name: 'B1'
  reference_frame: 'world'
  pose:
    position: {x: -3.9608819384409353, y: 0.45929834454066515, z: 0.040000687022617255}
    orientation: {x: -2.6994734507055647e-06, y: 3.77677665257361e-07, z: 0.9999987571263347, w: 0.001576622452033168}
  twist:
    linear:  {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}"

4. Third pink line: 
rosservice call /gazebo/set_model_state "model_state:
  model_name: 'B1'
  reference_frame: 'world'
  pose:
    position: {x: -4.030194411692881, y: -2.3089761660560773, z: 0.04000053984738465}
    orientation: {x: 3.886040698488489e-10, y: 2.0192185688131462e-06, z: -0.0008701127515915282, w: -0.9999996214497896}
  twist:
    linear:  {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}"
"""

class Teleport: 
    def __init__(self, model_name="B1", reference_frame="world", service="/gazebo/set_model_state"):

        self.model_name = model_name
        self.reference_frame = reference_frame

        rospy.wait_for_service(service)
        self.set_state = rospy.ServiceProxy(service, SetModelState)

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
          
    def teleport_quat(self, x, y, z, qx, qy, qz, qw):
        """Convenience: teleport using a quaternion directly (matches your saved poses)."""
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = self.reference_frame

        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z

        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw

        state.twist.linear.x = state.twist.linear.y = state.twist.linear.z = 0.0
        state.twist.angular.x = state.twist.angular.y = state.twist.angular.z = 0.0

        resp = self.set_state(state)
        # if not resp.success:
        #     rospy.logwarn("Teleport failed: %s", resp.status_message)

