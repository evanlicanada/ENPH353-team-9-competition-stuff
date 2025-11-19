#!/usr/bin/env python3

from std_msgs.msg import String
import rospy
from geometry_msgs.msg import Twist
import time


class robot_controller:
    def __init__(self):
        self.twist_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)

        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        self.move = Twist()

        time.sleep(5)

        self.move.linear.x = 1

        self.score_pub.publish("team9,pswrd,0,0")
        self.twist_pub.publish(self.move)

        time.sleep(5)

        self.move.linear.x = 0.0
        self.twist_pub.publish(self.move)
        self.score_pub.publish("team9,pswrd,-1,0")



rospy.init_node("main_node")
robot = robot_controller()
rospy.spin()