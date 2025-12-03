#!/usr/bin/env python3

from std_msgs.msg import String
import rospy
from geometry_msgs.msg import Twist
import time


class robot_controller:

    def __init__(self):
        self.text_reader = rospy.Subscriber("/text_read", String, self.send_clue)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)

        self.sign_pos = 1

        time.sleep(3)

        self.start_timer()


    def start_timer(self):
        self.score_pub.publish("team9,pswrd,0,0")
        print("TIMER STARTED!!")

    def stop_timer(self):
        self.score_pub.publish("team9,pswrd,-1,0")

    def send_clue(self, msg):
        self.score_pub.publish(f"team9,pswrd,{self.sign_pos},{msg}")
        self.sign_pos+=1



rospy.init_node("main_node")
robot = robot_controller()
rospy.spin()