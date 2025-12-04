#!/usr/bin/env python3

from std_msgs.msg import String, Int16
import rospy
from geometry_msgs.msg import Twist
import time


class robot_controller:

    def __init__(self):
        self.text_reader = rospy.Subscriber("/text_read", String, self.send_clue)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        self.pos_pub = rospy.Publisher("/update_pos", Int16, queue_size=10)

        self.sign_pos = 1

        time.sleep(3)

        self.start_timer()


    def start_timer(self):
        self.score_pub.publish("team9,pswrd,0,0")
        self.pos_pub.publish(0)
        print("TIMER STARTED!!")

    def stop_timer(self):
        self.score_pub.publish("team9,pswrd,-1,0")

    def send_clue(self, msg):
        self.score_pub.publish(f"team9,pswrd,{self.sign_pos},{msg.data}")
        # if(self.sign_pos == 5):# Skip sign 6 for now
        #     self.sign_pos += 1
        self.pos_pub.publish(self.sign_pos)
        self.sign_pos+=1
        print(f"Next sign: {self.sign_pos}")

        if(self.sign_pos == 8):
            time.sleep(5)
            self.stop_timer()
        



rospy.init_node("main_node")
robot = robot_controller()
rospy.spin()