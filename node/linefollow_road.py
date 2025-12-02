#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class LineFollowerPID:
    def __init__(self):
        rospy.init_node('main_node', anonymous=False)

        # ---- Parameters
        self.image_topic   = '/B1/rrbot/camera1/image_raw'
        self.cmd_vel_topic = '/B1/cmd_vel'
        self.thresh        = 130    #Grayscale threshold value
        self.v_forward     = 0.5

        # PID gains
        self.Kp = 3.0
        self.Ki = 0.01
        self.Kd = 1.5

        # ---- ROS I/O ----
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.cb, queue_size=1, buff_size=2**24)

        self.started = False
        self.finished = False
        self.start_time = None

        # PID state
        self.prev_err = 0.0
        self.integral = 0.0

        rospy.loginfo("main_node: sub=%s  pub=%s", self.image_topic, self.cmd_vel_topic)

    def cb(self, msg: Image):
        # --- Start timer & score on first frame ---
        if not self.started:
            self.started = True
            self.start_time = rospy.Time.now()
            rospy.loginfo("Starting score & timer")
            self.score_pub.publish("team9,pswrd,0,0")  # start run

        # --- Check elapsed time ---
        elapsed = (rospy.Time.now() - self.start_time).to_sec()

        if elapsed >= 10.0:
            # Time is up: stop robot & stop score ONCE
            if not self.finished:
                rospy.loginfo("10 seconds elapsed, stopping robot & score")
                stop = Twist()
                self.cmd_pub.publish(stop)
                self.score_pub.publish("team9,pswrd,-1,0")  # end run
                self.finished = True
            return  # ignore further control

        # If we haven't started or already finished, nothing to do
        if self.finished:
            return

        # --- Normal line-following control below ---

        # Convert ROS → OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("cv_bridge: %s", e)
            return

        h, w, _ = frame.shape

        # Bottom-row threshold logic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY_INV)

        bottom = mask[-1, :]           # last row
        cols = np.where(bottom > 0)[0] # white pixels on bottom row

        if cols.size == 0:
            # Lost the line: stop safely
            rospy.logwarn("Line lost - stopping")
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        center_x = int(cols.mean())

        # Error in [-1, 1]
        err = (center_x - (w / 2.0)) / (w / 2.0)

        # PID update
        self.integral += err
        deriv = err - self.prev_err

        wz = -(self.Kp*err + self.Ki*self.integral + self.Kd*deriv)

        # Publish command
        cmd = Twist()
        cmd.linear.x = self.v_forward
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

        self.prev_err = err

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    LineFollowerPID().run()