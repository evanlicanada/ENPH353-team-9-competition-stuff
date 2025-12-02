#!/usr/bin/env python3
import math
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class DualTrackLineFollower:
    def __init__(self):
        rospy.init_node("dual_track_line_follower", anonymous=False)

        # ---------------- Parameters ----------------
        self.image_topic = rospy.get_param("~image_topic", "/B1/rrbot/camera1/image_raw")
        self.cmd_topic = rospy.get_param("~cmd_vel_topic", "/B1/cmd_vel")

        # Speed
        self.v_nominal = float(rospy.get_param("~v_nominal", 0.5)) # m/s
        self.v_min = float(rospy.get_param("~v_min", 0.08)) # when uncertain
        self.wz_max = float(rospy.get_param("~wz_max", 2.5)) # rad/s clamp

        # ROI and band selection
        self.roi_start = float(rospy.get_param("~roi_start", 0.55)) # start y fraction
        self.band_start = float(rospy.get_param("~band_start", 0.70)) # within ROI, start y fraction (bottom band)

        # PID gains (start conservative; tune)
        self.Kp = float(rospy.get_param("~Kp", 2.2))
        self.Ki = float(rospy.get_param("~Ki", 0.05))
        self.Kd = float(rospy.get_param("~Kd", 0.6))

        # Integral anti-windup
        self.i_max = float(rospy.get_param("~i_max", 1.0))

        # Mask thresholds (HLS white detector) - should work in Gazebo
        self.white_L_min = int(rospy.get_param("~white_L_min", 170))
        self.white_S_max = int(rospy.get_param("~white_S_max", 90))

        # Morphology kernel sizes
        self.tophat_ksize = int(rospy.get_param("~tophat_ksize", 21)) # try 15..31
        self.morph_ksize = int(rospy.get_param("~morph_ksize", 9)) # try 5..11

        # Requirements for detection
        self.min_pixels = int(rospy.get_param("~min_pixels", 200))
        self.min_conf = float(rospy.get_param("~min_conf", 0.20))

        # Debug view (OpenCV windows). Set false if headless.
        self.debug_view = bool(rospy.get_param("~debug_view", False))

        # ---------------- ROS I/O ----------------
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_cb,
                                    queue_size=1, buff_size=2**24)

        # ---------------- State ----------------
        self.prev_err = 0.0
        self.integral = 0.0
        self.prev_t = None

        rospy.loginfo("DualTrackLineFollower:")
        rospy.loginfo("  image: %s", self.image_topic)
        rospy.loginfo("  cmd:   %s", self.cmd_topic)

    # ---------- Vision ----------
    def _detect_lane_center(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # ROI crop
        y0 = int(h * self.roi_start)
        roi = frame_bgr[y0:h, :]
        rh, rw = roi.shape[:2]

        # Convert spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)

        # Contrast normalize helps faint grass markings
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_c = clahe.apply(gray)

        # A) White paint (paved road)
        # HLS channels: [H, L, S]
        lower = (0, self.white_L_min, 0)
        upper = (255, 255, self.white_S_max)
        mask_white = cv2.inRange(hls, lower, upper)

        # B) Top-hat + Otsu (faint thin features)
        k = max(3, self.tophat_ksize | 1)  # make odd
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat = cv2.morphologyEx(gray_c, cv2.MORPH_TOPHAT, kernel)
        _, mask_tophat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine
        mask = cv2.bitwise_or(mask_white, mask_tophat)

        # Clean
        mk = max(3, self.morph_ksize | 1)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2)

        # Bottom band for steering
        band_y0 = int(rh * self.band_start)
        band = mask[band_y0:rh, :]

        ys, xs = np.where(band > 0)
        if xs.size < self.min_pixels:
            if self.debug_view:
                self._debug_show(roi, mask, None, conf=0.0)
            return None, 0.0

        # Robust edges (works for two boundary lines and for “blob-like” lines)
        x_left = float(np.percentile(xs, 5))
        x_right = float(np.percentile(xs, 95))
        center_x = 0.5 * (x_left + x_right)

        # Confidence heuristic:
        # - width should not be tiny (no info) and not absurdly wide (mask exploded)
        width_norm = (x_right - x_left) / float(rw)
        width_score = np.clip((width_norm - 0.10) / 0.60, 0.0, 1.0)  # tuned heuristic
        pix_score = np.clip(xs.size / float(band.size * 0.06), 0.0, 1.0)
        conf = float(0.6 * width_score + 0.4 * pix_score)

        if self.debug_view:
            self._debug_show(roi, mask, center_x, conf)

        return center_x, conf

    def _debug_show(self, roi, mask, center_x, conf):
        # Draw center line on ROI
        vis = roi.copy()
        h, w = vis.shape[:2]
        if center_x is not None:
            cx = int(np.clip(center_x, 0, w-1))
            cv2.line(vis, (cx, 0), (cx, h-1), (0, 0, 255), 2)

        cv2.putText(vis, f"conf={conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("roi", vis)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)

    # ---------- Control ----------
    def _pid(self, err, dt):
        # integral with anti-windup
        self.integral += err * dt
        self.integral = float(np.clip(self.integral, -self.i_max, self.i_max))

        deriv = (err - self.prev_err) / dt

        u = self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        self.prev_err = err
        return u

    def image_cb(self, msg: Image):
        # Convert ROS -> OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("cv_bridge: %s", e)
            return

        h, w = frame.shape[:2]

        # dt
        t = rospy.Time.now().to_sec()
        if self.prev_t is None:
            self.prev_t = t
            return
        dt = max(1e-3, t - self.prev_t)
        self.prev_t = t

        # Detect lane center
        center_x, conf = self._detect_lane_center(frame)

        cmd = Twist()

        # If uncertain: slow down or stop
        if center_x is None or conf < self.min_conf:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # Normalized error [-1, 1]
        err = (center_x - (w / 2.0)) / (w / 2.0)

        # PID output -> angular velocity (negative sign to match your earlier convention)
        wz = -self._pid(err, dt)

        # Clamp angular velocity
        wz = float(np.clip(wz, -self.wz_max, self.wz_max))

        # Speed schedule: slower when less confident
        v = self.v_min + (self.v_nominal - self.v_min) * float(np.clip(conf, 0.0, 1.0))

        cmd.linear.x = v
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    DualTrackLineFollower().run()
