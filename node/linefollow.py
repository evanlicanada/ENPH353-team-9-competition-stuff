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
        self.Kp = float(rospy.get_param("~Kp", 1.75))
        self.Ki = float(rospy.get_param("~Ki", 0.00))
        self.Kd = float(rospy.get_param("~Kd", 0.2))

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

        self.lane_width_px = None
        self.lane_width_alpha = rospy.get_param("~lane_width_alpha", 0.2)  # smoothing
        self.min_side_pixels = rospy.get_param("~min_side_pixels", 40)     # per-side threshold
        self.last_center_x = None


        rospy.loginfo("DualTrackLineFollower:")
        rospy.loginfo("  image: %s", self.image_topic)
        rospy.loginfo("  cmd:   %s", self.cmd_topic)

    def _center_from_xs(self, xs, rw):
        """
        Given x-pixels from a band, compute (center_x, conf, left_inner, right_inner)
        using inner edges to avoid riding the boundary.
        """
        if xs is None or xs.size == 0:
            return None, 0.0, None, None

        mid = rw / 2.0
        xs_left  = xs[xs < mid]
        xs_right = xs[xs >= mid]

        left_ok  = xs_left.size  >= self.min_side_pixels
        right_ok = xs_right.size >= self.min_side_pixels

        left_inner = right_inner = None

        if left_ok:
            # inner edge of left boundary is its RIGHTMOST pixels
            left_inner = float(np.percentile(xs_left, 95))
        if right_ok:
            # inner edge of right boundary is its LEFTMOST pixels
            right_inner = float(np.percentile(xs_right, 5))

        # Update lane width when both inner edges exist
        if left_ok and right_ok:
            width = right_inner - left_inner
            if width > 10:
                if self.lane_width_px is None:
                    self.lane_width_px = width
                else:
                    a = self.lane_width_alpha
                    self.lane_width_px = (1 - a) * self.lane_width_px + a * width

            center = 0.5 * (left_inner + right_inner)
            conf = 1.0
            return center, conf, left_inner, right_inner

        # If one side missing, infer using learned lane width
        if self.lane_width_px is not None:
            if left_ok:
                center = left_inner + 0.5 * self.lane_width_px
                return center, 0.6, left_inner, None
            if right_ok:
                center = right_inner - 0.5 * self.lane_width_px
                return center, 0.6, None, right_inner

        return None, 0.0, None, None

    # ---------- Vision ----------
    def _detect_lane_center(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # ROI crop
        y0 = int(h * self.roi_start)
        roi = frame_bgr[y0:h, :]
        rh, rw = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hls  = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)

        # Contrast normalize
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_c = clahe.apply(gray)

        # A) White paint (paved road)
        lower = (0, self.white_L_min, 0)
        upper = (255, 255, self.white_S_max)
        mask_white = cv2.inRange(hls, lower, upper)

        # B) Top-hat + Otsu (faint thin features)
        k = max(3, self.tophat_ksize | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat = cv2.morphologyEx(gray_c, cv2.MORPH_TOPHAT, kernel)
        _, mask_tophat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = cv2.bitwise_or(mask_white, mask_tophat)

        # Clean
        mk = max(3, self.morph_ksize | 1)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2)

        # --- KEY CHANGE 1: thicken thin lines so xs.size isn't tiny ---
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

        # --- Try multiple bands and COMPUTE a center for each (lookahead) ---
        bands = [
            ("top",    int(rh * 0.40), int(rh * 0.60)),  # look-ahead
            ("mid",    int(rh * 0.60), int(rh * 0.80)),
            ("bottom", int(rh * 0.80), rh),
        ]

        centers = []
        debug_band_boxes = []

        for name, yA, yB in bands:
            band = mask[yA:yB, :]
            _, xs = np.where(band > 0)

            # Require some minimum pixels in this band before using it
            if xs.size < max(20, int(self.min_pixels * 0.25)):
                continue

            c, conf, l_in, r_in = self._center_from_xs(xs, rw)
            if c is None:
                continue

            centers.append((name, c, conf))
            debug_band_boxes.append((yA, yB))

        if not centers:
            if self.debug_view:
                self._debug_show(roi, mask, None, conf=0.0)
            return None, 0.0
        
        # --------- NEW: pick a stable candidate (prevents “wrong line lock”) ---------
        # Prefer mid band, and prefer solutions close to the previous center.
        # This avoids the "top band latches onto crossing/branch" problem.

        # Gate candidates: reject huge jumps compared to last center
        if self.last_center_x is not None:
            gate = 0.25 * rw  # allow up to 25% image width jump
            gated = [(n, c, cf) for (n, c, cf) in centers if abs(c - self.last_center_x) < gate]
            if gated:
                centers = gated

        band_penalty = {"mid": 0.0, "bottom": 20.0, "top": 45.0}  # px penalty (top is least reliable in that turn)

        if self.last_center_x is None:
            # no history yet: just prefer mid band
            best_name, center_x, conf = min(centers, key=lambda t: band_penalty.get(t[0], 30.0))
        else:
            # choose closest-to-last plus band penalty
            best_name, center_x, conf = min(
                centers,
                key=lambda t: abs(t[1] - self.last_center_x) + band_penalty.get(t[0], 30.0)
            )

        # Smooth center to reduce jitter
        if self.last_center_x is None:
            self.last_center_x = center_x
        else:
            self.last_center_x = 0.7 * self.last_center_x + 0.3 * center_x

        center_x = float(np.clip(self.last_center_x, 0, rw - 1))

        # Optional debug: show chosen band name
        if self.debug_view:
            vis = roi.copy()
            for (yA, yB) in debug_band_boxes:
                cv2.rectangle(vis, (0, yA), (rw-1, yB-1), (0, 255, 0), 2)
            cv2.putText(vis, f"chosen={best_name}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            self._debug_show(vis, mask, center_x, conf)

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

        if center_x is None:
            rospy.logwarn_throttle(1.0, "No lane detected -> publishing STOP")
        else:
            rospy.loginfo_throttle(1.0, f"Lane detected: conf={conf:.2f}, center_x={center_x:.1f}")


        cmd = Twist()

        # If uncertain: slow down or stop
        if center_x is None:
            rospy.logwarn_throttle(1.0, f"Stopping. center_x={center_x} conf={conf:.2f}")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.integral = 0.0
            self.prev_err = 0.0
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

        rospy.loginfo_throttle(1.0, f"cmd v={v:.2f} wz={wz:.2f} conf={conf:.2f}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    DualTrackLineFollower().run()
