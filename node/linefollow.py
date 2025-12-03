#!/usr/bin/env python3
import math
import rospy
import numpy as np
import cv2
import traceback

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
        self.Kp = float(rospy.get_param("~Kp", 2.5))
        self.Ki = float(rospy.get_param("~Ki", 0.00))
        self.Kd = float(rospy.get_param("~Kd", 0.1))

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

        self.roi_frac = float(rospy.get_param("~roi_frac", 0.50))       # use bottom 50%
        self.l_thresh = int(rospy.get_param("~l_thresh", 170))          # threshold on enhanced L channel
        self.kernel_size = int(rospy.get_param("~kernel_size", 5))

        self.img_count = 0

        # Edge memory
        self.left_x = None
        self.right_x = None
        self.last_left_t = 0.0
        self.last_right_t = 0.0
        self.edge_timeout = float(rospy.get_param("~edge_timeout", 0.6))  # seconds keep edge "alive"

        # Association gating
        self.assoc_gate_px = float(rospy.get_param("~assoc_gate_px", 120.0))  # max jump allowed for matching

        # Smoothing (separate from last_center_x)
        self.edge_alpha = float(rospy.get_param("~edge_alpha", 0.25))


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

    def _inner_edge_x(self, contour, side, rw):
        """
        Returns the x-position of the *inner edge* of a boundary contour.
        - side="left": take the RIGHTMOST pixels of that contour (inner edge of left boundary)
        - side="right": take the LEFTMOST pixels of that contour (inner edge of right boundary)
        """
        pts = contour.reshape(-1, 2)
        xs = pts[:, 0].astype(np.float32)

        if xs.size == 0:
            return None

        if side == "left":
            return float(np.percentile(xs, 95))
        elif side == "right":
            return float(np.percentile(xs, 5))
        else:
            # fallback: median
            return float(np.median(xs))


    def _detect_lane_center(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # 1) ROI: bottom fraction of the image
        roi = frame_bgr[int(h/2):, :]
        rh, rw = roi.shape[:2]

        # 2) HLS -> take L channel
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        # 3) CLAHE on L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        # 4) Blur
        blurred = cv2.GaussianBlur(enhanced_l, (5, 13), 200)

        # 5) Threshold (binary)_, thresh
        _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

        # 6) Morph open to remove speckle
        k = max(3, self.kernel_size | 1)
        kernel = np.ones((k, k), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Optional: thicken a bit so contours form nicely
        clean = cv2.dilate(clean, kernel, iterations=1)
        mask_vis = clean

        # 7) Contours (external is usually better here than TREE)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if self.debug_view:
                self._debug_show(roi, clean, None, conf=0.0)
            return None, 0.0

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        contours = [
            contour for contour in contours
            if cv2.contourArea(contour) > 5000
        ]

        overlay = roi.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # draw the remembered edges if they exist
        if self.left_x is not None:
            cv2.line(overlay, (int(self.left_x), 0), (int(self.left_x), rh-1), (255, 0, 0), 2)   # blue
        if self.right_x is not None:
            cv2.line(overlay, (int(self.right_x), 0), (int(self.right_x), rh-1), (0, 255, 255), 2) # yellow

        t = rospy.Time.now().to_sec()

        # Extract a representative x for each contour (use median x as a neutral feature for matching)
        cxs = []
        areas = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 80:
                continue
            pts = c.reshape(-1, 2)
            x_med = float(np.median(pts[:, 0]))
            cxs.append((c, x_med, area))
        cxs.sort(key=lambda z: z[2], reverse=True)

        """
        if len(cxs) == 0:
            if self.debug_view:
                self._debug_show(roi, clean, None, conf=0.0)
            return None, 0.0
        """

        # Helper: smooth update
        def ema(prev, new, a):
            return new if prev is None else (1 - a) * prev + a * new

        # Case A: two contours -> assign left/right by x and update memory
        if len(cxs) >= 2:
            c1, x1, _ = cxs[0]
            c2, x2, _ = cxs[1]
            if x1 <= x2:
                left_c, right_c = c1, c2
            else:
                left_c, right_c = c2, c1

            left_inner = self._inner_edge_x(left_c, "left", rw)
            right_inner = self._inner_edge_x(right_c, "right", rw)

            self.left_x = ema(self.left_x, left_inner, self.edge_alpha)
            self.right_x = ema(self.right_x, right_inner, self.edge_alpha)
            self.last_left_t = t
            self.last_right_t = t

            width = self.right_x - self.left_x
            if width > 10:
                self.lane_width_px = ema(self.lane_width_px, width, self.lane_width_alpha)

            center_x = 0.5 * (self.left_x + self.right_x)
            conf = 1.0

        # Case B: one contour -> ASSOCIATION using memory (prevents wrong-way turning)
        else:
            c, x_med, _ = cxs[0]

            left_alive = (self.left_x is not None) and ((t - self.last_left_t) < self.edge_timeout)
            right_alive = (self.right_x is not None) and ((t - self.last_right_t) < self.edge_timeout)

            # Decide whether this contour is left or right by nearest previous edge
            side = None
            if left_alive and right_alive:
                if abs(x_med - self.left_x) <= abs(x_med - self.right_x):
                    side = "left"
                else:
                    side = "right"
            elif left_alive:
                side = "left"
            elif right_alive:
                side = "right"
            else:
                # no memory yet -> last-resort fallback
                side = "left" if x_med < (rw / 2.0) else "right"

            if side == "left":
                left_inner = self._inner_edge_x(c, "left", rw)
                # Gate out obvious mis-associations (prevents sudden flip)
                if left_alive and abs(left_inner - self.left_x) > self.assoc_gate_px:
                    return None, 0.0

                self.left_x = ema(self.left_x, left_inner, self.edge_alpha)
                self.last_left_t = t

                # Predict right edge
                if right_alive:
                    right_pred = self.right_x
                elif self.lane_width_px is not None:
                    right_pred = self.left_x + self.lane_width_px
                else:
                    return None, 0.0

                center_x = 0.5 * (self.left_x + right_pred)
                conf = 0.6

            else:  # side == "right"
                right_inner = self._inner_edge_x(c, "right", rw)
                if right_alive and abs(right_inner - self.right_x) > self.assoc_gate_px:
                    return None, 0.0

                self.right_x = ema(self.right_x, right_inner, self.edge_alpha)
                self.last_right_t = t

                if left_alive:
                    left_pred = self.left_x
                elif self.lane_width_px is not None:
                    left_pred = self.right_x - self.lane_width_px
                else:
                    return None, 0.0

                center_x = 0.5 * (left_pred + self.right_x)
                conf = 0.6

        # Clamp and smooth center_x (optional, but helps)
        center_x = float(np.clip(center_x, 0, rw - 1))
        if self.last_center_x is None:
            self.last_center_x = center_x
        else:
            self.last_center_x = 0.7 * self.last_center_x + 0.3 * center_x
        center_x = float(self.last_center_x)

        if self.debug_view:
            # center line (red)
            cv2.line(overlay, (int(center_x), 0), (int(center_x), rh-1), (0, 0, 255), 2)

            # text: conf + side memory status
            tnow = rospy.Time.now().to_sec()
            left_alive = (self.left_x is not None) and ((tnow - self.last_left_t) < self.edge_timeout)
            right_alive = (self.right_x is not None) and ((tnow - self.last_right_t) < self.edge_timeout)
            cv2.putText(overlay, f"conf={conf:.2f}  L_alive={int(left_alive)} R_alive={int(right_alive)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("mask", mask_vis)
            cv2.imshow("overlay", overlay)
            cv2.waitKey(1)

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
        self.img_count += 1
        rospy.loginfo_throttle(
            1.0,
            f"image_cb alive: count={self.img_count} stamp={msg.header.stamp.to_sec():.3f} enc={msg.encoding} size={msg.width}x{msg.height}"
        )
        
        # Convert ROS -> OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
        try:
            center_x, conf = self._detect_lane_center(frame)
        except Exception as e:
            rospy.logerr_throttle(1.0, f"_detect_lane_center crashed: {e}\n{traceback.format_exc()}")
            return


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
