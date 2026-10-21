import cv2
import numpy as np
from collections import deque

class PolynomialKalmanFilter:
    """
    Filtro de Kalman para estabilizar temporalmente las curvas polinómicas del carril:
    Estado x = [a, b, c, da, db, dc]^T (coeficientes y sus derivadas temporales).
    """
    def __init__(self, dt=1.0/30.0):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt
        self.H = np.zeros((3, 6))
        for i in range(3):
            self.H[i, i] = 1.0
        self.P = np.eye(6) * 10.0
        self.Q = np.eye(6) * 0.001
        self.Q[0, 0] = 1e-8
        self.Q[1, 1] = 1e-5
        self.Q[2, 2] = 0.5
        self.R = np.eye(3) * 0.8
        self.initialized = False
        self.frames_without_measurement = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.frames_without_measurement += 1
        return self.x[:3].flatten()

    def update(self, z):
        self.frames_without_measurement = 0
        if not self.initialized:
            self.x[:3, 0] = z
            self.initialized = True
            return z
        z_col = np.array(z).reshape((3, 1))
        y = z_col - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        return self.x[:3].flatten()


class AdvancedLaneDetector:
    """
    Detector de Carriles ADAS 2.0 (Iteración 4 - Perfeccionado).
    Características:
    - Síntesis geométrica automática en glorietas y curvas cerradas cuando ambas ventanas colapsan.
    - Doble Filtro de Kalman cinemático para predicción por inercia.
    - Downsampling acelerado a 360p con escalado analítico exacto (~4.5 ms).
    - Zonificación de peligro asfáltica (Safe / Warning / Critical).
    - Cálculo de curvatura vial (Radio en metros) y Lane Departure Warning (LDW).
    """
    def __init__(self, smoothing_frames=8):
        self.smoothing_frames = smoothing_frames
        self.kf_left = PolynomialKalmanFilter(dt=1.0/30.0)
        self.kf_right = PolynomialKalmanFilter(dt=1.0/30.0)

        self.M = None
        self.Minv = None
        self.internal_size = (640, 360)

        self.current_left_fit = None
        self.current_right_fit = None

        self.curvature_radius_m = None
        self.curvature_text = "RECTA"
        self.lane_center_offset_m = 0.0
        self.departure_warning = "CENTRAL"
        self.has_lane_lock = False
        self.learned_lane_width = 0.52  # Proporción normal del ancho de carril

    def _init_perspective_matrices(self, w_small, h_small):
        src = np.float32([
            [w_small * 0.12, h_small * 0.95],
            [w_small * 0.38, h_small * 0.65],
            [w_small * 0.62, h_small * 0.65],
            [w_small * 0.90, h_small * 0.95]
        ])
        offset = w_small * 0.20
        dst = np.float32([
            [offset, h_small],
            [offset, 0],
            [w_small - offset, 0],
            [w_small - offset, h_small]
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def color_and_gradient_filter(self, small_frame):
        hls = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        white_mask = cv2.inRange(l_channel, 175, 255)
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([38, 255, 255]))

        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel = np.abs(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobel / (np.max(abs_sobel) + 1e-5))
        sobel_mask = cv2.inRange(scaled_sobel, 30, 255)

        combined = cv2.bitwise_or(white_mask, yellow_mask)
        combined = cv2.bitwise_or(combined, sobel_mask)

        h, w = small_frame.shape[:2]
        horizon_mask = np.zeros_like(combined)
        horizon_mask[int(h * 0.45):, :] = 255
        combined = cv2.bitwise_and(combined, horizon_mask)

        return combined

    def sliding_windows(self, warped):
        h, w = warped.shape[:2]
        histogram = np.sum(warped[int(h * 0.55):, :], axis=0)
        midpoint = int(w // 2)

        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]

        leftx_base = np.argmax(left_hist)
        rightx_base = np.argmax(right_hist) + midpoint

        if histogram[leftx_base] < 60:
            leftx_base = int(w * 0.20)
        if histogram[rightx_base] < 60:
            rightx_base = int(w * 0.80)

        nwindows = 9
        window_height = int(h // nwindows)
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = int(w * 0.10)
        minpix = 35

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        left_fit = None
        right_fit = None

        if len(left_lane_inds) > 70:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            left_fit = np.polyfit(lefty, leftx, 2)

        if len(right_lane_inds) > 70:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def update_kalman_and_telemetry(self, left_fit_native, right_fit_native, h_native, w_native):
        pred_left = self.kf_left.predict()
        pred_right = self.kf_right.predict()

        measured_valid = False
        if left_fit_native is not None and right_fit_native is not None:
            y_eval_bot = h_native - 1
            y_eval_mid = int(h_native * 0.65)

            xl_b = left_fit_native[0]*y_eval_bot**2 + left_fit_native[1]*y_eval_bot + left_fit_native[2]
            xr_b = right_fit_native[0]*y_eval_bot**2 + right_fit_native[1]*y_eval_bot + right_fit_native[2]
            wb = xr_b - xl_b

            # Detección de colapso en curvas/glorietas (ambas ventanas cayeron en la misma línea)
            if abs(wb) < 0.25 * w_native:
                # Síntesis geométrica: usar la línea de mayor confianza y desplazar por learned_lane_width
                if xl_b < w_native * 0.5:  # Es la línea izquierda del camellón
                    right_fit_native = left_fit_native.copy()
                    right_fit_native[2] += self.learned_lane_width * w_native
                else:  # Es la línea derecha
                    left_fit_native = right_fit_native.copy()
                    left_fit_native[2] -= self.learned_lane_width * w_native
                
                xl_b = left_fit_native[0]*y_eval_bot**2 + left_fit_native[1]*y_eval_bot + left_fit_native[2]
                xr_b = right_fit_native[0]*y_eval_bot**2 + right_fit_native[1]*y_eval_bot + right_fit_native[2]
                wb = xr_b - xl_b

            xl_m = left_fit_native[0]*y_eval_mid**2 + left_fit_native[1]*y_eval_mid + left_fit_native[2]
            xr_m = right_fit_native[0]*y_eval_mid**2 + right_fit_native[1]*y_eval_mid + right_fit_native[2]
            wm = xr_m - xl_m

            if (0.28 * w_native < wb < 1.45 * w_native) and (0.22 * w_native < wm < 1.40 * w_native) and (wb > 0) and (wm > 0):
                measured_valid = True
                self.learned_lane_width = 0.95 * self.learned_lane_width + 0.05 * (wb / w_native)

        if measured_valid:
            self.current_left_fit = self.kf_left.update(left_fit_native)
            self.current_right_fit = self.kf_right.update(right_fit_native)
            self.has_lane_lock = True
        elif self.has_lane_lock and self.kf_left.frames_without_measurement < 25:
            self.current_left_fit = pred_left
            self.current_right_fit = pred_right
        else:
            self.has_lane_lock = False

        # Telemetría de Curvatura y Salida de carril (LDW)
        if self.current_left_fit is not None and self.current_right_fit is not None:
            y_eval = h_native - 1
            a_avg = (self.current_left_fit[0] + self.current_right_fit[0]) / 2.0
            b_avg = (self.current_left_fit[1] + self.current_right_fit[1]) / 2.0

            if abs(a_avg) > 1e-6:
                ym_per_pix = 30.0 / h_native
                xm_per_pix = 3.7 / (0.6 * w_native)
                curverad = ((1 + (2 * a_avg * y_eval * ym_per_pix + b_avg)**2)**1.5) / (abs(2 * a_avg) + 1e-6)
                self.curvature_radius_m = min(9999, curverad * xm_per_pix)

                if self.curvature_radius_m < 300:
                    dir_str = "DER" if a_avg > 0 else "IZQ"
                    self.curvature_text = f"CURVA CERRADA {dir_str}"
                elif self.curvature_radius_m < 900:
                    dir_str = "DER" if a_avg > 0 else "IZQ"
                    self.curvature_text = f"CURVA {dir_str}"
                else:
                    self.curvature_text = "RECTA"
            else:
                self.curvature_radius_m = 9999
                self.curvature_text = "RECTA"

            xl = self.current_left_fit[0]*y_eval**2 + self.current_left_fit[1]*y_eval + self.current_left_fit[2]
            xr = self.current_right_fit[0]*y_eval**2 + self.current_right_fit[1]*y_eval + self.current_right_fit[2]
            lane_center = (xl + xr) / 2.0
            vehicle_center = w_native / 2.0
            lane_w_px = max(10, xr - xl)
            offset_px = vehicle_center - lane_center
            self.lane_center_offset_m = (offset_px / lane_w_px) * 3.5

            if self.lane_center_offset_m > 0.45:
                self.departure_warning = "DESVIO DER"
            elif self.lane_center_offset_m < -0.45:
                self.departure_warning = "DESVIO IZQ"
            else:
                self.departure_warning = "CENTRAL"

        return self.has_lane_lock

    def detect_lanes(self, frame, risk_level="SAFE"):
        h, w = frame.shape[:2]
        w_small, h_small = self.internal_size

        if self.M is None:
            self._init_perspective_matrices(w_small, h_small)

        small = cv2.resize(frame, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        binary_small = self.color_and_gradient_filter(small)
        warped_small = cv2.warpPerspective(binary_small, self.M, (w_small, h_small), flags=cv2.INTER_NEAREST)

        left_fit_s, right_fit_s = self.sliding_windows(warped_small)

        scale_x = w / w_small
        scale_y = h / h_small

        left_fit_native = None
        if left_fit_s is not None:
            left_fit_native = np.array([
                left_fit_s[0] * scale_x / (scale_y**2),
                left_fit_s[1] * scale_x / scale_y,
                left_fit_s[2] * scale_x
            ])

        right_fit_native = None
        if right_fit_s is not None:
            right_fit_native = np.array([
                right_fit_s[0] * scale_x / (scale_y**2),
                right_fit_s[1] * scale_x / scale_y,
                right_fit_s[2] * scale_x
            ])

        is_valid = self.update_kalman_and_telemetry(left_fit_native, right_fit_native, h, w)

        if not is_valid or self.current_left_fit is None or self.current_right_fit is None:
            return frame.copy(), None, False

        y_top = int(h * 0.65)
        y_split = int(h * 0.82)
        y_bottom = h - 1

        ploty_far = np.linspace(y_top, y_split, 20)
        ploty_near = np.linspace(y_split, y_bottom, 20)

        left_fit = self.current_left_fit
        right_fit = self.current_right_fit

        lx_far = left_fit[0]*ploty_far**2 + left_fit[1]*ploty_far + left_fit[2]
        rx_far = right_fit[0]*ploty_far**2 + right_fit[1]*ploty_far + right_fit[2]
        pts_far_l = np.array([np.transpose(np.vstack([lx_far, ploty_far]))])
        pts_far_r = np.array([np.flipud(np.transpose(np.vstack([rx_far, ploty_far])))])
        pts_far = np.hstack((pts_far_l, pts_far_r))

        lx_near = left_fit[0]*ploty_near**2 + left_fit[1]*ploty_near + left_fit[2]
        rx_near = right_fit[0]*ploty_near**2 + right_fit[1]*ploty_near + right_fit[2]
        pts_near_l = np.array([np.transpose(np.vstack([lx_near, ploty_near]))])
        pts_near_r = np.array([np.flipud(np.transpose(np.vstack([rx_near, ploty_near])))])
        pts_near = np.hstack((pts_near_l, pts_near_r))

        overlay = frame.copy()

        if risk_level == "CRITICAL":
            near_color = (0, 0, 220)
            far_color = (0, 160, 240)
        elif risk_level == "WARNING":
            near_color = (0, 165, 255)
            far_color = (0, 200, 100)
        else:
            near_color = (0, 220, 100)
            far_color = (0, 200, 100)

        cv2.fillPoly(overlay, np.int_([pts_far]), far_color)
        cv2.fillPoly(overlay, np.int_([pts_near]), near_color)

        ploty_all = np.linspace(y_top, y_bottom, 35)
        lx_all = left_fit[0]*ploty_all**2 + left_fit[1]*ploty_all + left_fit[2]
        rx_all = right_fit[0]*ploty_all**2 + right_fit[1]*ploty_all + right_fit[2]
        pts_left_line = np.array([np.transpose(np.vstack([lx_all, ploty_all]))])
        pts_right_line = np.array([np.transpose(np.vstack([rx_all, ploty_all]))])

        cv2.polylines(frame, np.int_([pts_left_line]), isClosed=False, color=(0, 255, 255), thickness=3)
        cv2.polylines(frame, np.int_([pts_right_line]), isClosed=False, color=(0, 255, 255), thickness=3)

        result = cv2.addWeighted(overlay, 0.32, frame, 0.68, 0)
        ego_poly = np.vstack([pts_far[0], pts_near[0]]).astype(np.int32)

        return result, ego_poly, True
