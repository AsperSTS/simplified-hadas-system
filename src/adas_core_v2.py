import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from distance_estimator import DistanceEstimator
from config import CLASS_NAMES

class ObjectTrackerAndRiskEngine:
    """
    Motor de seguimiento multiobjeto con soporte de Usuarios Vulnerables (VRU),
    TTC físico, estimación de distancia calibrada y filtrado Ego-Lane.
    """
    def __init__(self, fps=30.0):
        self.fps = fps
        self.distance_estimator = DistanceEstimator(
            camera_height=1.2,
            camera_pitch=3.0,
            focal_length=1500,
            fov_vertical=60,
            fov_horizontal=85
        )
        self.track_history = {}
        # Mapeo de clases ADAS con iconos distintivos
        self.priority_classes = {
            0: "Peatón",
            1: "Bicicleta",
            2: "Auto",
            3: "Moto",
            5: "Autobús",
            7: "Camión",
            9: "Semáforo",
            11: "Alto"
        }
        self.class_icons = {
            0: "P",
            1: "B",
            2: "A",
            3: "M",
            5: "Bus",
            7: "C",
            11: "STOP"
        }
        self.detected_signs = []

    def update_tracks(self, results, frame_height, frame_width, ego_lane_poly=None, current_time=None):
        if current_time is None:
            current_time = time.time()

        tracked_objects = []
        highest_risk_level = "SAFE"
        critical_ttc_obj = None
        self.detected_signs = []

        if len(results) == 0 or results[0].boxes is None:
            return tracked_objects, highest_risk_level, critical_ttc_obj, self.detected_signs

        boxes_data = results[0].boxes
        boxes = boxes_data.xyxy.int().cpu().tolist()
        class_ids = boxes_data.cls.int().cpu().tolist()
        confidences = boxes_data.conf.float().cpu().tolist()
        track_ids = boxes_data.id.int().cpu().tolist() if hasattr(boxes_data, 'id') and boxes_data.id is not None else [i for i in range(len(boxes))]

        active_ids = set()

        for i, box in enumerate(boxes):
            cid = class_ids[i]
            tid = track_ids[i]
            conf = confidences[i]
            active_ids.add(tid)

            # Señal de Alto (Stop Sign = 11)
            if cid == 11:
                sign_dist = self.distance_estimator.estimate_distance(box, cid, frame_height, frame_width)
                self.detected_signs.append({'name': 'ALTO', 'distance': sign_dist, 'box': box})
                continue

            # Vehículos y peatones
            if cid not in [0, 1, 2, 3, 5, 7]:
                continue

            x1, y1, x2, y2 = box
            bottom_center = ((x1 + x2) // 2, y2)

            raw_dist = self.distance_estimator.estimate_distance(box, cid, frame_height, frame_width)

            # Suavizado exponencial de distancia por objeto
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=12)
                smoothed_dist = raw_dist
            else:
                prev_dist = self.track_history[tid][-1][1]
                smoothed_dist = prev_dist * 0.70 + raw_dist * 0.30

            self.track_history[tid].append((current_time, smoothed_dist, bottom_center))

            # Verificar si está dentro del carril propio
            in_lane = False
            if ego_lane_poly is not None and len(ego_lane_poly) > 0:
                poly = np.ascontiguousarray(ego_lane_poly, dtype=np.int32)
                dist_to_poly = cv2.pointPolygonTest(poly, (float(bottom_center[0]), float(bottom_center[1])), False)
                in_lane = dist_to_poly >= 0

            # Velocidad relativa y TTC
            rel_speed = 0.0
            ttc = None

            if len(self.track_history[tid]) >= 4:
                old_t, old_d, _ = self.track_history[tid][0]
                dt = current_time - old_t
                if dt > 0.08:
                    rel_speed = (old_d - smoothed_dist) / dt
                    if rel_speed > 0.6:  # Acercamiento > 2 km/h
                        ttc = smoothed_dist / rel_speed

            # Reglas de Alerta: Prioridad estricta para usuarios vulnerables (motos/peatones)
            alert = "SAFE"
            is_vru = cid in [0, 1, 3]  # Peatón, Bici, Moto

            if in_lane:
                if (ttc is not None and ttc < (2.2 if is_vru else 1.9)) or smoothed_dist < (6.0 if is_vru else 4.8):
                    alert = "CRITICAL"
                    highest_risk_level = "CRITICAL"
                    critical_ttc_obj = (tid, smoothed_dist, ttc)
                elif (ttc is not None and ttc < 4.0) or smoothed_dist < 14.0:
                    alert = "WARNING"
                    if highest_risk_level != "CRITICAL":
                        highest_risk_level = "WARNING"
            else:
                if smoothed_dist < 3.5:
                    alert = "WARNING"
                else:
                    alert = "SAFE"

            cname = self.priority_classes.get(cid, CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "Obj")
            tracked_objects.append({
                'track_id': tid,
                'class_id': cid,
                'class_name': cname,
                'confidence': conf,
                'box': box,
                'distance': smoothed_dist,
                'rel_speed': rel_speed,
                'ttc': ttc,
                'in_lane': in_lane,
                'is_vru': is_vru,
                'alert': alert
            })

        # Limpiar tracks viejos
        current_saved_ids = list(self.track_history.keys())
        for tid in current_saved_ids:
            if tid not in active_ids:
                if len(self.track_history[tid]) > 0:
                    last_t = self.track_history[tid][-1][0]
                    if current_time - last_t > 2.0:
                        del self.track_history[tid]

        return tracked_objects, highest_risk_level, critical_ttc_obj, self.detected_signs


class SpatialTrafficLightAnalyzer:
    def __init__(self):
        self.history = deque(maxlen=6)

    def analyze(self, frame, boxes, class_ids):
        h_frame, w_frame = frame.shape[:2]
        detected_colors = []

        for i, box in enumerate(boxes):
            if class_ids[i] != 9:
                continue

            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1

            if w_box < 6 or h_box < 15 or x1 < 0 or y1 < 0 or x2 > w_frame or y2 > h_frame:
                continue

            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_roi = roi.shape[0]

            t1 = int(h_roi / 3)
            t2 = int(2 * h_roi / 3)

            roi_top = hsv[0:t1, :]
            roi_mid = hsv[t1:t2, :]
            roi_bot = hsv[t2:h_roi, :]

            m_red1 = cv2.inRange(roi_top, np.array([0, 100, 100]), np.array([10, 255, 255]))
            m_red2 = cv2.inRange(roi_top, np.array([170, 100, 100]), np.array([180, 255, 255]))
            red_pixels = cv2.countNonZero(m_red1) + cv2.countNonZero(m_red2)

            m_yellow = cv2.inRange(roi_mid, np.array([18, 90, 100]), np.array([35, 255, 255]))
            yellow_pixels = cv2.countNonZero(m_yellow)

            m_green = cv2.inRange(roi_bot, np.array([45, 90, 90]), np.array([90, 255, 255]))
            green_pixels = cv2.countNonZero(m_green)

            min_thr = max(8, int(w_box * (h_roi / 3) * 0.08))

            if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels >= min_thr:
                detected_colors.append(1)
            elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels >= min_thr:
                detected_colors.append(2)
            elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels >= min_thr:
                detected_colors.append(3)

        state = 0
        if detected_colors:
            counts = np.bincount(detected_colors, minlength=4).astype(float)
            counts[1] *= 1.3
            state = int(np.argmax(counts))

        self.history.append(state)
        smoothed_state = int(np.bincount(self.history, minlength=4).argmax())

        status_text = "CONTINUAR"
        status_color = (240, 240, 240)
        if smoothed_state == 1:
            status_text = "DETENERSE (ROJO)"
            status_color = (30, 30, 255)
        elif smoothed_state == 2:
            status_text = "PRECAUCIÓN (AMARILLO)"
            status_color = (0, 220, 255)
        elif smoothed_state == 3:
            status_text = "AVANZAR (VERDE)"
            status_color = (0, 255, 80)

        return smoothed_state, status_text, status_color


class BlackBoxRecorder:
    def __init__(self, buffer_seconds=4, fps=30.0, output_dir="registros_incidentes"):
        self.fps = fps
        self.buffer_size = int(buffer_seconds * fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_frames = 0
        self.incidents_recorded = 0

    def add_frame(self, frame):
        self.frame_buffer.append(frame.copy())
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

    def trigger_incident(self, trigger_reason, metadata=""):
        if self.cooldown_frames > 0 or len(self.frame_buffer) < 15:
            return None

        self.cooldown_frames = int(self.fps * 5)
        self.incidents_recorded += 1
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"incidente_{self.incidents_recorded:03d}_{trigger_reason}_{timestamp_str}.mp4"

        h, w = self.frame_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filename), fourcc, self.fps, (w, h))

        for f in self.frame_buffer:
            out.write(f)
        out.release()

        print(f"🚨 [CAJA NEGRA] Incidente guardado: {filename.name} | Causa: {trigger_reason} | {metadata}")
        return str(filename)


class DrivingTelemetryAndSafetyScorer:
    def __init__(self):
        self.total_frames = 0
        self.frames_in_lane = 0
        self.ldw_events = 0
        self.fcw_events = 0
        self.safe_frames = 0

    def record_frame(self, ego_lane_valid, departure_warning, risk_level, critical_obj=None):
        self.total_frames += 1
        if ego_lane_valid:
            self.frames_in_lane += 1
            if departure_warning != "CENTRAL":
                self.ldw_events += 1

        if risk_level == "CRITICAL":
            self.fcw_events += 1
        elif risk_level == "SAFE":
            self.safe_frames += 1

    def compute_safety_score(self):
        if self.total_frames == 0:
            return 100.0
        lane_adherence = (self.frames_in_lane / self.total_frames) * 100.0
        ldw_penalty = min(25.0, (self.ldw_events / self.total_frames) * 50.0)
        fcw_penalty = min(35.0, (self.fcw_events / self.total_frames) * 100.0)
        score = max(10.0, 100.0 - ldw_penalty - fcw_penalty)
        return score, lane_adherence


def render_modern_hud(frame, tracked_objects, traffic_light_info, ego_lane_valid, fps, risk_level, lane_detector=None, traffic_signs=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Bounding boxes y badges con soporte VRU
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['box']
        alert = obj['alert']
        in_lane = obj['in_lane']
        is_vru = obj['is_vru']

        if alert == "CRITICAL":
            color = (0, 0, 255)
            thick = 3
        elif alert == "WARNING":
            color = (0, 165, 255)
            thick = 2
        else:
            color = (0, 220, 120) if in_lane else (170, 170, 170)
            thick = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        prefix = "VRU " if is_vru else ""
        label_parts = [f"#{obj['track_id']} {prefix}{obj['class_name']}", f"{obj['distance']:.1f}m"]
        if obj['ttc'] is not None and obj['ttc'] < 10.0:
            label_parts.append(f"TTC:{obj['ttc']:.1f}s")
        if in_lane:
            label_parts.append("[EN CARRIL]")

        label = " | ".join(label_parts)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
        
        badge_y1 = max(0, y1 - 18)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + t_size[0] + 6, y1), (15, 15, 15), -1)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + t_size[0] + 6, y1), color, 1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Señales de tránsito
    if traffic_signs:
        for s in traffic_signs:
            sx1, sy1, sx2, sy2 = s['box']
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
            cv2.putText(frame, f"STOP ({s['distance']:.1f}m)", (sx1, max(20, sy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)

    # 3. Barra superior Glassmorphism
    bar_height = 48
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (12, 14, 18), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # A) Estado de Carril + Indicador de Curvatura + LDW
    lane_color = (0, 255, 100) if ego_lane_valid else (120, 120, 120)
    curve_str = f" [{lane_detector.curvature_text}]" if lane_detector and ego_lane_valid else ""
    ldw_str = ""
    if lane_detector and ego_lane_valid:
        if lane_detector.departure_warning != "CENTRAL":
            ldw_str = f" | {lane_detector.departure_warning} ({abs(lane_detector.lane_center_offset_m):.2f}m)"
            lane_color = (0, 165, 255)

    lane_str = f"CARRIL: {'ACTIVO' if ego_lane_valid else 'BUSCANDO'}{curve_str}{ldw_str}"
    cv2.putText(frame, lane_str, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, lane_color, 2, cv2.LINE_AA)

    # B) Estado del Semáforo
    _, tl_text, tl_color = traffic_light_info
    cv2.putText(frame, f"SEMÁFORO: {tl_text}", (int(w * 0.42), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tl_color, 2, cv2.LINE_AA)

    # C) Alerta FCW
    if risk_level == "CRITICAL":
        alert_str = "¡FRENADO CRÍTICO!"
        alert_col = (0, 0, 255)
    elif risk_level == "WARNING":
        alert_str = "PRECAUCIÓN"
        alert_col = (0, 165, 255)
    else:
        alert_str = "SEGURO"
        alert_col = (0, 230, 80)
    cv2.putText(frame, f"ESTADO: {alert_str}", (int(w * 0.70), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, alert_col, 2, cv2.LINE_AA)

    # D) FPS
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 78, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
