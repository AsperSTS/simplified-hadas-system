import sys
import os
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# V1 Imports
from lane_detector import LaneDetector as LaneDetectorV1
from yolo_detector import obstacles_detector as obstacles_detector_v1
from distance_estimator import add_distance_estimation as add_distance_v1, DistanceEstimator as DistanceEstimatorV1
from stop_lights import process_stop_light as process_stop_light_v1
from config import *

# V2 Imports
from lane_detector_v2 import AdvancedLaneDetector as LaneDetectorV2
from adas_core_v2 import ObjectTrackerAndRiskEngine, SpatialTrafficLightAnalyzer, BlackBoxRecorder, render_modern_hud

def run_v2_evaluation(video_path, max_frames=200):
    print("\n" + "="*50)
    print("=== EJECUTANDO ADAS 2.0 (VERSION MEJORADA) ===")
    print("="*50)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    # Dispositivo óptimo (MPS si está disponible)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Inferencia acelerada en: {device.upper()}")

    # Inicializar componentes V2
    model = YOLO("yolo11n.pt")
    lane_detector_v2 = LaneDetectorV2(smoothing_frames=8)
    risk_engine = ObjectTrackerAndRiskEngine(fps=fps_in)
    traffic_light_analyzer = SpatialTrafficLightAnalyzer()
    black_box = BlackBoxRecorder(buffer_seconds=4, fps=fps_in, output_dir=SRC_DIR / "incidentes_v2")
    from adas_core_v2 import DrivingTelemetryAndSafetyScorer
    safety_scorer = DrivingTelemetryAndSafetyScorer()

    times_total = []
    times_lane = []
    times_yolo_track = []
    times_risk = []

    lane_valid_count = 0
    in_lane_objects_detected = 0
    ttc_alerts_count = 0
    unique_tracks = set()

    frame_count = 0
    start_time = time.time()
    v2_samples = {}
    prev_risk_level = "SAFE"

    while frame_count < max_frames:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        current_timestamp = time.time()

        # 1. Lane Detection 2.0 (con Filtro de Kalman y Heatmap de Peligro en Asfalto)
        t_lane_0 = time.perf_counter()
        lane_frame, ego_poly, lane_valid = lane_detector_v2.detect_lanes(frame, risk_level=prev_risk_level)
        t_lane_1 = time.perf_counter()
        if lane_valid:
            lane_valid_count += 1

        # 2. YOLO Tracking (ByteTrack en MPS continuo a 480p)
        t_track_0 = time.perf_counter()
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device, imgsz=480)
        t_track_1 = time.perf_counter()

        # 3. Risk Engine (Distancia + In-Lane + TTC + Señales)
        t_risk_0 = time.perf_counter()
        tracked_objs, risk_level, crit_obj, traffic_signs = risk_engine.update_tracks(
            results, h, w, ego_lane_poly=ego_poly, current_time=current_timestamp
        )
        prev_risk_level = risk_level

        for obj in tracked_objs:
            unique_tracks.add(obj['track_id'])
            if obj['in_lane']:
                in_lane_objects_detected += 1
            if obj['alert'] in ['WARNING', 'CRITICAL']:
                ttc_alerts_count += 1

        # 4. Semáforos (Verificación espacial 3 tercios)
        boxes_raw = results[0].boxes.xyxy.int().cpu().tolist() if results[0].boxes is not None else []
        classes_raw = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes is not None else []
        tl_info = traffic_light_analyzer.analyze(frame, boxes_raw, classes_raw)

        # 5. Render HUD Moderno
        current_fps = 1000.0 / ((time.perf_counter() - t0) * 1000)
        hud_frame = render_modern_hud(lane_frame, tracked_objs, tl_info, lane_valid, current_fps, risk_level, lane_detector_v2, traffic_signs)
        t_risk_1 = time.perf_counter()

        # 6. Caja Negra & Telemetría de Seguridad
        black_box.add_frame(hud_frame)
        safety_scorer.record_frame(lane_valid, lane_detector_v2.departure_warning, risk_level, crit_obj)

        if risk_level == "CRITICAL" and crit_obj is not None:
            tid, d, ttc_val = crit_obj
            ttc_str = f"{ttc_val:.1f}s" if ttc_val else "N/A"
            black_box.trigger_incident("FCW_CRITICO", f"Track #{tid} a {d:.1f}m, TTC={ttc_str}")

        t_end = time.perf_counter()

        times_lane.append((t_lane_1 - t_lane_0) * 1000)
        times_yolo_track.append((t_track_1 - t_track_0) * 1000)
        times_risk.append((t_risk_1 - t_risk_0) * 1000)
        times_total.append((t_end - t0) * 1000)

        if frame_count in [50, 150]:
            v2_samples[frame_count] = hud_frame.copy()

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"V2: Procesados {frame_count}/{max_frames} frames...")

    cap.release()
    total_time = time.time() - start_time

    avg_total_ms = np.mean(times_total)
    avg_fps = 1000.0 / avg_total_ms
    avg_lane_ms = np.mean(times_lane)
    avg_track_ms = np.mean(times_yolo_track)
    avg_risk_ms = np.mean(times_risk)

    safety_score, lane_adherence = safety_scorer.compute_safety_score()

    print("\n" + "="*50)
    print("=== RESULTADOS ADAS 2.0 (ITERACION 3 - KALMAN + HEATMAP) ===")
    print("="*50)
    print(f"Frames procesados: {frame_count}")
    print(f"Tiempo total: {total_time:.2f} s")
    print(f"FPS Promedio: {avg_fps:.2f} FPS (Latencia media: {avg_total_ms:.2f} ms/frame)")
    print(f"Detección de carril activa (Kalman Lock): {lane_valid_count}/{frame_count} frames ({lane_valid_count/frame_count*100:.1f}%)")
    print(f"Adherencia al carril (Centrado): {lane_adherence:.1f}%")
    print(f"Calificación de Seguridad Vial (Safety Score): {safety_score:.1f} / 100")
    print(f"Vehículos y peatones rastreados con ID único: {len(unique_tracks)}")
    print(f"Alertas de colisión frontal calculadas con TTC real: {ttc_alerts_count}")
    print(f"Incidentes críticos guardados en Caja Negra: {black_box.incidents_recorded}")
    print("\n--- Desglose de Latencia V2 ---")
    print(f"1. Lane Detector V2 (Kalman + IPM):   {avg_lane_ms:6.2f} ms")
    print(f"2. YOLO Inferencia + ByteTrack (MPS):  {avg_track_ms:6.2f} ms [EN TODOS LOS FRAMES]")
    print(f"3. Risk Engine + Semáforo + HUD:        {avg_risk_ms:6.2f} ms")
    print("="*50)

    # Guardar comparativas visuales
    out_dir = SRC_DIR / "benchmark_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f_idx, v2_img in v2_samples.items():
        v1_path = out_dir / f"sample_frame_{f_idx}.jpg"
        if v1_path.exists():
            v1_img = cv2.imread(str(v1_path))
            # Crear imagen lado a lado (V1 vs V2)
            h_im, w_im = v1_img.shape[:2]
            half_w = w_im // 2
            v1_resized = cv2.resize(v1_img, (half_w, int(h_im * 0.5)))
            v2_resized = cv2.resize(v2_img, (half_w, int(h_im * 0.5)))
            
            # Títulos
            cv2.putText(v1_resized, "LINEA BASE (V1)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(v2_resized, "ADAS 2.0 (MEJORADO)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            comparison = np.hstack([v1_resized, v2_resized])
            cmp_path = out_dir / f"comparison_side_by_side_{f_idx}.jpg"
            cv2.imwrite(str(cmp_path), comparison)
            print(f"Comparativa visual guardada en: {cmp_path.name}")

if __name__ == "__main__":
    vpath = SRC_DIR / "video1_720.mp4"
    run_v2_evaluation(vpath, max_frames=200)
