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

# V1 Components
from lane_detector import LaneDetector as LaneDetectorV1
from yolo_detector import obstacles_detector as obstacles_detector_v1
from distance_estimator import add_distance_estimation as add_distance_v1, DistanceEstimator as DistanceEstimatorV1
from stop_lights import process_stop_light as process_stop_light_v1
from config import *

# V2 Components
from lane_detector_v2 import AdvancedLaneDetector as LaneDetectorV2
from adas_core_v2 import ObjectTrackerAndRiskEngine, SpatialTrafficLightAnalyzer, BlackBoxRecorder, render_modern_hud

def run_multi_frame_evaluation(video_path):
    print("==================================================")
    print("=== EVALUACIÓN MULTI-FRAME EXTENDIDA (16 PUNTOS) ===")
    print("==================================================")

    # 16 puntos distribuidos a lo largo de los 8001 frames (4.4 minutos)
    target_frames = [
        250, 600, 1000, 1500, 2000, 2500, 3000, 3500,
        4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500
    ]
    max_target = max(target_frames)
    print(f"Puntos de evaluación seleccionados: {target_frames}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error al abrir {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Dispositivo de aceleración: {device.upper()}")

    # Modelos e instancias
    model = YOLO("yolo11n.pt")
    
    # Instancias V1
    lane_v1 = LaneDetectorV1(smoothing_frames=4)
    dist_v1 = DistanceEstimatorV1(camera_height=1, camera_pitch=4, focal_length=1500, fov_vertical=60, fov_horizontal=85)
    last_detections_v1 = None

    # Instancias V2
    lane_v2 = LaneDetectorV2(smoothing_frames=8)
    risk_engine_v2 = ObjectTrackerAndRiskEngine(fps=fps_in)
    tl_analyzer_v2 = SpatialTrafficLightAnalyzer()

    out_dir = SRC_DIR / "multi_frame_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    prev_risk_level = "SAFE"

    # Métricas comparativas
    results_table = []

    print("\nIniciando procesamiento a lo largo del video...")
    t_start = time.time()

    while frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video antes de alcanzar todos los targets.")
            break

        # ==========================================
        # 1. Procesamiento con Pipeline V1
        # ==========================================
        res_v1 = lane_v1.detect_lanes(frame.copy())
        res_v1, last_detections_v1 = obstacles_detector_v1(
            model, res_v1, frame_idx, last_detections_v1
        )
        res_v1 = add_distance_v1(
            res_v1, last_detections_v1['boxes'], last_detections_v1['class_ids'], dist_v1
        )
        res_v1 = process_stop_light_v1(
            res_v1, last_detections_v1['boxes'], last_detections_v1['class_ids']
        )
        # Identificar si V1 tenía carril activo
        v1_lane_active = len(lane_v1.left_fits) > 0 and len(lane_v1.right_fits) > 0

        # ==========================================
        # 2. Procesamiento con Pipeline V2
        # ==========================================
        current_time = time.time()
        lane_frame_v2, ego_poly, lane_valid_v2 = lane_v2.detect_lanes(frame.copy(), risk_level=prev_risk_level)

        results_yolo = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device, imgsz=480)

        tracked_objs, risk_level, crit_obj, traffic_signs = risk_engine_v2.update_tracks(
            results_yolo, h, w, ego_lane_poly=ego_poly, current_time=current_time
        )
        prev_risk_level = risk_level

        boxes_raw = results_yolo[0].boxes.xyxy.int().cpu().tolist() if results_yolo[0].boxes is not None else []
        classes_raw = results_yolo[0].boxes.cls.int().cpu().tolist() if results_yolo[0].boxes is not None else []
        tl_info = tl_analyzer_v2.analyze(frame, boxes_raw, classes_raw)

        res_v2 = render_modern_hud(
            lane_frame_v2, tracked_objs, tl_info, lane_valid_v2, 35.0, risk_level, lane_v2, traffic_signs
        )

        # ==========================================
        # 3. Guardar comparativa si es un frame target
        # ==========================================
        if frame_idx in target_frames:
            saved_count += 1
            timestamp_sec = frame_idx / fps_in

            # Crear imagen lado a lado (V1 vs V2)
            half_w = w // 2
            v1_half = cv2.resize(res_v1, (half_w, int(h * 0.5)))
            v2_half = cv2.resize(res_v2, (half_w, int(h * 0.5)))

            # Overlays de cabecera en la comparativa
            cv2.putText(v1_half, f"V1 CLASICO | Frame {frame_idx} ({timestamp_sec:.1f}s)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(v2_half, f"ADAS 2.0 | Offset: {lane_v2.lane_center_offset_m:+.2f}m | TTC: {risk_level}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            comp_img = np.hstack([v1_half, v2_half])
            out_file = out_dir / f"comparison_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_file), comp_img)

            # Extraer métricas de este frame
            in_lane_objs = [o for o in tracked_objs if o['in_lane']]
            min_ttc = min([o['ttc'] for o in tracked_objs if o['ttc'] is not None], default=None)
            ttc_str = f"{min_ttc:.1f}s" if min_ttc else "N/A"

            results_table.append({
                'frame': frame_idx,
                'time_s': timestamp_sec,
                'v1_lane': "Sí (Rígido)" if v1_lane_active else "No",
                'v2_lane': "Sí (Kalman)" if lane_valid_v2 else "No",
                'v2_offset': f"{lane_v2.lane_center_offset_m:+.2f}m",
                'tracked_count': len(tracked_objs),
                'in_lane_count': len(in_lane_objs),
                'risk_level': risk_level,
                'min_ttc': ttc_str,
                'semáforo': tl_info[1]
            })

            print(f"[{saved_count}/{len(target_frames)}] Guardado frame {frame_idx:04d} ({timestamp_sec:.1f}s) | Objs: {len(tracked_objs)} | In-Lane: {len(in_lane_objs)} | Riesgo: {risk_level} | Semáforo: {tl_info[1]}")

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"Progreso: {frame_idx}/{max_target} frames analizados ({(frame_idx/max_target)*100:.1f}%)...")

    cap.release()
    elapsed = time.time() - t_start

    print("\n" + "="*80)
    print("=== TABLA COMPARATIVA DE LOS 16 FRAMES EVALUADOS A LO LARGO DEL VIDEO ===")
    print("="*80)
    print(f"{'Frame':<7} {'Tiempo':<8} {'Carril V1':<12} {'Carril V2':<12} {'Desvío LDW':<12} {'Objs':<6} {'En Carril':<10} {'Riesgo':<10} {'TTC Min':<9} {'Semáforo'}")
    print("-"*80)
    for r in results_table:
        print(f"{r['frame']:<7} {r['time_s']:<6.1f}s  {r['v1_lane']:<12} {r['v2_lane']:<12} {r['v2_offset']:<12} {r['tracked_count']:<6} {r['in_lane_count']:<10} {r['risk_level']:<10} {r['min_ttc']:<9} {r['semáforo']}")
    print("="*80)
    print(f"Tiempo total de procesamiento de {max_target} frames: {elapsed:.2f} s ({max_target/elapsed:.1f} FPS promedio global).")

if __name__ == "__main__":
    vpath = SRC_DIR / "video1_720.mp4"
    run_multi_frame_evaluation(vpath)
