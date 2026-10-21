import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO
from utils import fps_counter, draw_text, record_safety_videos

import torch
from lane_detector import LaneDetector
from yolo_detector import obstacles_detector
from distance_estimator import add_distance_estimation
from config import *
from stop_lights import process_stop_light

# V2 Components
from lane_detector_v2 import AdvancedLaneDetector
from adas_core_v2 import ObjectTrackerAndRiskEngine, SpatialTrafficLightAnalyzer, BlackBoxRecorder, render_modern_hud

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sistema de detección para conducción autónoma')
    parser.add_argument('--input', '-i', type=str, default='0',
                        help='Ruta al video o número de cámara (por defecto: cámara 0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Ruta para guardar el video procesado (opcional)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Mostrar video en tiempo real')
    parser.add_argument('--quality', '-q', type=int, default=90,
                        help='Calidad del video de salida (1-100, por defecto: 90)')
    parser.add_argument('--enhance', '-e', action='store_true',
                        help='Aplicar mejoras de imagen al video de salida')
    parser.add_argument('--version', '-v', type=int, choices=[1, 2], default=2,
                        help='Versión del pipeline ADAS (1: Clásica/Baseline, 2: Avanzada/Mejorada)')
    return parser.parse_args()

def process_video(input_source, output_path=None, show_video=True, quality=90, enhance=True, version=2):
    # Intentar interpretar la entrada como número de cámara
    try:
        source = int(input_source)
    except ValueError:
        source = input_source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente de video: {input_source}")
        return
    
    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Valor predeterminado si no se puede obtener el FPS

    # Configurar el VideoWriter si se especificó una ruta de salida
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        if video_writer.isOpened():
            print(f"Guardando video en: {output_path}")
        else:
            print(f"ADVERTENCIA: No se pudo inicializar VideoWriter para {output_path}")
            video_writer = None

    # Inicializar contador FPS
    fps_tracker = fps_counter()
    next(fps_tracker)
    
    # Dispositivo para inferencia
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Iniciando pipeline ADAS v{version} (Device: {device.upper()})...")

    # Inicializar modelo YOLO
    model = YOLO("yolo11n.pt")
    frame_count = 0

    if version == 2:
        lane_detector = AdvancedLaneDetector(smoothing_frames=8)
        risk_engine = ObjectTrackerAndRiskEngine(fps=fps)
        traffic_analyzer = SpatialTrafficLightAnalyzer()
        black_box = BlackBoxRecorder(buffer_seconds=4, fps=fps)
    else:
        lane_detector = LaneDetector(smoothing_frames=4)
        last_detections = None

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error en la captura.")
                break

            current_time = time.time()

            if version == 2:
                # 1. Detección de carril con Kalman Filter y Heatmap de Peligro
                lane_frame, ego_poly, lane_valid = lane_detector.detect_lanes(frame, risk_level=prev_risk_level if 'prev_risk_level' in locals() else "SAFE")

                # 2. Inferencia y tracking continuo con ByteTrack (acelerado 480p)
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device, imgsz=480)

                # 3. Motor de riesgo (TTC + In-Lane + Señales de tránsito)
                tracked_objs, risk_level, crit_obj, traffic_signs = risk_engine.update_tracks(
                    results, height, width, ego_lane_poly=ego_poly, current_time=current_time
                )
                prev_risk_level = risk_level

                # 4. Semáforos topológicos
                boxes_raw = results[0].boxes.xyxy.int().cpu().tolist() if results[0].boxes is not None else []
                classes_raw = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes is not None else []
                tl_info = traffic_analyzer.analyze(frame, boxes_raw, classes_raw)

                # 5. Render HUD moderno con telemetría de carril y señales
                calc_fps = fps_tracker.send(time.time())
                result_frame = render_modern_hud(lane_frame, tracked_objs, tl_info, lane_valid, calc_fps, risk_level, lane_detector, traffic_signs)

                # 6. Caja negra
                black_box.add_frame(result_frame)
                if risk_level == "CRITICAL" and crit_obj is not None:
                    tid, d, ttc_val = crit_obj
                    ttc_str = f"{ttc_val:.1f}s" if ttc_val else "N/A"
                    black_box.trigger_incident("FCW_CRITICO", f"Track #{tid} a {d:.1f}m, TTC={ttc_str}")

            else:
                # Pipeline Clásico V1
                result_frame = lane_detector.detect_lanes(frame)
                result_frame, last_detections = obstacles_detector(
                    model, result_frame, frame_count, last_detections
                )
                result_frame = add_distance_estimation(
                    result_frame, last_detections['boxes'], last_detections['class_ids']
                )
                result_frame = process_stop_light(
                    result_frame, last_detections['boxes'], last_detections['class_ids']
                )
                current_fps = fps_tracker.send(time.time())
                draw_text(result_frame, f"FPS: {current_fps:.1f}", (10, 20))

            # Guardar el frame si se especificó salida
            if video_writer:
                video_writer.write(result_frame)

            if show_video:
                cv2.imshow('ADAS System', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_arguments()
    process_video(args.input, args.output, args.show, args.quality, args.enhance, args.version)