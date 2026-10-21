import sys
import os
import time
from pathlib import Path

# Agregar src al sys.path para importar correctamente
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
from ultralytics import YOLO

from lane_detector import LaneDetector
from yolo_detector import obstacles_detector
from distance_estimator import add_distance_estimation, DistanceEstimator
from stop_lights import process_stop_light
from config import *

def run_benchmark(video_path, max_frames=200):
    print(f"=== INICIANDO BENCHMARK DE LÍNEA BASE ===")
    print(f"Video: {video_path}")
    print(f"Frames máximos a evaluar: {max_frames}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Resolución: {width}x{height}, FPS nativo: {native_fps:.1f}, Frames totales: {total_frames}")

    # Inicializar componentes
    print("Cargando modelo YOLO11n...")
    model = YOLO("yolo11n.pt")
    lane_detector = LaneDetector(smoothing_frames=4)
    distance_estimator = DistanceEstimator(camera_height=1, camera_pitch=4, focal_length=1500, fov_vertical=60, fov_horizontal=85)

    # Métricas de tiempo (en ms)
    times_lane = []
    times_yolo = []
    times_dist = []
    times_light = []
    times_total = []

    # Métricas de detección
    lane_success_count = 0
    detections_by_class = {}
    warning_counts = {'critical': 0, 'warning': 0, 'safe': 0}
    stop_light_counts = {0: 0, 1: 0, 2: 0, 3: 0} # 0=desconocido, 1=rojo, 2=amarillo, 3=verde

    frame_count = 0
    last_detections = None

    start_bench_time = time.time()

    sample_frames_saved = 0
    os.makedirs(SRC_DIR / "benchmark_output", exist_ok=True)

    while frame_count < max_frames:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Lane detection
        t_lane_start = time.perf_counter()
        result_frame = lane_detector.detect_lanes(frame)
        t_lane_end = time.perf_counter()

        # Chequear si detectó carriles válidos en este frame
        if len(lane_detector.left_fits) > 0 and len(lane_detector.right_fits) > 0:
            lane_success_count += 1

        # 2. YOLO Obstacles
        t_yolo_start = time.perf_counter()
        result_frame, last_detections = obstacles_detector(
            model, result_frame, frame_count, last_detections
        )
        t_yolo_end = time.perf_counter()

        # Recopilar clases detectadas
        for cid in last_detections['class_ids']:
            cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"id_{cid}"
            detections_by_class[cname] = detections_by_class.get(cname, 0) + 1

        # 3. Distance estimation
        t_dist_start = time.perf_counter()
        # Contar alertas de distancia antes de dibujar
        for i, box in enumerate(last_detections['boxes']):
            cid = last_detections['class_ids'][i]
            if cid in [0, 1, 2, 3, 5, 7]:
                dist = distance_estimator.estimate_distance(box, cid, height, width)
                wlevel, _ = distance_estimator.get_warning_level(dist)
                warning_counts[wlevel] += 1

        result_frame = add_distance_estimation(
            result_frame, last_detections['boxes'], last_detections['class_ids'], distance_estimator
        )
        t_dist_end = time.perf_counter()

        # 4. Stop lights
        t_light_start = time.perf_counter()
        result_frame = process_stop_light(
            result_frame, last_detections['boxes'], last_detections['class_ids']
        )
        t_light_end = time.perf_counter()

        t_total_end = time.perf_counter()

        # Guardar tiempos
        times_lane.append((t_lane_end - t_lane_start) * 1000)
        times_yolo.append((t_yolo_end - t_yolo_start) * 1000)
        times_dist.append((t_dist_end - t_dist_start) * 1000)
        times_light.append((t_light_end - t_light_start) * 1000)
        times_total.append((t_total_end - t0) * 1000)

        # Guardar frames de muestra (ej. frame 50 y 150)
        if frame_count in [50, 150] and sample_frames_saved < 2:
            out_sample = SRC_DIR / "benchmark_output" / f"sample_frame_{frame_count}.jpg"
            cv2.imwrite(str(out_sample), result_frame)
            sample_frames_saved += 1

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Procesados {frame_count}/{max_frames} frames...")

    cap.release()
    total_elapsed = time.time() - start_bench_time

    # Cálculos estadísticos
    avg_total_ms = np.mean(times_total)
    avg_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0
    avg_lane_ms = np.mean(times_lane)
    avg_yolo_ms = np.mean(times_yolo)
    avg_dist_ms = np.mean(times_dist)
    avg_light_ms = np.mean(times_light)

    print("\n" + "="*50)
    print("=== RESULTADOS DE LÍNEA BASE (BASELINE) ===")
    print("="*50)
    print(f"Frames procesados: {frame_count}")
    print(f"Tiempo total: {total_elapsed:.2f} s")
    print(f"FPS Promedio: {avg_fps:.2f} FPS (Latencia media por frame: {avg_total_ms:.2f} ms)")
    print("\n--- Desglose de Latencia por Módulo ---")
    print(f"1. Lane Detector:        {avg_lane_ms:6.2f} ms ({avg_lane_ms/avg_total_ms*100:5.1f}%)")
    print(f"2. YOLO Inferencia:      {avg_yolo_ms:6.2f} ms ({avg_yolo_ms/avg_total_ms*100:5.1f}%) [Inferencia cada 2 frames]")
    print(f"3. Estimación Distancia: {avg_dist_ms:6.2f} ms ({avg_dist_ms/avg_total_ms*100:5.1f}%)")
    print(f"4. Análisis Semáforos:   {avg_light_ms:6.2f} ms ({avg_light_ms/avg_total_ms*100:5.1f}%)")
    print(f"   Otros (I/O, etc.):    {avg_total_ms - (avg_lane_ms + avg_yolo_ms + avg_dist_ms + avg_light_ms):6.2f} ms")

    print("\n--- Métricas de Calidad y Detección ---")
    print(f"Detección de carril activa: {lane_success_count}/{frame_count} frames ({lane_success_count/frame_count*100:.1f}%)")
    print("Objetos detectados por clase (total de instancias acumuladas):")
    for cname, count in sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cname}: {count}")

    print("Alertas de distancia emitidas:")
    print(f"  - Seguras (>20m):   {warning_counts['safe']}")
    print(f"  - Advertencia (7-20m): {warning_counts['warning']}")
    print(f"  - Críticas (<7m):   {warning_counts['critical']}")
    print("="*50)

if __name__ == "__main__":
    vpath = SRC_DIR / "video1_720.mp4"
    run_benchmark(vpath, max_frames=200)
