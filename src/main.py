import cv2
import argparse
import time
from ultralytics import YOLO
from utils import fps_counter, draw_text, record_safety_videos

from lane_detector import LaneDetector
from yolo_detector import obstacles_detector
from distance_estimator import add_distance_estimation
from config import *
from stop_lights import process_stop_light

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sistema de detección para conducción autónoma')
    parser.add_argument('--input', '-i', type=str, default='0',
                        help='Ruta al video o número de cámara (por defecto: cámara 0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Ruta para guardar el video procesado (opcional)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Mostrar video en tiempo real')
    return parser.parse_args()

def process_video(input_source, output_path=None, show_video=True):
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
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # Inicializar contador FPS
    fps_tracker = fps_counter()
    next(fps_tracker)  # Inicializar el generador
    
    # Inicializar modelo y variables para detección
    model = YOLO("yolo11n.pt")
    frame_count = 0
    last_detections = None  # Para almacenar detecciones entre frames
    
    lane_detector = LaneDetector(smoothing_frames=4)
    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error en la captura.")
                break
            
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

            # Calcular y mostrar FPS
            current_fps = fps_tracker.send(time.time())
            draw_text(result_frame, f"FPS: {current_fps:.1f}", (10, 20))

            if show_video:
                cv2.imshow('Video', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

            # Esperar si el procesamiento fue muy rápido
            elapsed = time.time() - start_time
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    finally:
        # Liberar recursos
        cap.release()

if __name__ == "__main__":
    args = parse_arguments()
    process_video(args.input, args.output, args.show)