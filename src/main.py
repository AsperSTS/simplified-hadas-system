import cv2
import argparse
import time
from pedestrian_detector import detect_pedestrians
from lane_detector import detect_lanes
from obstacle_detector import detect_obstacles, detect_obstacles_depth
from vehicle_detector import detect_vehicles, detect_vehicles_by_color_shape
from utils import fps_counter, draw_text

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Configurar grabador de video si se especificó una salida
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Inicializar contador FPS
    fps_tracker = fps_counter()
    next(fps_tracker)  # Inicializar el generador
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error en la captura.")
                break
            
            # Crear una copia del frame para dibujar resultados
            result_frame = frame.copy()
            
            # Aplicar los detectores
            # result_frame = detect_obstacles(result_frame)
            # result_frame = detect_vehicles(result_frame)
            # result_frame = detect_pedestrians(result_frame)
            result_frame = detect_lanes(result_frame)
            
            
            # Calcular y mostrar FPS
            current_fps = fps_tracker.send(time.time())
            draw_text(result_frame, f"FPS: {current_fps:.1f}", (20, 40))
            
            # Mostrar el resultado si se solicitó
            if show_video:
                # print("Mostrando")
                cv2.imshow('Video', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Guardar el frame procesado si se especificó una salida
            if out:
                out.write(result_frame)
    
    finally:
        # Liberar recursos
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_arguments()
    process_video(args.input, args.output, args.show)