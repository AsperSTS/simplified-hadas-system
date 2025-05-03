# from ultralytics import YOLO
import cv2
import numpy as np
from utils import *
from config import *
def obstacles_detector(model, frame, frame_count, width, height, last_detections=None):
    """
    Detecta obstáculos usando un modelo YOLO
    
    Args:
        model: Modelo YOLO cargado
        frame: Frame actual a procesar
        frame_count: Contador de frames
        width: Ancho del frame
        height: Alto del frame
        last_detections: Diccionario con las últimas detecciones válidas
    
    Returns:
        tuple: Frame procesado y diccionario con las últimas detecciones
    """
    # Si no hay detecciones previas, inicializar
    if last_detections is None:
        last_detections = {
            'boxes': [],
            'class_ids': [],
            'confidences': []
        }
    
    roi_vertices = np.array([
                [(0, height),
                 (0, height-(height//5)), 
                 (width//3+(height//7), height//1.8), 
                 (2*width//3-(height//7), height//1.8), 
                 (width, height-(height//5)),
                 (width, height)]
            ], dtype=np.int32)
    
    # Procesar cada 3 frames
    if frame_count % 3 == 0:
        result_frame = frame.copy()
        roi_frame = region_of_interest(result_frame, roi_vertices)
        results = model(roi_frame)

        # Limpiar las listas anteriores
        new_boxes = []
        new_class_ids = []
        new_confidences = []

        for r in results:
            boxes = r.boxes.xyxy.int().cpu().tolist()
            confidences = r.boxes.conf.float().cpu().tolist()
            class_ids = r.boxes.cls.int().cpu().tolist()
            
            # Guardar las detecciones actuales
            for i, box in enumerate(boxes):
                if confidences[i] > 0.4:  # Umbral de confianza
                    new_boxes.append(box)
                    new_class_ids.append(class_ids[i])
                    new_confidences.append(confidences[i])
                    
                    # Dibujar el rectángulo en el frame actual
                    xmin, ymin, xmax, ymax = box
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
                    color = COLORS_PER_CLASS[class_id]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Actualizar las detecciones guardadas
        last_detections['boxes'] = new_boxes
        last_detections['class_ids'] = new_class_ids
        last_detections['confidences'] = new_confidences
    else:
        # En los fotogramas intermedios, dibuja las últimas detecciones
        for i, box in enumerate(last_detections['boxes']):
            xmin, ymin, xmax, ymax = box
            class_id = last_detections['class_ids'][i]
            confidence = last_detections['confidences'][i]
            label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
            color = COLORS_PER_CLASS[class_id]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Dibujar ROI
    cv2.polylines(frame, [roi_vertices], isClosed=True, color=(255, 165, 0), thickness=2)

    return frame, last_detections