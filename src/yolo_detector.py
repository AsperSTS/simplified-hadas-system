# from ultralytics import YOLO

import cv2
import numpy as np
from utils import *        # Utilidades personalizadas (ej. funciones de dibujo, posprocesamiento)
from config import *       # Configuración, como nombres de clases y colores por clase

def obstacles_detector(model, frame, frame_count, last_detections=None):
    """
    Detecta obstáculos en un frame utilizando un modelo YOLO.

    Args:
        model: Modelo YOLO previamente cargado.
        frame: Frame actual (imagen en formato BGR) a analizar.
        frame_count: Número actual del frame, usado para realizar detección cada ciertos frames.
        last_detections: Diccionario con las últimas detecciones válidas realizadas.
                         Formato: {'boxes': [], 'class_ids': [], 'confidences': []}

    Returns:
        tuple: Frame con las detecciones dibujadas y las detecciones actuales actualizadas.
    """
    
    # Inicializar el diccionario de detecciones si no se pasó como argumento
    if last_detections is None:
        last_detections = {
            'boxes': [],          # Coordenadas de las cajas detectadas
            'class_ids': [],      # ID de clase por cada caja
            'confidences': []     # Confianza de cada detección
        }

    # Solo ejecutar la detección en ciertos frames (cada 2 frames en este caso)
    if frame_count % 2 == 0:
        # Realizar la inferencia con el modelo
        results = model(frame)

        # Inicializar listas temporales para guardar nuevas detecciones
        new_boxes = []
        new_class_ids = []
        new_confidences = []

        # Iterar sobre los resultados obtenidos
        for r in results:
            # Extraer las coordenadas de las cajas delimitadoras como enteros
            boxes = r.boxes.xyxy.int().cpu().tolist()
            # Extraer las puntuaciones de confianza
            confidences = r.boxes.conf.float().cpu().tolist()
            # Extraer las clases detectadas
            class_ids = r.boxes.cls.int().cpu().tolist()

            # Evaluar cada detección individualmente
            for i, box in enumerate(boxes):
                if confidences[i] > 0.3:  # Filtrar detecciones con baja confianza
                    new_boxes.append(box)
                    new_class_ids.append(class_ids[i])
                    new_confidences.append(confidences[i])

                    # Dibujar la caja en el frame original
                    xmin, ymin, xmax, ymax = box
                    class_id = class_ids[i]
                    confidence = confidences[i]

                    # Crear la etiqueta con el nombre de la clase y la confianza
                    label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"

                    # Obtener el color asociado a la clase
                    color = COLORS_PER_CLASS[class_id]

                    # Dibujar el rectángulo y el texto sobre el frame
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
                    cv2.putText(frame, label, (xmin, ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Actualizar el diccionario con las detecciones más recientes
        last_detections['boxes'] = new_boxes
        last_detections['class_ids'] = new_class_ids
        last_detections['confidences'] = new_confidences

    else:
        # Si no se hace nueva detección, reutilizar las últimas y dibujarlas
        for i, box in enumerate(last_detections['boxes']):
            xmin, ymin, xmax, ymax = box
            class_id = last_detections['class_ids'][i]
            confidence = last_detections['confidences'][i]

            # Crear etiqueta y color como en la detección anterior
            label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
            color = COLORS_PER_CLASS[class_id]

            # Dibujar la caja y etiqueta en el frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(frame, label, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Retornar el frame con las anotaciones y las últimas detecciones
    return frame, last_detections
