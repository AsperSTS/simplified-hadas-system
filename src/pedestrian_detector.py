import cv2
import numpy as np
from utils import draw_text, non_max_suppression

def detect_pedestrians(frame):
    """
    Detecta peatones en la imagen utilizando HOG + SVM.
    
    Args:
        frame: Imagen de entrada
    
    Returns:
        Imagen con peatones detectados y marcados
    """
    # Crear una copia del frame
    result = frame.copy()
    height, width = frame.shape[:2]
    
    # Inicializar detector HOG
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Aplicar el detector HOG con escala reducida para mejor rendimiento
    scale = 1.05
    win_stride = (8, 8)
    padding = (16, 16)
    
    # Redimensionar el frame para mejorar velocidad y efectividad
    resized_frame = cv2.resize(frame, (width // 2, height // 2))
    
    # Detectar personas
    pedestrian_rects, weights = hog.detectMultiScale(
        resized_frame,
        winStride=win_stride,
        padding=padding,
        scale=scale
    )
    
    # Ajustar las coordenadas al tamaño original
    if len(pedestrian_rects) > 0:
        pedestrian_rects = np.array([[x * 2, y * 2, w * 2, h * 2] for (x, y, w, h) in pedestrian_rects])
        
        # Convertir de (x, y, w, h) a (x1, y1, x2, y2)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrian_rects])
        
        # Aplicar supresión no máxima
        pick = non_max_suppression(boxes, overlap_thresh=0.4)
        
        # Dibujar los rectángulos resultantes
        for i in pick:
            (x, y, w, h) = pedestrian_rects[i]
            color = (0, 0, 255)  # Rojo para peatones
            
            # Dibujar el rectángulo
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Mostrar etiqueta con el valor de confianza (corregido)
            confidence = weights[i] if weights.ndim == 1 else weights[i][0]
            label = f"Peaton: {confidence:.2f}"
            draw_text(result, label, (x, y - 10), color)
            
    # Añadir texto indicativo
    draw_text(result, f"Peatones detectados: {len(pick) if 'pick' in locals() else 0}", (20, 70))
    
    return result

def detect_pedestrians_contours(frame):
    """
    Método alternativo para detectar movimiento de peatones usando detección de cambios.
    Se puede utilizar como complemento al detector HOG.
    
    Args:
        frame: Current video frame
        
    Returns:
        Frame with detected movement areas
    """
    # Esta función requeriría acceso a frames anteriores para comparar
    # En una implementación completa, necesitaríamos almacenar frames anteriores
    # Aquí se muestra una implementación simple para detección de movimiento
    
    # Supongamos que tenemos un frame anterior
    if not hasattr(detect_pedestrians_contours, "prev_frame"):
        detect_pedestrians_contours.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_pedestrians_contours.prev_frame = cv2.GaussianBlur(detect_pedestrians_contours.prev_frame, (21, 21), 0)
        return frame
    
    # Preparar frame actual
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Calcular diferencia absoluta
    frame_delta = cv2.absdiff(detect_pedestrians_contours.prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Dilatar para llenar huecos
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Actualizar frame anterior
    detect_pedestrians_contours.prev_frame = gray
    
    # Dibujar contornos significativos
    result = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Filtrar contornos pequeños
            continue
        
        (x, y, w, h) = cv2.boundingRect(contour)
        # Dentro del bucle de dibujado
        aspect_ratio = h / float(w)
        if aspect_ratio < 1.5 or aspect_ratio > 4.0:
            continue  # No parece una persona

        # # Descartar formas que no parecen personas
        # if h < 1.5 * w:  # Las personas suelen ser más altas que anchas
        #     continue 
        
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    return result