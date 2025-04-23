import cv2
import numpy as np
from utils import draw_text, non_max_suppression


# Inicializar el sustractor de fondo como variable global
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

def detect_vehicles(frame):
    """
    Detecta vehículos en la imagen utilizando sustracción de fondo y análisis de contornos.
    """
    result = frame.copy()
    height, width = frame.shape[:2]

    # Reducir tamaño para acelerar procesamiento
    small_frame = cv2.resize(frame, (width // 2, height // 2))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar sustracción de fondo
    fgmask = fgbg.apply(blurred)

    # Operaciones morfológicas para limpiar la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar contornos
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:  # Ignorar objetos pequeños
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Filtrar por forma y posición (más anchos que altos, parte inferior del frame)
        if 1.2 < aspect_ratio < 4.5 and y > height // 4:
            # Ajustar al tamaño original del frame
            vehicles.append([x*2, y*2, w*2, h*2])

    # Aplicar supresión no máxima
    if vehicles:
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in vehicles])
        pick = non_max_suppression(boxes, overlap_thresh=0.4)
        vehicles = [vehicles[i] for i in pick]

    # Dibujar resultados
    for (x, y, w, h) in vehicles:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
        draw_text(result, "Vehiculo", (x, y - 5), (0, 255, 255))

    draw_text(result, f"Vehículos detectados: {len(vehicles)}", (20, 130))
    return result


def detect_vehicles_by_color_shape(frame):
    """
    Detecta vehículos basándose en análisis de color y forma.
    
    Args:
        frame: Imagen de entrada
    
    Returns:
        Lista de rectángulos [x, y, w, h] que representan los vehículos detectados
    """
    height, width = frame.shape[:2]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar umbralización adaptativa
    thresh = cv2.adaptiveThreshold(
        blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Aplicar operaciones morfológicas para mejorar la detección
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # Definir región de interés - carretera
    roi_vertices = np.array([
        [(0, height),(0, height-(height//4)), (width//3+(height//7), height//2.1), (2*width//3-(height//7), height//2.1), (width, height-(height//4)),(width, height)]
    ], dtype=np.int32)
    
    # Crear una máscara con la región de interés
    mask = np.zeros_like(closing)
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_closing = cv2.bitwise_and(closing, mask)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(masked_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos
    vehicles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrar por área
        if area > 1000 and area < 50000:
            # Calcular rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar por proporción ancho/alto (los coches suelen ser más anchos que altos)
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 4.0 and y > height // 3:
                vehicles.append([x, y, w, h])
    
    # Aplicar supresión no máxima si hay muchas detecciones solapadas
    if len(vehicles) > 0:
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in vehicles])
        pick = non_max_suppression(boxes, overlap_thresh=0.4)
        vehicles = [vehicles[i] for i in pick]
    
    return vehicles


            # for (x, y, w, h) in vehicles:
            #     cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0