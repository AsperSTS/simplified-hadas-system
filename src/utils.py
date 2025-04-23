import cv2
import numpy as np
import time
from collections import deque

def draw_text(image, text, position, color=(0, 255, 0), thickness=2, size=0.7):
    """Dibuja texto en la imagen con un fondo negro para mejor visibilidad."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, size, thickness)[0]
    
    # Dibujar un rectángulo negro como fondo
    cv2.rectangle(image, 
                 (position[0]-5, position[1]-text_size[1]-5),
                 (position[0]+text_size[0]+5, position[1]+5), 
                 (0, 0, 0), -1)
    
    # Dibujar el texto
    cv2.putText(image, text, position, font, size, color, thickness, cv2.LINE_AA)
    
    return image

def fps_counter(buffer_size=30):
    """Generador para calcular FPS promedio en un buffer de tiempo."""
    buffer = deque(maxlen=buffer_size)
    prev_time = time.time()
    
    while True:
        curr_time = yield len(buffer) / sum(buffer) if buffer else 0
        time_diff = curr_time - prev_time
        buffer.append(time_diff)
        prev_time = curr_time

def region_of_interest(image, vertices):
    """
    Aplica una máscara a la imagen para conservar solo la región de interés.
    
    Args:
        image: Imagen de entrada
        vertices: Array de vértices que definen la región de interés
    
    Returns:
        Imagen con máscara aplicada
    """
    mask = np.zeros_like(image)
    
    # Determinar el número de canales de color
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    # Rellenar polígono
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    # Aplicar la máscara
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def auto_canny(image, sigma=0.33):
    """
    Aplica el detector de bordes Canny con umbrales calculados automáticamente.
    
    Args:
        image: Imagen en escala de grises
        sigma: Factor para ajustar los umbrales
    
    Returns:
        Imagen con bordes detectados
    """
    # Calcular la mediana de intensidades
    v = np.median(image)
    
    # Aplicar la fórmula automática para los umbrales
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    # Aplicar Canny
    return cv2.Canny(image, lower, upper)

def distance_to_camera(known_width, focal_length, perceived_width):
    """
    Estimar la distancia a un objeto usando la relación entre su tamaño real y aparente.
    
    Args:
        known_width: Ancho real del objeto en unidades del mundo (metros)
        focal_length: Distancia focal calibrada
        perceived_width: Ancho del objeto en píxeles
    
    Returns:
        Distancia estimada en las mismas unidades que known_width
    """
    return (known_width * focal_length) / perceived_width

def non_max_suppression(boxes, probs=None, overlap_thresh=0.3):
    """
    Aplica supresión no máxima a las cajas delimitadoras.
    
    Args:
        boxes: Lista de cajas delimitadoras [x1, y1, x2, y2]
        probs: Probabilidades asociadas a cada caja (opcional)
        overlap_thresh: Umbral de solapamiento para la supresión
    
    Returns:
        Índices de las cajas seleccionadas
    """
    # Si no hay cajas, devolver una lista vacía
    if len(boxes) == 0:
        return []
    
    # Convertir a float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    # Inicializar la lista de índices seleccionados
    pick = []
    
    # Coordenadas de las cajas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calcular el área de las cajas y ordenarlas
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # Mantener bucle mientras queden índices
    while len(idxs) > 0:
        # Tomar el último índice y añadirlo a la lista
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Encontrar coordenadas de solapamiento
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Calcular ancho y alto de solapamiento
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Calcular ratio de solapamiento
        overlap = (w * h) / area[idxs[:last]]
        
        # Eliminar índices con solapamiento por encima del umbral
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    # Devolver solo los índices seleccionados
    return pick