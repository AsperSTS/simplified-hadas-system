import numpy as np
import cv2
from collections import deque

# ==== UMBRALES HSV CENTRALIZADOS ====
COLOR_THRESHOLDS = {
    'red1': ([0, 100, 50], [10, 255, 255]),
    'red2': ([170, 100, 50], [180, 255, 255]),
    'yellow': ([20, 100, 50], [40, 255, 255]),
    'green': ([50, 100, 50], [90, 255, 255])
}

# ==== HISTORIAL PARA SUAVIZADO ====
previous_colors = deque(maxlen=5)

# ==== FUNCIONES PRINCIPALES ====

def detectar_color_semaforo(frame, bbox):
    """
    Detecta el color predominante del semáforo dentro de la región de interés (ROI).
    """
    xmin, ymin, xmax, ymax = bbox

    # Validar coordenadas
    if not (0 <= xmin < xmax <= frame.shape[1]) or not (0 <= ymin < ymax <= frame.shape[0]):
        return 0

    roi = frame[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        return 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Crear máscaras de color
    mask_red1 = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['red1']))
    mask_red2 = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['red2']))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['yellow']))
    mask_green = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['green']))

    # Contar píxeles
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    # Umbral mínimo para detección válida
    min_pixel_threshold = 30
    if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels > min_pixel_threshold:
        return 1
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > min_pixel_threshold:
        return 2
    elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > min_pixel_threshold:
        return 3
    else:
        return 0

def mean_of_colors(colors):
    """
    Devuelve el color más frecuente, con sesgo a rojo por seguridad.
    """
    if not colors:
        return 0

    counts = np.bincount(colors, minlength=4).astype(float)
    counts[1] *= 1.2  # Sesgo a rojo
    return int(np.argmax(counts))

def display_message(frame, message, position=(50, 50), font_size=1, font_color=(255, 255, 255), thickness=2, background_color=(0, 0, 0)):
    """
    Muestra un mensaje con fondo en el frame.
    """
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
    text_width, text_height = text_size

    rect_x1 = position[0] - 5
    rect_y1 = position[1] - text_height - 5
    rect_x2 = position[0] + text_width + 5
    rect_y2 = position[1] + 5

    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)
    cv2.putText(frame, message, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, thickness)

def process_stop_light(frame, boxes, class_ids):
    """
    Procesa las detecciones de semáforos, determina el color predominante, suaviza y muestra mensaje.
    """
    global previous_colors
    stop_light_colors = []

    for i, box in enumerate(boxes):
        if class_ids[i] == 9:  # ID del semáforo
            color_id = detectar_color_semaforo(frame, box)
            stop_light_colors.append(color_id)

    mean_color = mean_of_colors(stop_light_colors)

    previous_colors.append(mean_color)
    smoothed_color = int(np.bincount(previous_colors).argmax())

    if smoothed_color == 0:
        font_var = (255, 255, 255)
        message = "Continue con precaucion"
    elif smoothed_color == 1:
        font_var = (12, 12, 255)
        message = "Detengase completamente"
    elif smoothed_color == 2:
        font_var = (0, 255, 255)
        message = "Detengase con precaucion"
    elif smoothed_color == 3:
        font_var = (52, 240, 7)
        message = "Avance"
    else:
        font_var = (255, 255, 255)
        message = "Estado desconocido"

    display_message(frame, message, position=(frame.shape[1] // 2 - 150, 50),
                    font_color=font_var, font_size=1)

    return frame
