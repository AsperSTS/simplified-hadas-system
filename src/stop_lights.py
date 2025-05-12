import numpy as np
import cv2
from collections import deque

# ==== UMBRALES HSV CENTRALIZADOS PARA DETECCIÓN DE COLORES ====
# Cada color está definido con un rango bajo y alto en el espacio de color HSV.
# Se definen dos rangos para el rojo, ya que este color se encuentra en ambos extremos del círculo HSV.
COLOR_THRESHOLDS = {
    'red1': ([0, 100, 50], [10, 255, 255]),       # Rojo (primer rango)
    'red2': ([170, 100, 50], [180, 255, 255]),    # Rojo (segundo rango)
    'yellow': ([20, 100, 50], [40, 255, 255]),    # Amarillo
    'green': ([50, 100, 50], [90, 255, 255])      # Verde
}

# ==== HISTORIAL DE COLORES PARA SUAVIZADO ====
# Almacena los últimos 5 colores detectados para suavizar la salida y evitar parpadeos.
previous_colors = deque(maxlen=5)

# ==== FUNCIONES PRINCIPALES ====

def detectar_color_semaforo(frame, bbox):
    """
    Detecta el color predominante del semáforo dentro de una caja delimitadora (bounding box).
    
    Args:
        frame: Imagen en formato BGR.
        bbox: Caja delimitadora en formato (xmin, ymin, xmax, ymax).

    Returns:
        int: Código del color detectado.
            0 = desconocido o sin detección válida,
            1 = rojo,
            2 = amarillo,
            3 = verde.
    """
    xmin, ymin, xmax, ymax = bbox

    # Verificar que las coordenadas estén dentro de los límites del frame
    if not (0 <= xmin < xmax <= frame.shape[1]) or not (0 <= ymin < ymax <= frame.shape[0]):
        return 0

    # Recortar la región de interés (ROI) del frame
    roi = frame[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        return 0

    # Convertir ROI a espacio de color HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Crear máscaras binarias para cada color
    mask_red1 = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['red1']))
    mask_red2 = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['red2']))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['yellow']))
    mask_green = cv2.inRange(hsv, *map(np.array, COLOR_THRESHOLDS['green']))

    # Contar la cantidad de píxeles para cada color
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    # Definir umbral mínimo de píxeles para considerar detección válida
    min_pixel_threshold = 30

    # Comparar las cantidades de píxeles para decidir el color predominante
    if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels > min_pixel_threshold:
        return 1  # Rojo
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > min_pixel_threshold:
        return 2  # Amarillo
    elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > min_pixel_threshold:
        return 3  # Verde
    else:
        return 0  # Indeterminado

def mean_of_colors(colors):
    """
    Determina el color más frecuente de una lista de colores, aplicando un sesgo hacia el rojo
    por motivos de seguridad (mejor prevenir que avanzar).

    Args:
        colors: Lista de enteros representando colores detectados.

    Returns:
        int: Color más frecuente (0-3).
    """
    if not colors:
        return 0

    # Contar frecuencia de cada color (0 a 3)
    counts = np.bincount(colors, minlength=4).astype(float)

    # Aplicar sesgo: aumentar el peso del rojo (índice 1)
    counts[1] *= 1.2

    # Retornar el índice (color) con mayor frecuencia
    return int(np.argmax(counts))

def display_message(frame, message, position=(50, 50), font_size=1, font_color=(255, 255, 255), thickness=2, background_color=(0, 0, 0)):
    """
    Muestra un mensaje de texto con fondo en un frame.

    Args:
        frame: Imagen en la que se mostrará el mensaje.
        message: Texto a mostrar.
        position: Coordenadas (x, y) del texto.
        font_size: Tamaño de fuente.
        font_color: Color del texto.
        thickness: Grosor del texto.
        background_color: Color del rectángulo de fondo.
    """
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
    text_width, text_height = text_size

    # Calcular coordenadas del rectángulo de fondo
    rect_x1 = position[0] - 5
    rect_y1 = position[1] - text_height - 5
    rect_x2 = position[0] + text_width + 5
    rect_y2 = position[1] + 5

    # Dibujar rectángulo de fondo
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)
    # Dibujar el texto sobre el rectángulo
    cv2.putText(frame, message, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, thickness)

def process_stop_light(frame, boxes, class_ids):
    """
    Procesa las detecciones de objetos para identificar y analizar semáforos,
    determinando su color actual, aplicando suavizado y mostrando un mensaje en pantalla.

    Args:
        frame: Imagen actual.
        boxes: Lista de cajas delimitadoras (bounding boxes) para cada objeto detectado.
        class_ids: Lista de IDs de clase correspondientes a cada objeto detectado.

    Returns:
        frame: Imagen modificada con el mensaje visual del estado del semáforo.
    """
    global previous_colors  # Usar el historial global de colores
    stop_light_colors = []

    # Revisar todas las detecciones
    for i, box in enumerate(boxes):
        if class_ids[i] == 9:  # El ID 9 corresponde al semáforo
            color_id = detectar_color_semaforo(frame, box)
            stop_light_colors.append(color_id)

    # Obtener el color predominante en esta iteración
    mean_color = mean_of_colors(stop_light_colors)

    # Añadir al historial para suavizado
    previous_colors.append(mean_color)
    # Obtener el color más frecuente en el historial
    smoothed_color = int(np.bincount(previous_colors).argmax())

    # Decidir el mensaje y color del texto según el semáforo
    if smoothed_color == 0:
        font_var = (255, 255, 255)  # Neutral, sin informacion
        message = "Continue con precaucion"
    elif smoothed_color == 1:
        font_var = (12, 12, 255)    # Rojo
        message = "Detengase completamente"
    elif smoothed_color == 2:
        font_var = (0, 255, 255)    # Amarillo
        message = "Detengase con precaucion"
    elif smoothed_color == 3:
        font_var = (52, 240, 7)     # Verde
        message = "Avance"
    else:
        font_var = (255, 255, 255)
        message = "Estado desconocido"

    # Mostrar el mensaje en el frame
    display_message(
        frame,
        message,
        position=(frame.shape[1] // 2 - 150, 50),  # Centrado horizontalmente
        font_color=font_var,
        font_size=1
    )

    return frame
