import cv2
import numpy as np
import math

class DistanceEstimator:
    def __init__(self, camera_height=1.2, camera_pitch=0, focal_length=1800, fov_vertical=70, fov_horizontal=90):
        """
        Inicializa el estimador de distancia con los parámetros de la cámara.

        Args:
            camera_height: Altura de la cámara desde el suelo en metros.
            camera_pitch: Ángulo de inclinación de la cámara en grados (hacia abajo desde la horizontal).
            focal_length: Longitud focal en píxeles.
            fov_vertical: Campo de visión vertical en grados.
            fov_horizontal: Campo de visión horizontal en grados.
        """
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch * math.pi / 180  # Convertir a radianes
        self.focal_length = focal_length
        self.fov_vertical = fov_vertical * math.pi / 180  # Convertir a radianes
        self.fov_horizontal = fov_horizontal * math.pi / 180  # Convertir a radianes

        # Alturas promedio conocidas para distintas clases de objetos (en metros)
        self.class_heights = {
            0: 1.7,    # persona
            1: 1.6,    # bicicleta
            2: 1.7,    # auto
            3: 1.7,    # motocicleta (considerando también al conductor)
            5: 3.5,    # autobús
            7: 3.4,    # camión
        }

        # Anchos promedio conocidos para distintas clases de objetos (en metros)
        self.class_widths = {
            0: 0.6,    # persona (ancho de hombros)
            1: 0.7,    # bicicleta
            2: 1.9,    # auto
            3: 0.7,    # motocicleta
            5: 2.55,   # autobús
            7: 2.55,   # camión
        }

        # Umbrales de distancia para los diferentes niveles de advertencia
        self.warning_thresholds = {
            'critical': 7,        # Menor a 7 m - crítico
            'warning': 20,        # Entre 7 y 20 m - advertencia
            'safe': float('inf')  # Mayor a 20 m - seguro
        }

    def estimate_distance(self, box, class_id, frame_height, frame_width=None):
        """
        Estima la distancia al objeto detectado usando varios métodos.

        Args:
            box: Caja delimitadora de detección [x1, y1, x2, y2]
            class_id: ID de clase del objeto detectado
            frame_height: Altura del fotograma en píxeles
            frame_width: Ancho del fotograma en píxeles (opcional)

        Returns:
            distance: Distancia estimada en metros
        """
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        box_width = x2 - x1

        # Centro inferior de la caja, usado como referencia del suelo
        box_center_x = (x1 + x2) / 2
        bottom_center_x = box_center_x
        bottom_center_y = y2

        # Obtener dimensiones reales del objeto
        real_height = self.class_heights.get(class_id, 1.7)
        real_width = self.class_widths.get(class_id, 0.6)

        # Estimar el ancho si no se proporciona (relación de aspecto 16:9)
        if frame_width is None:
            frame_width = int(frame_height * 16 / 9)

        # Coordenadas normalizadas del centro inferior de la caja
        x_normalized = (bottom_center_x - frame_width / 2) / (frame_width / 2)
        y_normalized = (frame_height / 2 - bottom_center_y) / (frame_height / 2)

        # Ángulos desde el centro de la imagen
        vertical_angle = y_normalized * (self.fov_vertical / 2)
        horizontal_angle = x_normalized * (self.fov_horizontal / 2)

        # Ángulo final considerando la inclinación de la cámara
        pitch_angle = self.camera_pitch + vertical_angle

        # Estimar distancia hacia adelante desde la cámara
        if pitch_angle > 0:
            forward_distance = self.camera_height / math.tan(pitch_angle)
        else:
            forward_distance = 50.0  # Valor por defecto si la cámara apunta hacia arriba

        # Componente lateral de la distancia
        lateral_distance = forward_distance * math.tan(horizontal_angle)

        # Distancia total en el plano del suelo
        ground_plane_distance = math.sqrt(forward_distance**2 + lateral_distance**2)

        # Método alternativo: estimación por altura aparente
        height_based_distance = (real_height * self.focal_length) / box_height

        # Método alternativo: estimación por ancho aparente
        width_based_distance = (real_width * self.focal_length) / box_width

        # Factores de ponderación para combinar métodos
        position_factor = 1.0 - min(1.0, abs(x_normalized))
        size_factor = min(1.0, (box_width * box_height) / (frame_width * frame_height) * 20)

        w_ground = 0.2 * position_factor + 0.1
        w_height = 0.4 * size_factor + 0.3
        w_width = 0.3 * size_factor + 0.2

        # Normalizar pesos
        total_weight = w_ground + w_height + w_width
        w_ground /= total_weight
        w_height /= total_weight
        w_width /= total_weight

        # Promedio ponderado de las tres estimaciones
        distance = (
            w_ground * ground_plane_distance +
            w_height * height_based_distance +
            w_width * width_based_distance 
        )

        # Ajuste manual (offset)
        distance = np.subtract(distance, 13.9)

        # Limitar a un rango razonable
        distance = max(1.5, min(100.0, distance))
        return distance

    def get_warning_level(self, distance):
        """
        Determina el nivel de advertencia en función de la distancia estimada.

        Args:
            distance: Distancia estimada al objeto

        Returns:
            warning_level: Nivel ('critical', 'warning', 'safe')
            color: Color en BGR correspondiente al nivel
        """
        if distance <= self.warning_thresholds['critical']:
            return 'critical', (0, 0, 255)  # Rojo
        elif distance <= self.warning_thresholds['warning']:
            return 'warning', (0, 165, 255)  # Naranja
        else:
            return 'safe', (0, 180, 0)  # Verde

def draw_distance_info(frame, box, distance, warning_level, warning_color):
    """
    Dibuja la información de distancia y el nivel de advertencia sobre el fotograma.

    Args:
        frame: Imagen actual del video
        box: Caja de detección [x1, y1, x2, y2]
        distance: Distancia estimada
        warning_level: Nivel de advertencia
        warning_color: Color BGR correspondiente al nivel
    """
    x1, y1, x2, y2 = box

    # Texto con la distancia
    distance_text = f"{distance:.1f}m"
    cv2.putText(frame, distance_text, (x1, y1 - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, warning_color, 2)

    # Indicador visual para nivel crítico
    if warning_level == 'critical':
        triangle_pts = np.array([
            [(x1 + x2) // 2, y1 - 30],
            [x1 + (x2 - x1) // 4, y1 - 10],
            [x2 - (x2 - x1) // 4, y1 - 10]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [triangle_pts], warning_color)
        cv2.putText(frame, "!", ((x1 + x2) // 2 - 5, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def add_distance_estimation(frame, boxes, class_ids, distance_estimator=None):
    """
    Añade la estimación de distancia a los objetos detectados en el fotograma.

    Args:
        frame: Fotograma actual
        boxes: Lista de cajas de detección [x1, y1, x2, y2]
        class_ids: Lista de IDs de clase correspondientes a cada caja
        distance_estimator: Instancia de DistanceEstimator (opcional)

    Returns:
        frame: Fotograma con la información de distancia añadida
    """
    if distance_estimator is None:
        # Inicializar con parámetros predeterminados
        distance_estimator = DistanceEstimator(
            camera_height=1,
            camera_pitch=4,
            focal_length=1500,
            fov_vertical=60,
            fov_horizontal=85
        )

    height, width = frame.shape[:2]
    result = frame.copy()

    for i, box in enumerate(boxes):
        class_id = class_ids[i]

        # Solo estimar para objetos relevantes (vehículos y personas)
        if class_id in [0, 1, 2, 3, 5, 7]:
            distance = distance_estimator.estimate_distance(box, class_id, height, width)
            warning_level, warning_color = distance_estimator.get_warning_level(distance)
            draw_distance_info(result, box, distance, warning_level, warning_color)

    return result
