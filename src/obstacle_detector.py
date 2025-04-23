import cv2
import numpy as np
from utils import auto_canny, region_of_interest, draw_text, distance_to_camera
import matplotlib.pyplot as plt

class ObstacleTracker:
    """Clase para rastrear obstáculos a través de frames consecutivos."""
    
    def __init__(self, max_disappeared=10, max_distance=50):
        """
        Inicializa el rastreador de obstáculos.
        
        Args:
            max_disappeared: Número máximo de frames que un objeto puede desaparecer antes de ser eliminado
            max_distance: Distancia euclidiana máxima para considerar que dos detecciones son el mismo objeto
        """
        # Almacena los obstáculos detectados y sus características
        self.objects = {}  # {ID: {"rect": (x, y, w, h), "features": features, "disappeared": count}}
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Inicializar detector de características
        self.feature_detector = cv2.SIFT_create()  # Podemos usar SIFT, ORB, AKAZE según rendimiento
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def extract_features(self, frame, rect):
        """
        Extrae características de la región del obstáculo.
        
        Args:
            frame: Imagen completa
            rect: Rectángulo del obstáculo (x, y, w, h)
        
        Returns:
            Características detectadas (keypoints y descriptores)
        """
        x, y, w, h = rect
        # Asegurarse de que los límites están dentro de la imagen
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return None, None
        
        # Extraer la región de interés (ROI)
        roi = frame[y:y+h, x:x+w]
        
        # Convertir a escala de grises si es necesario
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Detectar keypoints y calcular descriptores
        keypoints, descriptors = self.feature_detector.detectAndCompute(roi_gray, None)
        
        # Si no se encuentran características suficientes, retornar None
        if descriptors is None or len(descriptors) < 5:
            return None, None
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Empareja características entre dos descriptores.
        
        Args:
            desc1: Descriptores del primer objeto
            desc2: Descriptores del segundo objeto
        
        Returns:
            Puntuación de similitud (número de coincidencias / total de características)
        """
        if desc1 is None or desc2 is None or len(desc1) < 5 or len(desc2) < 5:
            return 0
        
        # Encontrar coincidencias entre descriptores
        matches = self.feature_matcher.match(desc1, desc2)
        
        # Si no hay suficientes coincidencias, retornar baja similitud
        if len(matches) < 3:
            return 0
        
        # Calcular puntuación normalizada (0-1)
        score = len(matches) / min(len(desc1), len(desc2))
        return score
    
    def update(self, frame, rects):
        """
        Actualiza el rastreador con nuevas detecciones.
        
        Args:
            frame: Frame actual
            rects: Lista de rectángulos detectados [(x, y, w, h), ...]
        
        Returns:
            Diccionario de objetos rastreados con sus IDs
        """
        # Si no hay detecciones, incrementar contador de desapariciones
        if len(rects) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]["disappeared"] += 1
                
                # Eliminar objetos que han desaparecido por mucho tiempo
                if self.objects[obj_id]["disappeared"] > self.max_disappeared:
                    del self.objects[obj_id]
            
            return self.objects
        
        # Extraer características de las nuevas detecciones
        new_features = []
        for rect in rects:
            keypoints, descriptors = self.extract_features(frame, rect)
            new_features.append((rect, keypoints, descriptors))
        
        # Si no hay objetos rastreados, registrar todos como nuevos
        if len(self.objects) == 0:
            for rect, keypoints, descriptors in new_features:
                if descriptors is not None:
                    self.objects[self.next_id] = {
                        "rect": rect,
                        "keypoints": keypoints,
                        "descriptors": descriptors,
                        "disappeared": 0
                    }
                    self.next_id += 1
        
        # Si hay objetos rastreados, intentar emparejarlos con las nuevas detecciones
        else:
            # IDs de objetos existentes y banderas para detecciones usadas
            object_ids = list(self.objects.keys())
            used_objects = set()
            used_rects = set()
            
            # Calcular similitud entre objetos existentes y nuevas detecciones
            for i, obj_id in enumerate(object_ids):
                if self.objects[obj_id]["descriptors"] is None:
                    continue
                
                best_match = -1
                best_score = 0.3  # Umbral mínimo de similitud
                
                for j, (rect, _, descriptors) in enumerate(new_features):
                    if j in used_rects or descriptors is None:
                        continue
                    
                    # Calcular similitud basada en características
                    score = self.match_features(self.objects[obj_id]["descriptors"], descriptors)
                    
                    # Si la similitud es alta, guardar como mejor coincidencia
                    if score > best_score:
                        best_score = score
                        best_match = j
                
                # Si se encontró una buena coincidencia
                if best_match != -1:
                    # Actualizar objeto existente con nueva posición y características
                    rect, keypoints, descriptors = new_features[best_match]
                    self.objects[obj_id]["rect"] = rect
                    self.objects[obj_id]["keypoints"] = keypoints
                    self.objects[obj_id]["descriptors"] = descriptors
                    self.objects[obj_id]["disappeared"] = 0
                    
                    # Marcar como utilizados
                    used_objects.add(obj_id)
                    used_rects.add(best_match)
            
            # Registrar detecciones no emparejadas como nuevos objetos
            for j, (rect, keypoints, descriptors) in enumerate(new_features):
                if j not in used_rects and descriptors is not None:
                    self.objects[self.next_id] = {
                        "rect": rect,
                        "keypoints": keypoints,
                        "descriptors": descriptors,
                        "disappeared": 0
                    }
                    self.next_id += 1
            
            # Incrementar contador de desapariciones para objetos no emparejados
            for obj_id in object_ids:
                if obj_id not in used_objects:
                    self.objects[obj_id]["disappeared"] += 1
                    
                    # Eliminar objetos que han desaparecido por mucho tiempo
                    if self.objects[obj_id]["disappeared"] > self.max_disappeared:
                        del self.objects[obj_id]
        
        return self.objects


# Inicializar el rastreador de obstáculos como variable global
obstacle_tracker = ObstacleTracker(max_disappeared=5, max_distance=50)

def detect_obstacles(frame):
    """
    Detecta obstáculos en la carretera usando técnicas de segmentación y detección de contornos,
    con seguimiento de objetos mediante emparejamiento de características.
    
    Args:
        frame: Imagen de entrada
    
    Returns:
        Imagen con obstáculos marcados
    """
    global obstacle_tracker
    
    # Crear una copia del frame
    result = frame.copy()
    height, width = frame.shape[:2]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Detección de bordes adaptativa
    edges = auto_canny(blurred)
    
    # Definir región de interés - camino por delante
    roi_vertices = np.array([
        [(0, height),(0, height-(height//4)), (width//3+(height//7), height//1.8), (2*width//3-(height//7), height//1.8), (width, height-(height//4)),(width, height)]
    ], dtype=np.int32)
    
    # Aplicar máscara
    masked_edges = region_of_interest(edges, roi_vertices)
    
    
    # plt.imshow(masked_edges, cmap='gray')
    # plt.title('Bordes Enmascarados')
    # plt.savefig('bordes_enmascarados_alta_calidad.jpg')
    
    # Encontrar contornos
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos de obstáculos
    obstacle_rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrar por área para reducir falsos positivos
        if area > 100 and area < 10000:  # Ajustar estos valores según necesidad
            # Calcular rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar por posición y proporción
            if y > height // 2 and h > 20:  # Obstáculos en la mitad inferior
                obstacle_rects.append((x, y, w, h))
    
    # Actualizar el rastreador con los nuevos rectángulos
    tracked_objects = obstacle_tracker.update(frame, obstacle_rects)
    
    # Dibujar obstáculos rastreados
    obstacle_count = 0
    for obj_id, obj_data in tracked_objects.items():
        if obj_data["disappeared"] == 0:  # Solo mostrar objetos actualmente visibles
            x, y, w, h = obj_data["rect"]
            
            # Dibujar rectángulo con ID único
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 165, 0), 2)
            
            # Estimar "distancia" basada en la posición vertical
            distance = ((height - y) / height) * 20  # Valor aproximado en metros
            
            # Mostrar información
            info_text = f"ID:{obj_id}, {distance:.1f}m, W:{w}, H:{h}"
            draw_text(result, info_text, (x, y - 5), (255, 165, 0))
            
            obstacle_count += 1
    
    # Dibujar polígono de región de interés
    cv2.polylines(result, [roi_vertices], isClosed=True, color=(255, 165, 0), thickness=2)
    
    # Añadir texto indicativo
    draw_text(result, f"Obstaculos: {obstacle_count}", (20, 100))
    
    return result

def detect_obstacles_depth(frame):
    """
    Método alternativo para detectar obstáculos usando diferencias de color y textura.
    
    Args:
        frame: Imagen de entrada
    
    Returns:
        Imagen con obstáculos marcados
    """
    # Crear una copia del frame
    result = frame.copy()
    height, width = frame.shape[:2]
    
    # Convertir a espacio de color LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Separar canales
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Aplicar ecualización de histograma adaptativa al canal L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    # Recomponer imagen mejorada
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Convertir a escala de grises
    enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    
    # Umbralización adaptativa
    binary = cv2.adaptiveThreshold(
        enhanced_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Aplicar operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Definir región de interés
    roi_vertices = np.array([
        [(0, height), (width//3, height//2), (2*width//3, height//2), (width, height)]
    ], dtype=np.int32)
    
    # Aplicar máscara
    masked_binary = region_of_interest(opening, roi_vertices)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar y dibujar contornos de obstáculos
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrar por área
        if area > 200 and area < 15000:
            # Calcular rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar por posición
            if y > height // 2:
                # Dibujar rectángulo
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 165, 255), 2)
    
    return result

# import cv2
# import numpy as np
# from utils import auto_canny, region_of_interest, draw_text, distance_to_camera
# import matplotlib.pyplot as plt

# def detect_obstacles(frame):
#     """
#     Detecta obstáculos en la carretera usando técnicas de segmentación y detección de contornos.
    
#     Args:
#         frame: Imagen de entrada
    
#     Returns:
#         Imagen con obstáculos marcados
#     """
#     # Crear una copia del frame
#     result = frame.copy()
#     height, width = frame.shape[:2]
    
#     # Convertir a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Aplicar desenfoque
#     blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
#     # Detección de bordes adaptativa
#     edges = auto_canny(blurred)
    
#     # Definir región de interés - camino por delante
#     roi_vertices = np.array([
#         [(0, height), (width//3, height//2), (2*width//3, height//2), (width, height)]
#     ], dtype=np.int32)
    
#     # Aplicar máscara
#     masked_edges = region_of_interest(edges, roi_vertices)
#     # Mostrar la imagen enmascarada
#     # plt.imshow(masked_edges, cmap='gray') # 'cmap' se usa para especificar el mapa de colores (gris para imágenes en escala de grises)
#     # plt.title('Bordes Enmascarados') # Opcional: añadir un título
#     # plt.savefig('Bordes_Enmascarados.jpg')
#     # Encontrar contornos
#     contours, _ = cv2.findContours(masked_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filtrar y dibujar contornos de obstáculos
#     obstacle_count = 0
#     for contour in contours:
#         area = cv2.contourArea(contour)
        
#         # Filtrar por área para reducir falsos positivos
#         if area > 100 and area < 10000:  # Ajustar estos valores según necesidad
#             # Calcular rectángulo delimitador
#             x, y, w, h = cv2.boundingRect(contour)
            
#             # Filtrar por posición y proporción
#             if y > height // 2 and h > 20:  # Obstáculos en la mitad inferior
#                 # Dibujar rectángulo
#                 cv2.rectangle(result, (x, y), (x + w, y + h), (255, 165, 0), 2)
                
#                 # Estimar "distancia" basada en la posición vertical
#                 # (Una calibración real requeriría más trabajo)
#                 distance = ((height - y) / height) * 20  # Valor aproximado en metros
                
#                 # Mostrar información
#                 # info_text = f"Obstaculo: {distance:.1f}m"
#                 info_text = f"{distance:.1f}m, W:{w}, H{h}"
#                 draw_text(result, info_text, (x, y - 5), (255, 165, 0))
                
#                 obstacle_count += 1
    
#     # Dibujar polígono de región de interés
#     cv2.polylines(result, [roi_vertices], isClosed=True, color=(255, 165, 0), thickness=2)
    
#     # Añadir texto indicativo
#     draw_text(result, f"Obstaculos: {obstacle_count}", (20, 100))
    
#     return result

# def detect_obstacles_depth(frame):
#     """
#     Método alternativo para detectar obstáculos usando diferencias de color y textura.
    
#     Args:
#         frame: Imagen de entrada
    
#     Returns:
#         Imagen con obstáculos marcados
#     """
#     # Crear una copia del frame
#     result = frame.copy()
#     height, width = frame.shape[:2]
    
#     # Convertir a espacio de color LAB
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
#     # Separar canales
#     l_channel, a_channel, b_channel = cv2.split(lab)
    
#     # Aplicar ecualización de histograma adaptativa al canal L
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_l = clahe.apply(l_channel)
    
#     # Recomponer imagen mejorada
#     enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
#     enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
#     # Convertir a escala de grises
#     enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    
#     # Umbralización adaptativa
#     binary = cv2.adaptiveThreshold(
#         enhanced_gray, 
#         255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY_INV, 
#         11, 
#         2
#     )
    
#     # Aplicar operaciones morfológicas
#     kernel = np.ones((5, 5), np.uint8)
#     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
#     # Definir región de interés
#     roi_vertices = np.array([
#         [(0, height), (width//3, height//2), (2*width//3, height//2), (width, height)]
#     ], dtype=np.int32)
    
#     # Aplicar máscara
#     masked_binary = region_of_interest(opening, roi_vertices)
    
#     # Encontrar contornos
#     contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filtrar y dibujar contornos de obstáculos
#     for contour in contours:
#         area = cv2.contourArea(contour)
        
#         # Filtrar por área
#         if area > 200 and area < 15000:
#             # Calcular rectángulo delimitador
#             x, y, w, h = cv2.boundingRect(contour)
            
#             # Filtrar por posición
#             if y > height // 2:
#                 # Dibujar rectángulo
#                 cv2.rectangle(result, (x, y), (x + w, y + h), (0, 165, 255), 2)
    
#     return result