import cv2
import numpy as np
from utils import auto_canny, region_of_interest, draw_text
from collections import deque

class LaneDetector:
    def __init__(self, smoothing_frames=5):
        self.left_fits = deque(maxlen=smoothing_frames)
        self.right_fits = deque(maxlen=smoothing_frames)
        self.frame_count = 0
        self.smoothing_frames = smoothing_frames
        self.last_left_fit = np.array([0, -1.5, 1204.8]) 
        self.last_right_fit = np.array([0, 1.5, 262.6])
        self.coef_ranges = {
            'a': (-0.002, 0.002),  # Más margen para curvatura
            'b': (-2.0, 2.0),      # Más margen para pendiente
            'c': (-1000, 1000)     # Aunque 'c' no se usa, mejor darle un rango válido
        }
    def detect_lanes(self, frame):
        """Detecta y marca carriles con curvas polinómicas suavizadas"""
        result = frame.copy()
        height, width = frame.shape[:2]
        
        self.frame_count += 1
        
        # Procesar frame solo cada N frames o si es el primero
        if self.frame_count % self.smoothing_frames == 1 or self.last_left_fit is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = auto_canny(blur)
            
            # Región de interés trapezoidal
            roi_vertices = np.array([[
                (width * 0.12, height),
                (width * 0.42, height * 0.58),
                (width * 0.58, height * 0.58),
                (width * 0.88, height)
            ]], dtype=np.int32)

            
            masked_edges = region_of_interest(edges, roi_vertices)
            
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi/180,
                threshold=25,
                minLineLength=30,     # Líneas más largas
                maxLineGap=100        # Gap menor para evitar conexiones erróneas
            )
            
            left_points = []
            right_points = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    if x2 - x1 == 0:
                        continue
                    
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filtrado mejorado
                    angle = abs(np.arctan(slope) * 180 / np.pi)
                    if angle < 15 or angle > 75:  # Rangos más estrictos
                        continue
                    
                    # Solo incluir puntos en la mitad inferior de la imagen
                    if y1 < height * 0.5 and y2 < height * 0.5:
                        continue
                    
                    if slope < 0 and x1 < width//2:  # Izquierda
                        left_points.extend([(x1, y1), (x2, y2)])
                    elif slope > 0 and x1 > width//2:  # Derecha
                        right_points.extend([(x1, y1), (x2, y2)])
            
            # Ajuste polinómico
            left_fit = self._fit_polynomial(left_points, height) if left_points else None
            right_fit = self._fit_polynomial(right_points, height) if right_points else None
            
            # Verificar validez de los ajustes
            left_fit = self._validate_fit(left_fit, "left")
            right_fit = self._validate_fit(right_fit, "right")
            
            # Almacenar ajustes válidos
            if left_fit is not None:
                self.left_fits.append(left_fit)
                self.last_left_fit = left_fit
            
            if right_fit is not None:
                self.right_fits.append(right_fit)
                self.last_right_fit = right_fit
        
        self._draw_lanes(result)
        
        # Dibujar ROI - Region of interest
        roi_vertices = np.array([[
            (width * 0.12, height),
            (width * 0.42, height * 0.58),
            (width * 0.58, height * 0.58),
            (width * 0.88, height)
        ]], dtype=np.int32)

        cv2.polylines(result, [roi_vertices], isClosed=True, color=(0, 120, 255), thickness=2)
        
        draw_text(result, "Deteccion de carriles", (width - 220, 70))
        
        return result
    
    def _fit_polynomial(self, points, height):
        """Ajusta polinomio de segundo grado a puntos de carriles"""
        if not points or len(points) < 5:  # Requerir más puntos para ajuste confiable
            return None
        
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        # Eliminar outliers usando desviación estándar
        mean_x = np.mean(x)
        std_x = np.std(x)
        valid_indices = np.where(np.abs(x - mean_x) <= 2 * std_x)[0]
        
        if len(valid_indices) < 5:
            return None
        
        x = x[valid_indices]
        y = y[valid_indices]
        
        try:
            fit = np.polyfit(y, x, 2)
            return fit
        except:
            return None
    
    def _validate_fit(self, fit, side):
        """Valida que el ajuste esté dentro de rangos razonables"""
        if fit is None:
            return None
        
        a, b, c = fit
        
        # Verificar coeficientes dentro de rangos aceptables
        if not (self.coef_ranges['a'][0] <= a <= self.coef_ranges['a'][1]):
            return None
        
        # Para líneas izquierdas, pendiente debe ser negativa
        if side == "left" and b > 0:
            return None
        
        # Para líneas derechas, pendiente debe ser positiva
        if side == "right" and b < 0:
            return None
        
        # Si hay un ajuste anterior, verificar que no sea muy diferente
        prev_fits = self.left_fits if side == "left" else self.right_fits
        if prev_fits:
            prev_fit = self._get_smooth_fit(prev_fits)
            prev_a, prev_b, prev_c = prev_fit
            
            # Limitar el cambio máximo en coeficientes
            if abs(a - prev_a) > 0.0003 or abs(b - prev_b) > 0.5:
                # Si hay mucho cambio, mantener el anterior con ligera adaptación
                return np.array([
                    prev_a * 0.9 + a * 0.1,
                    prev_b * 0.9 + b * 0.1,
                    prev_c * 0.9 + c * 0.1
                ])
        
        return fit
    
    def _get_smooth_fit(self, fits):
        """Promedia los últimos N ajustes polinómicos"""
        if not fits:
            return None
        
        coeffs = np.array(fits)
        return np.mean(coeffs, axis=0)
    
    def _draw_lanes(self, image):
        """Dibuja carriles como curvas polinómicas suavizadas"""
        height, width = image.shape[:2]
        
        # Obtener ajustes suavizados
        left_fit = self._get_smooth_fit(self.left_fits) if self.left_fits else self.last_left_fit
        right_fit = self._get_smooth_fit(self.right_fits) if self.right_fits else self.last_right_fit
        
        # Dibujar curvas
        if left_fit is not None:
            self._draw_poly_line(image, left_fit, height, (0, 255, 0), 3)
            # print(f"Left: {left_fit}")
        
        if right_fit is not None:
            self._draw_poly_line(image, right_fit, height, (0, 255, 0), 3)
            # print(f"Right: {right_fit}")
    
    def _draw_poly_line(self, img, fit, height, color, thickness, start_percent=0.65):
        """Dibuja una línea polinómica más corta"""
        # Iniciar desde un punto más abajo (60% de la altura)
        start_height = int(height * start_percent)
        ploty = np.linspace(start_height, height-1, 30)
        # print(plo)
        plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        
        # Convertir a puntos enteros
        pts = np.array([np.transpose(np.vstack([plotx, ploty]))], dtype=np.int32)
        
        # Dibujar la curva
        cv2.polylines(img, pts, isClosed=False, color=color, thickness=thickness)

# Variable global para mantener el estado
lane_detector = None

def detect_lanes(frame):
    """Función compatible con el código original"""
    global lane_detector
    if lane_detector is None:
        lane_detector = LaneDetector(smoothing_frames=4)
    return lane_detector.detect_lanes(frame)
