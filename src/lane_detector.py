import cv2
import numpy as np
from utils import auto_canny, region_of_interest, draw_text
from collections import deque

class LaneDetector:
    """
    Clase encargada de detectar carriles en una imagen mediante ajuste polinómico,
    aplicando suavizado temporal y validación para mejorar la estabilidad.
    """
    def __init__(self, smoothing_frames=3):
        """
        Inicializa el detector de carriles.

        Args:
            smoothing_frames (int): Número de frames para aplicar suavizado temporal.
        """
        self.left_fits = deque(maxlen=smoothing_frames)  # Historial de curvas izquierda
        self.right_fits = deque(maxlen=smoothing_frames) # Historial de curvas derecha
        self.frame_count = 0
        self.smoothing_frames = smoothing_frames

        # Ajustes predeterminados (fallback)
        self.last_left_fit = np.array([0, -1.5, 1204.8]) 
        self.last_right_fit = np.array([0, 1.5, 262.6])

        # Rangos válidos para los coeficientes del polinomio
        self.coef_ranges = {
            'a': (-0.001, 0.001),  # Curvatura
            'b': (-1.0, 1.0),      # Pendiente
            'c': (-1000, 1000)     # Intersección (no se valida directamente)
        }
    def enhance_yellow_white(self, frame):
        """
        Realza los colores amarillo y blanco en una imagen.

        Args:
            frame (np.array): Imagen BGR de entrada.

        Returns:
            np.array: Máscara combinada de amarillo y blanco (escala de grises).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]

        # Rangos para el color amarillo (ajustar según sea necesario)
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Rangos para el color blanco (ajustar según sea necesario)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([255, 30, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Combinar las máscaras
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

        # Operaciones morfológicas (opcional)
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        return dilated_mask
    def detect_lanes(self, frame):
        """
        Detección principal de carriles en el frame actual.

        Args:
            frame (np.array): Imagen BGR de entrada.

        Returns:
            np.array: Imagen con los carriles detectados dibujados.
        """
        result = frame.copy()
        height, width = frame.shape[:2]
        
        self.frame_count += 1

        # Solo se actualiza cada N frames o si no hay ajustes previos
        if self.frame_count % self.smoothing_frames == 1 or self.last_left_fit is None:
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # blur = cv2.GaussianBlur(gray, (3, 3), 0)
            blur = cv2.bilateralFilter(enhanced, 5, 75, 75)
            
            
            edges = auto_canny(blur)
            

            # Define una región de interés trapezoidal
            roi_vertices = np.array([[
                (width * 0.12, height),
                (width * 0.35, height * 0.65),
                (width * 0.65, height * 0.65),
                (width * 0.88, height)
            ]], dtype=np.int32)

            masked_edges = region_of_interest(edges, roi_vertices)

            # Detección de líneas con Hough Transform
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi/180,
                threshold=25,
                minLineLength=30,
                maxLineGap=100
            )

            left_points = []
            right_points = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    if x2 - x1 == 0:  # Evita división por cero
                        continue

                    slope = (y2 - y1) / (x2 - x1)
                    angle = abs(np.arctan(slope) * 180 / np.pi)

                    # Filtra líneas con ángulos poco inclinados o muy verticales
                    if angle < 15 or angle > 75:
                        continue

                    # Se ignoran líneas por encima del centro vertical de la imagen
                    if y1 < height * 0.5 and y2 < height * 0.5:
                        continue

                    # Clasifica líneas según su pendiente y posición
                    if slope < 0 and x1 < width // 2:
                        left_points.extend([(x1, y1), (x2, y2)])
                    elif slope > 0 and x1 > width // 2:
                        right_points.extend([(x1, y1), (x2, y2)])

            # Ajuste polinómico y validación
            left_fit = self._fit_polynomial(left_points, height) if left_points else None
            right_fit = self._fit_polynomial(right_points, height) if right_points else None

            left_fit = self._validate_fit(left_fit, "left")
            right_fit = self._validate_fit(right_fit, "right")

            if left_fit is not None:
                self.left_fits.append(left_fit)
                self.last_left_fit = left_fit

            if right_fit is not None:
                self.right_fits.append(right_fit)
                self.last_right_fit = right_fit

        self._draw_lanes(result)
        roi_vertices = np.array([[
                (width * 0.12, height),
                (width * 0.40, height * 0.65),
                (width * 0.60, height * 0.65),
                (width * 0.88, height)
            ]], dtype=np.int32)
        cv2.polylines(result, [roi_vertices], isClosed=True, color=(0, 120, 255), thickness=1)
        draw_text(result, "Deteccion de carriles", (width - 320, 25))

        return result

    def _fit_polynomial(self, points, height):
        """
        Ajusta un polinomio de segundo grado a una lista de puntos.

        Args:
            points (list): Lista de puntos (x, y).
            height (int): Altura de la imagen.

        Returns:
            np.array: Coeficientes del polinomio (a, b, c).
        """
        if not points or len(points) < 5:
            return None

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        # Filtro de outliers por desviación estándar
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
        """
        Valida un ajuste polinómico y lo suaviza si hay cambios bruscos.

        Args:
            fit (np.array): Ajuste a validar.
            side (str): 'left' o 'right'.

        Returns:
            np.array: Ajuste validado (o suavizado).
        """
        if fit is None:
            return None

        a, b, c = fit

        # Verifica que los coeficientes estén dentro de rangos razonables
        if not (self.coef_ranges['a'][0] <= a <= self.coef_ranges['a'][1]):
            return None

        if side == "left" and b > 0:
            return None
        if side == "right" and b < 0:
            return None

        # Comparación con el historial para detectar cambios bruscos
        prev_fits = self.left_fits if side == "left" else self.right_fits
        if prev_fits:
            prev_fit = self._get_smooth_fit(prev_fits)
            prev_a, prev_b, prev_c = prev_fit

            if abs(a - prev_a) > 0.0006 or abs(b - prev_b) > 0.8:
                return np.array([
                    prev_a * 0.9 + a * 0.1,
                    prev_b * 0.9 + b * 0.1,
                    prev_c * 0.9 + c * 0.1
                ])

        return fit

    def _get_smooth_fit(self, fits):
        """
        Promedia los últimos N ajustes para suavizar la detección.

        Args:
            fits (deque): Historial de coeficientes.

        Returns:
            np.array: Ajuste promedio.
        """
        if not fits:
            return None
        coeffs = np.array(fits)
        return np.mean(coeffs, axis=0)

    def _draw_lanes(self, image):
        """
        Dibuja los carriles detectados en la imagen.

        Args:
            image (np.array): Imagen sobre la que se dibujan los carriles.
        """
        height, width = image.shape[:2]
        left_fit = self._get_smooth_fit(self.left_fits) if self.left_fits else self.last_left_fit
        right_fit = self._get_smooth_fit(self.right_fits) if self.right_fits else self.last_right_fit

        if left_fit is not None:
            self._draw_poly_line(image, left_fit, height, (0, 255, 0), 2)
        if right_fit is not None:
            self._draw_poly_line(image, right_fit, height, (0, 255, 0), 2)

    def _draw_poly_line(self, img, fit, height, color, thickness, start_percent=0.70):
        """
        Dibuja una curva polinómica en la imagen.

        Args:
            img (np.array): Imagen destino.
            fit (np.array): Coeficientes del polinomio.
            height (int): Altura de la imagen.
            color (tuple): Color BGR.
            thickness (int): Grosor de la línea.
            start_percent (float): Porcentaje desde donde comenzar a dibujar (para evitar errores cerca del horizonte).
        """
        start_height = int(height * start_percent)
        ploty = np.linspace(start_height, height - 1, 30)
        plotx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]

        pts = np.array([np.transpose(np.vstack([plotx, ploty]))], dtype=np.int32)
        cv2.polylines(img, pts, isClosed=False, color=color, thickness=thickness)

