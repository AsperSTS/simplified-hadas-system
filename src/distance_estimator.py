import cv2
import numpy as np
import math

class DistanceEstimator:
    def __init__(self, camera_height=1.2, camera_pitch=0, focal_length=1800, fov_vertical=70, fov_horizontal=90):
        """
        Initialize distance estimator with camera parameters
        
        Args:
            camera_height: Height of camera from ground in meters
            camera_pitch: Camera pitch angle in degrees (tilt down from horizontal)
            focal_length: Focal length in pixels
            fov_vertical: Vertical field of view in degrees
            fov_horizontal: Horizontal field of view in degrees
        """
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch * math.pi / 180  # Convert to radians
        self.focal_length = focal_length
        self.fov_vertical = fov_vertical * math.pi / 180  # Convert to radians
        self.fov_horizontal = fov_horizontal * math.pi / 180  # Convert to radians
        
        # Known average heights of different object classes (in meters)
        # Increased slightly to make distances appear closer
        self.class_heights = {
            0: 1.7,    # person (average height)
            1: 1.6,    # bicyble
            2: 1.7,    # car (height from ground)
            3: 1.7,    # motorcycle (when considering rider too)
            5: 3.5,    # bus
            7: 3.4,    # truck
            
        }
        
        # Known average widths of different object classes (in meters)
        # Increased slightly to make distances appear closer
        self.class_widths = {
            0: 0.6,    # person (shoulder width)
            1: 0.7,    # bicyble
            2: 1.9,    # car (width)
            3: 0.7,    # motorcycle
            5: 2.55,    # bus
            7: 2.55,    # truck
            
        }
        
        # Default distance thresholds (in meters) for different warning levels
        # Increased thresholds to trigger warnings at greater distances
        self.warning_thresholds = {
            'critical': 7,    # Under 7 meters - critical (was 5)
            'warning': 20,    # 7-20 meters - warning (was 15)
            'safe': float('inf')  # Over 20 meters - safe
        }
    
    def estimate_distance(self, box, class_id, frame_height, frame_width=None):
        """
        Estimate distance to detected object using multiple methods and considering
        both vertical and horizontal position
        
        Args:
            box: Detection box [x1, y1, x2, y2]
            class_id: Class ID of detected object
            frame_height: Height of video frame in pixels
            frame_width: Width of video frame in pixels
            
        Returns:
            distance: Estimated distance in meters
        """
        # Get box dimensions
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        box_width = x2 - x1
        
        # Calculate box center
        box_center_x = (x1 + x2) / 2
        
        # Use bottom center of box for ground reference
        bottom_center_x = box_center_x
        bottom_center_y = y2
        
        # Get real-world dimensions of object class
        real_height = self.class_heights.get(class_id, 1.7)  # Default to human height
        real_width = self.class_widths.get(class_id, 0.6)    # Default to human width
        
        # Calculate distance using multiple methods
        
        # Method 1: Distance based on vertical position (ground plane)
        # Calculate vertical angle from image center
        if frame_width is None:
            # If frame_width is not provided, estimate it based on aspect ratio
            frame_width = int(frame_height * 16 / 9)  # Assume 16:9 aspect ratio
            
        # Calculate normalized coordinates (-1 to 1) from center of image
        # where (0,0) is the center of the image
        x_normalized = (bottom_center_x - frame_width / 2) / (frame_width / 2)
        y_normalized = (frame_height / 2 - bottom_center_y) / (frame_height / 2)
        
        # Calculate angles from center of image
        vertical_angle = y_normalized * (self.fov_vertical / 2)
        horizontal_angle = x_normalized * (self.fov_horizontal / 2)
        
        # Calculate actual angle from camera to object, considering camera pitch
        pitch_angle = self.camera_pitch + vertical_angle
        
        # Calculate forward distance using camera height and pitch angle
        if pitch_angle > 0:
            forward_distance = self.camera_height / math.tan(pitch_angle)
        else:
            # If angle is negative (looking up), use a default large distance
            forward_distance = 50.0
        
        # Calculate lateral distance component using horizontal angle
        lateral_distance = forward_distance * math.tan(horizontal_angle)
        
        # Calculate total ground distance using Pythagorean theorem
        ground_plane_distance = math.sqrt(forward_distance**2 + lateral_distance**2)
        
        # Method 2: Distance based on apparent size (height)
        # Using the pinhole camera model: distance = (real_height * focal_length) / apparent_height
        height_based_distance = (real_height * self.focal_length) / box_height
        
        # Method 3: Distance based on apparent size (width)
        # Similar to method 2 but using width
        width_based_distance = (real_width * self.focal_length) / box_width
        

        # Combine all methods with weighted average
        # Adjust weights based on object position and relative size
        
        # Calculate object position factor (0-1): how centered the object is
        # 0 = at the edge, 1 = perfectly centered
        position_factor = 1.0 - min(1.0, abs(x_normalized))
        
        # Calculate size factor (0-1): how large the object is relative to frame
        # 0 = tiny, 1 = taking up most of the frame
        size_factor = min(1.0, (box_width * box_height) / (frame_width * frame_height) * 20)
        
        # Adjust weights based on factors
        # Give more weight to size-based methods for closer/larger objects
        # Size-based methods tend to estimate shorter distances
        w_ground = 0.2 * position_factor + 0.1  # 0.1-0.3 (reduced weight for ground method)
        w_height = 0.4 * size_factor + 0.3      # 0.3-0.7 (increased weight for height method)
        w_width = 0.3 * size_factor + 0.2       # 0.2-0.5 (increased weight for width method)
        
        # Normalize weights to sum to 1
        total_weight = w_ground + w_height + w_width 
        w_ground /= total_weight
        w_height /= total_weight
        w_width /= total_weight
        
        # Compute weighted average
        distance = (
            w_ground * ground_plane_distance +
            w_height * height_based_distance +
            w_width * width_based_distance 
        )
        
        distance = np.subtract(distance, 13.9)

        # Apply reasonability constraints
        distance = max(1.5, min(100.0, distance))  # Limit to 1-100 meters range
        return distance
    
    def get_warning_level(self, distance):
        """Determine warning level based on distance"""
        if distance <= self.warning_thresholds['critical']:
            return 'critical', (0, 0, 255)  # Red
        elif distance <= self.warning_thresholds['warning']:
            return 'warning', (0, 165, 255)  # Orange
        else:
            return 'safe', (0, 180, 0)  # Green

def draw_distance_info(frame, box, distance, warning_level, warning_color):
    """Draw distance information and warning on frame"""
    x1, y1, x2, y2 = box
    
    # Draw distance text
    distance_text = f"{distance:.1f}m"
    cv2.putText(frame, distance_text, (x1, y1 - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, warning_color, 2)
    
    # Draw warning indicator
    if warning_level == 'critical':
        # Draw red flashing triangle for critical warning
        triangle_pts = np.array([
            [(x1 + x2) // 2, y1 - 30],
            [x1 + (x2 - x1) // 4, y1 - 10],
            [x2 - (x2 - x1) // 4, y1 - 10]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [triangle_pts], warning_color)
        cv2.putText(frame, "!", ((x1 + x2) // 2 - 5, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    

# Function to integrate with your existing obstacle detection
def add_distance_estimation(frame, boxes, class_ids, distance_estimator=None):
    """
    Add distance estimation to detected objects
    
    Args:
        frame: Current video frame
        boxes: List of detection boxes [x1, y1, x2, y2]
        class_ids: List of class IDs for each box
        distance_estimator: DistanceEstimator instance or None
    
    Returns:
        frame: Frame with distance information added
    """
    if distance_estimator is None:
        # Initialize with adjusted parameters to detect shorter distances
        distance_estimator = DistanceEstimator(
            camera_height=1,   # Reduced camera height (from 1.3m)
            camera_pitch=4,      # Slightly increased pitch (from 4 degrees)
            focal_length=1500,   # Reduced focal length (from 2500 pixels)
            fov_vertical=60,     # Increased vertical FOV (from 60 degrees)
            fov_horizontal=85    # Increased horizontal FOV (from 80 degrees)
        )
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
    for i, box in enumerate(boxes):
        class_id = class_ids[i]
        
        # Only estimate distance for vehicles and people
        if class_id in [0, 1, 2, 3, 5, 7]:  # person, bicycle,  car, motorbike, bus, truck, bike
            # Calculate distance
            distance = distance_estimator.estimate_distance(box, class_id, height, width)
            
            # Get warning level
            warning_level, warning_color = distance_estimator.get_warning_level(distance)
            
            # Draw information on frame
            draw_distance_info(result, box, distance, warning_level, warning_color)
            
    return result