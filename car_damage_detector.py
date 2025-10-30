import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

class CarDamageDetector:
    def __init__(self, model_path: str = None):
        self.model = self._load_model(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Damage severity mapping
        self.damage_severity = {
            'minor': ['scratch', 'paint damage'],
            'moderate': ['dent', 'crack', 'side mirror damage'],
            'severe': ['glass shatter', 'bumper damage', 'door damage']
        }

    def _load_model(self, model_path: str = None) -> YOLO:
        try:
            # Set up PyTorch to allow loading the model
            import torch
            torch.serialization.add_safe_globals([torch.serialization._get_restore_location])
            
            # Load the model
            if model_path and os.path.exists(model_path):
                return YOLO(model_path, verbose=False)
            return YOLO('yolov8n.pt', verbose=False)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try with a different approach if the first one fails
            try:
                from ultralytics.utils.downloads import attempt_download
                
                # Ensure the model file is downloaded
                model_file = attempt_download('yolov8n.pt')
                return YOLO(model_file, verbose=False)
                
            except Exception as e2:
                print(f"Fallback loading failed: {e2}")
                # As a last resort, try with safe loading disabled
                try:
                    import torch
                    torch.backends.cudnn.benchmark = True
                    if model_path and os.path.exists(model_path):
                        return YOLO(model_path, verbose=False)
                    return YOLO('yolov8n.pt', verbose=False)
                except Exception as e3:
                    print(f"Final loading attempt failed: {e3}")
                    raise RuntimeError("Failed to load the model after multiple attempts. Please check the logs for details.")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply strong contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray)
        
        # Use a combination of different thresholding methods
        # Method 1: Simple thresholding
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        thresh2 = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 7
        )
        
        # Method 3: Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Combine the thresholding methods
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, edges)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        
        # Close small holes and connect nearby regions
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Dilate to ensure damage areas are connected
        dilated = cv2.dilate(opened, kernel, iterations=1)
        
        return dilated

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def detect_damage(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Make a copy of the image to draw on
        result_image = image.copy()
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. Use adaptive thresholding to highlight potential damage
        thresh = cv2.adaptiveThreshold(
            filtered, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 2
        )
        
        # 4. Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 5. Find contours in the processed image
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find damage
        damage_areas = []
        
        for idx, contour in enumerate(contours):
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Skip very small or very large contours
            min_area = 50  # Minimum area in pixels
            max_area = 0.5 * (w * h)  # Maximum area as 50% of image
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Add some padding to the bounding box
            padding = max(3, int(0.01 * min_dim))  # At least 3px or 1% of min dimension
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_rect = min(image.shape[1] - x, w_rect + 2 * padding)
            h_rect = min(image.shape[0] - y, h_rect + 2 * padding)
            
            # Calculate aspect ratio and other features
            aspect_ratio = float(w_rect) / h_rect if w_rect > h_rect else float(h_rect) / w_rect
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Skip shapes that are too circular or too elongated
            if circularity > 0.9 or aspect_ratio > 20:
                continue
            
            # Get the region of interest for texture analysis
            roi = image[y:y+h_rect, x:x+w_rect]
            if roi.size == 0:
                continue
                
            # Convert ROI to grayscale for texture analysis
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(roi_gray)
            
            # Skip regions with too little variation (likely not damage)
            if std_dev < 10:  # Reduced threshold to detect more subtle damage
                continue
            
            # Classify damage type based on shape and texture
            if aspect_ratio > 4.0:
                damage_type = 'scratch' if area < 0.02 * (w * h) else 'scrape'
            elif circularity > 0.7:
                damage_type = 'dent'
            else:
                damage_type = 'scrape'
            
            # Determine severity based on size and type
            if area > 0.15 * (w * h):
                severity = 'severe'
            elif area > 0.03 * (w * h):
                severity = 'moderate'
            else:
                severity = 'minor'
            
            # Add to damage areas
            damage_id = f"damage_{idx}"
            damage_areas.append({
                'id': damage_id,
                'x': int(x),
                'y': int(y),
                'width': int(w_rect),
                'height': int(h_rect),
                'area': float(area),
                'type': damage_type,
                'severity': severity
            })
            
            # Draw on the result image with color based on severity
            if severity == 'severe':
                color = (0, 0, 255)  # Red for severe
            elif severity == 'moderate':
                color = (0, 165, 255)  # Orange for moderate
            else:
                color = (0, 255, 0)  # Green for minor
                
            cv2.rectangle(result_image, (x, y), (x + w_rect, y + h_rect), color, 2)
            
            # Add label
            label = f"{damage_type.capitalize()} ({severity})"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Prepare the damage information
        damage_detected = len(damage_areas) > 0
        
        # Determine overall severity
        if damage_detected:
            severities = [d['severity'] for d in damage_areas]
            if 'severe' in severities:
                overall_severity = 'severe'
            elif 'moderate' in severities:
                overall_severity = 'moderate'
            else:
                overall_severity = 'minor'
        else:
            overall_severity = 'none'
        
        damage_info = {
            'damage_detected': damage_detected,
            'damage_count': len(damage_areas),
            'damage_areas': damage_areas,
            'severity': overall_severity,
            'message': f"Found {len(damage_areas)} damage areas" if damage_detected else 'No damage detected',
            'timestamp': datetime.now().isoformat()
        }
        
        return result_image, damage_info
        
        # Find contours in the processed image
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        damage_areas = []
        min_contour_area = 200  # Slightly increased to filter out tiny noise
        max_area_ratio = 0.4  # Reduced to avoid false positives on large areas
        
        # Sort contours by area (largest first) and process them
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get the main car area (assume the largest contour is the car)
        if contours:
            main_car_contour = max(contours, key=cv2.contourArea)
            car_area = cv2.contourArea(main_car_contour)
            
            # Filter contours that are too small relative to the car
            min_relative_area = 0.001  # 0.1% of car area
            max_relative_area = 0.3    # 30% of car area
            min_contour_area = max(min_contour_area, car_area * min_relative_area)
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip contours that are too small or too large
            if area < min_contour_area or area > (image.shape[0] * image.shape[1] * max_area_ratio):
                continue
                
            # Get bounding box with dynamic padding based on size
            padding = max(3, int(min(10, area ** 0.5 * 0.1)))  # Dynamic padding
            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Skip if the bounding box is too close to the image edges (likely not damage)
            margin = 10
            if (x < margin or y < margin or 
                x + w > image.shape[1] - margin or 
                y + h > image.shape[0] - margin):
                continue
            
            # Calculate aspect ratio and solidity to filter out non-damage shapes
            aspect_ratio = float(w) / h if w > h else float(h) / w
            
            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Skip very narrow or very wide detections, or those with low solidity
            if aspect_ratio > 12 or solidity < 0.3:  # Increased aspect ratio tolerance, added solidity check
                continue
                
            # Get the region of interest with some padding
            roi_padding = min(10, min(w, h) // 2)  # Dynamic padding based on size
            roi = gray[max(0, y-roi_padding):min(y+h+roi_padding, image.shape[0]), 
                      max(0, x-roi_padding):min(x+w+roi_padding, image.shape[1])]
            
            if roi.size == 0:
                continue
                
            # Calculate standard deviation and mean to check for texture and brightness
            std_dev = np.std(roi)
            mean_val = np.mean(roi)
            
            # More sophisticated texture analysis
            if std_dev < 15 or (std_dev < 25 and mean_val < 50):  # Adjusted thresholds
                continue
                
            # Check if the region has enough edge density to be damage
            roi_edges = cv2.Canny(roi, 30, 100)
            edge_density = np.sum(roi_edges > 0) / (roi.shape[0] * roi.shape[1])
            
            if edge_density < 0.05:  # Skip regions with too few edges
                continue
                
            # More sophisticated damage type classification
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calculate extent (area / bounding rectangle area)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Classify based on shape characteristics
            if aspect_ratio > 3.0:
                damage_type = 'scratch' if area < 5000 else 'scrape'
            elif circularity > 0.7 and extent > 0.6:
                damage_type = 'dent'
            elif aspect_ratio > 1.8 or area < 2000:
                damage_type = 'scrape'
            else:
                # Default to scrape for anything that doesn't fit other categories
                damage_type = 'scrape'
            
            # Adjust based on area and edge density
            if area > 5000 and damage_type == 'scratch':
                damage_type = 'scrape'  # Large scratches are more like scrapes
            
            # Add to damage areas with unique ID
            damage_id = f"damage_{idx}"
            damage_areas.append({
                'id': damage_id,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': float(area),
                'type': damage_type,
                'severity': 'moderate'  # Will be updated later
            })
        
        # Assess severity for each damage area
        for damage in damage_areas:
            damage['severity'] = self._assess_damage_severity([damage])
        
        # Sort by area (largest first)
        damage_areas.sort(key=lambda x: x['area'], reverse=True)
        
        # Prepare damage information
        damage_detected = len(damage_areas) > 0
        damage_info = {
            'damage_detected': damage_detected,
            'damage_count': len(damage_areas),
            'damage_areas': damage_areas,
            'severity': self._assess_damage_severity(damage_areas) if damage_detected else 'none',
            'message': f"{len(damage_areas)} damage areas detected" if damage_detected else 'No damage detected'
        }
        
        # Draw all damages on the result image
        for damage in damage_areas:
            x, y = damage['x'], damage['y']
            w, h = damage['width'], damage['height']
            severity = damage['severity']
            damage_type = damage['type']
            
            # Draw semi-transparent overlay
            overlay = result_image.copy()
            color = (0, 0, 255) if severity == 'severe' else (0, 165, 255) if severity == 'moderate' else (0, 255, 0)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
            
            # Draw border
            border_color = (0, 0, 255) if severity == 'severe' else (0, 165, 255) if severity == 'moderate' else (0, 255, 0)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), border_color, 2)
            
            # Add label
            label = f"{damage_type.capitalize()}"
            cv2.putText(result_image, label, (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)
        
        return result_image, damage_info

    def _assess_damage_severity(self, damage_areas: List[Dict[str, Any]]) -> str:
        if not damage_areas:
            return "none"
            
        # Calculate total damaged area
        total_area = sum(d['area'] for d in damage_areas)
        
        # Get damage types
        damage_types = [d['type'] for d in damage_areas]
        type_counts = {t: damage_types.count(t) for t in set(damage_types)}
        
        # Adjust severity based on damage type and area
        severity_scores = []
        for damage in damage_areas:
            area = damage['area']
            damage_type = damage['type']
            
            # Base score from area
            if area < 500:
                base_score = 0.5  # minor
            elif area < 2000:
                base_score = 1.0  # moderate
            else:
                base_score = 1.5  # severe
            
            # Adjust based on damage type
            if damage_type in self.damage_severity['severe']:
                base_score *= 1.5
            elif damage_type in self.damage_severity['moderate']:
                base_score *= 1.2
                
            severity_scores.append(base_score)
        
        # Calculate average severity
        if not severity_scores:
            return "none"
            
        avg_severity = sum(severity_scores) / len(severity_scores)
        
        if avg_severity < 0.7:
            return "minor"
        elif avg_severity < 1.5:
            return "moderate"
        return "severe"