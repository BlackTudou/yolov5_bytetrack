"""
Camera calibration module for real-world distance estimation
"""

import numpy as np
import cv2
import pickle


class CameraCalibrator:
    """
    Camera calibration for estimating real-world distances from pixel measurements
    """
    def __init__(self, calibration_file=None):
        """
        Initialize calibrator

        Args:
            calibration_file: Path to saved calibration data
        """
        if calibration_file and calibration_file.exists():
            self.load_calibration(calibration_file)
        else:
            # Default calibration (will need to be calibrated for real deployment)
            # These are placeholder values
            self.focal_length = 800  # pixels (approximate for typical webcam)
            self.known_distance = 1.0  # meters
            self.known_width = 0.2  # meters (average person width at 1m distance)
            self.real_width_at_known_distance = 0.2  # meters

    def estimate_distance(self, pixel_width, pixel_height):
        """
        Estimate real-world distance from pixel measurements

        Args:
            pixel_width: Width in pixels
            pixel_height: Height in pixels

        Returns:
            Estimated distance in meters
        """
        # Method 1: Using width-based estimation
        if pixel_width > 0:
            # Assuming average person width of 0.4m
            known_real_width = 0.4  # meters
            distance_by_width = (known_real_width * self.focal_length) / pixel_width

        # Method 2: Using height-based estimation
        if pixel_height > 0:
            # Assuming average person height of 1.7m
            known_real_height = 1.7  # meters
            distance_by_height = (known_real_height * self.focal_length) / pixel_height

        # Use average of both methods
        if pixel_width > 0 and pixel_height > 0:
            estimated_distance = (distance_by_width + distance_by_height) / 2
        elif pixel_width > 0:
            estimated_distance = distance_by_width
        elif pixel_height > 0:
            estimated_distance = distance_by_height
        else:
            estimated_distance = 2.0  # default fallback

        return max(0.5, min(10.0, estimated_distance))  # Clamp between 0.5m and 10m

    def pixel_to_real_distance(self, pixel_distance, avg_object_distance=2.0):
        """
        Convert pixel distance to real-world distance

        Args:
            pixel_distance: Distance in pixels
            avg_object_distance: Average distance of objects from camera (meters)

        Returns:
            Real-world distance in meters
        """
        # Simple linear approximation
        # pixel_distance / focal_length gives angular distance
        # Real distance = angular_distance * avg_distance
        angular_distance = pixel_distance / self.focal_length
        real_distance = angular_distance * avg_object_distance

        return real_distance

    def calibrate_from_reference(self, reference_image_path, reference_width_m=2.0):
        """
        Calibrate camera from reference image with known dimensions

        Args:
            reference_image_path: Path to reference image
            reference_width_m: Known width in meters in the reference image
        """
        # This would load an image and measure a known object
        # For now, using default calibration
        print("Warning: Using default calibration. Calibrate manually for accurate results.")

    def save_calibration(self, filepath):
        """Save calibration data to file"""
        data = {
            'focal_length': self.focal_length,
            'known_distance': self.known_distance,
            'known_width': self.known_width,
            'real_width_at_known_distance': self.real_width_at_known_distance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_calibration(self, filepath):
        """Load calibration data from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.focal_length = data['focal_length']
        self.known_distance = data['known_distance']
        self.known_width = data['known_width']
        self.real_width_at_known_distance = data['real_width_at_known_distance']


if __name__ == '__main__':
    # Test calibration
    cal = CameraCalibrator()
    pixel_width = 100
    pixel_height = 200
    distance = cal.estimate_distance(pixel_width, pixel_height)
    print(f"Estimated distance: {distance:.2f}m")

