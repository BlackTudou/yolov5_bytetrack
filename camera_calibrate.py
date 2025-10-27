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
    def __init__(self, calibration_file=None, video_width=1920, video_height=1080, camera_height=3.0):
        """
        Initialize calibrator

        Args:
            calibration_file: Path to saved calibration data
            video_width: Video width in pixels
            video_height: Video height in pixels
            camera_height: Camera height above ground in meters (for perspective correction)
        """
        self.video_width = video_width
        self.video_height = video_height
        self.camera_height = camera_height

        if calibration_file and calibration_file.exists():
            self.load_calibration(calibration_file)
        else:
            # Improved default calibration
            # Estimate focal length based on video resolution
            self.focal_length = max(video_width, video_height) * 0.8  # Adaptive focal length

            # Person dimensions (more accurate estimates)
            self.person_height = 1.65  # Average person height (meters)
            self.person_width = 0.45  # Average shoulder width (meters)

            # Camera intrinsic parameters (estimated)
            self.camera_height = camera_height  # Height above ground

            self.known_distance = 1.0
            self.known_width = 0.2
            self.real_width_at_known_distance = 0.2

    def estimate_distance(self, pixel_width, pixel_height):
        """
        Estimate real-world distance from pixel measurements (Improved version)
        Uses weighted average and accounts for perspective

        Args:
            pixel_width: Width in pixels
            pixel_height: Height in pixels

        Returns:
            Estimated distance in meters
        """
        if pixel_width <= 0 or pixel_height <= 0:
            return 2.0  # fallback

        # Method 1: Using height-based estimation (more stable)
        # The height of a person is more consistent than width
        if pixel_height > 0:
            # Improved formula: distance = (known_height ¡Á focal_length) / pixel_height
            # With small correction factor for perspective
            distance_by_height = (self.person_height * self.focal_length) / pixel_height

            # Perspective correction (closer objects appear larger in pixels)
            # Apply correction based on estimated distance
            if distance_by_height > 2.0:
                # For distant objects, apply perspective correction
                distance_by_height *= 1.1

        # Method 2: Using width-based estimation (less reliable)
        if pixel_width > 0:
            distance_by_width = (self.person_width * self.focal_length) / pixel_width

            # Width varies more than height, so apply variance correction
            if distance_by_width > distance_by_height * 1.5 or distance_by_width < distance_by_height * 0.7:
                # If width estimate deviates too much, trust height more
                distance_by_width = distance_by_height

        # Weighted average (70% height, 30% width)
        # Height is generally more reliable as it varies less with pose
        if pixel_width > 0 and pixel_height > 0:
            estimated_distance = 0.7 * distance_by_height + 0.3 * distance_by_width
        elif pixel_height > 0:
            estimated_distance = distance_by_height
        elif pixel_width > 0:
            estimated_distance = distance_by_width
        else:
            estimated_distance = 2.0

        # Clamp to reasonable range
        return max(1.0, min(8.0, estimated_distance))

    def pixel_to_real_distance(self, pixel_distance, avg_object_distance=2.0):
        """
        Convert pixel distance to real-world distance (Improved version)

        Uses corrected perspective transformation that accounts for:
        1. The angular relationship between pixel and real-world coordinates
        2. The perspective distortion at different distances

        Args:
            pixel_distance: Distance in pixels
            avg_object_distance: Average distance of objects from camera (meters)

        Returns:
            Real-world distance in meters
        """
        if pixel_distance <= 0:
            return avg_object_distance * 0.1  # minimum 10% of avg distance

        # Improved formula that considers perspective geometry
        # The relationship is not purely linear due to perspective

        # Calculate angular distance (in radians)
        # The pixel represents an angle: angle = pixel / focal_length (in radians)
        angular_distance_rad = pixel_distance / self.focal_length

        # Convert to meters using average object distance
        # For perspective: real_distance ¡Ö angular_distance * avg_distance
        # Add correction factor for non-linear perspective
        real_distance = angular_distance_rad * avg_object_distance

        # Apply perspective correction
        # Objects closer to camera appear larger (non-linear scaling)
        if avg_object_distance < 2.0:
            # For close objects, reduce the scale
            real_distance *= 0.85
        elif avg_object_distance > 4.0:
            # For distant objects, the perspective effect is smaller
            real_distance *= 1.05

        # Ensure minimum reasonable distance
        return max(0.2, real_distance)

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

