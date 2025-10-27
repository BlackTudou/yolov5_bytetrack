"""
Example usage of Safety Distance Monitor System
"""

from main import SafetyDistanceMonitor
import cv2


def example_basic():
    """Basic usage example"""
    print("Example 1: Basic Usage")

    # Initialize monitor
    monitor = SafetyDistanceMonitor(
        model_path='yolov5s.pt',
        conf_threshold=0.5,
        min_distance=1.5,
        device='cpu'
    )

    # Run with camera
    monitor.run(source='camera')


def example_custom():
    """Custom configuration example"""
    print("Example 2: Custom Configuration")

    # Initialize with custom parameters
    monitor = SafetyDistanceMonitor(
        model_path='yolov5s.pt',
        conf_threshold=0.6,  # Higher confidence
        min_distance=2.0,    # 2 meters minimum
        device='cuda',       # Use GPU
        camera_id=0
    )

    # Run with video file
    monitor.run(source='video', video_path='test_video.mp4')


def example_image_processing():
    """Example of processing single images"""
    print("Example 3: Image Processing")

    from detection import YOLOv5Detector
    from tracker import ByteTracker
    from camera_calibrate import CameraCalibrator
    import numpy as np

    # Initialize components
    detector = YOLOv5Detector('yolov5s.pt', device='cpu')
    tracker = ByteTracker()
    calibrator = CameraCalibrator()

    # Load image
    img = cv2.imread('test_image.jpg')
    if img is None:
        print("Error: Could not load image")
        return

    # Detect objects
    detections = detector.detect(img, conf_threshold=0.5)
    print(f"Found {len(detections)} detections")

    # Track objects
    tracked = tracker.update(detections, frame_id=0)
    print(f"Tracking {len(tracked)} objects")

    # Process each tracked object
    for obj in tracked:
        print(f"Track ID: {obj.track_id}, Score: {obj.score:.2f}")
        tlwh = obj.tlwh
        print(f"  Box: x={tlwh[0]:.0f}, y={tlwh[1]:.0f}, "
              f"w={tlwh[2]:.0f}, h={tlwh[3]:.0f}")


def example_calibration():
    """Example of camera calibration"""
    print("Example 4: Camera Calibration")

    from camera_calibrate import CameraCalibrator

    # Initialize calibrator
    calibrator = CameraCalibrator()

    # Example: estimate distance from pixel measurements
    pixel_width = 100
    pixel_height = 200

    distance = calibrator.estimate_distance(pixel_width, pixel_height)
    print(f"Estimated distance: {distance:.2f} meters")

    # Convert pixel distance to real distance
    pixel_dist = 150
    real_dist = calibrator.pixel_to_real_distance(pixel_dist, avg_object_distance=2.0)
    print(f"Real distance: {real_dist:.2f} meters")

    # Save calibration
    calibrator.save_calibration('calibration.pkl')

    # Load calibration
    calibrator2 = CameraCalibrator()
    calibrator2.load_calibration('calibration.pkl')


def example_violation_detection():
    """Example of detecting violations"""
    print("Example 5: Violation Detection")

    monitor = SafetyDistanceMonitor(
        model_path='yolov5s.pt',
        min_distance=1.5,
        device='cpu'
    )

    # Access components
    tracker = monitor.tracker
    calibrator = monitor.calibrator

    # Example: simulate two tracked persons
    from tracker import STrack

    person1 = STrack([10, 10, 50, 100], 0.9, 0)
    person1.activate(0, 1)
    person1.tlwh = [10, 10, 50, 100]

    person2 = STrack([70, 10, 50, 100], 0.8, 0)
    person2.activate(0, 2)
    person2.tlwh = [70, 10, 50, 100]

    # Calculate distance
    pixel_dist, real_dist = monitor.calculate_distance(person1, person2)
    print(f"Pixel distance: {pixel_dist:.1f}")
    print(f"Real distance: {real_dist:.2f} meters")

    # Check violation
    if real_dist < monitor.min_distance:
        print("VIOLATION DETECTED!")
    else:
        print("Distance is safe")


if __name__ == '__main__':
    print("=" * 60)
    print("Safety Distance Monitor - Usage Examples")
    print("=" * 60)
    print()

    # Uncomment the example you want to run

    # example_basic()
    # example_custom()
    # example_image_processing()
    # example_calibration()
    example_violation_detection()

    print()
    print("Done!")

