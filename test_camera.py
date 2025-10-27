"""
Simple test script to verify camera access and basic functionality
"""

import cv2
import sys


def test_camera(camera_id=0):
    """Test camera access"""
    print(f"Testing camera {camera_id}...")

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return False

    print(f"Camera {camera_id} opened successfully!")
    print("Properties:")
    print(f"  Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"  Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    # Try to capture a frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame from camera")
        cap.release()
        return False

    print(f"Frame captured successfully! Shape: {frame.shape}")

    cap.release()
    return True


def list_cameras(max_cameras=5):
    """List available cameras"""
    print("Scanning for available cameras...")
    available = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"Camera {i}: Available")
            cap.release()

    if not available:
        print("No cameras found!")
    else:
        print(f"Found {len(available)} camera(s): {available}")

    return available


if __name__ == '__main__':
    print("=" * 50)
    print("Camera Test Script")
    print("=" * 50)

    # List available cameras
    available = list_cameras()

    if available:
        print("\nTesting first available camera...")
        test_camera(available[0])
    else:
        print("\nNo cameras found. Please check your camera connection.")
        sys.exit(1)

