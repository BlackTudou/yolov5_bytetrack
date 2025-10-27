"""
Factory Worker Safety Distance Monitoring and Early Warning System
Using YOLOv5s + ByteTrack for detection and tracking
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from tracker import ByteTracker, STrack
from camera_calibrate import CameraCalibrator
from detection import YOLOv5Detector
import argparse
import time
from typing import List, Tuple
import warnings

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)


class SafetyDistanceMonitor:
    """
    Main system for monitoring worker safety distances
    """
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45,
                 min_distance=1.5, device='cuda', camera_id=0, img_size=1280):
        """
        Initialize the monitoring system

        Args:
            model_path: Path to YOLOv5s model (.pt file)
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            min_distance: Minimum safe distance in meters
            device: Device for inference ('cuda' or 'cpu')
            camera_id: Camera device ID
            img_size: Input image size (larger = more accurate but slower)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_distance = min_distance
        self.device = device
        self.camera_id = camera_id

        # Load YOLOv5s model with larger input size
        print("Loading YOLOv5s model...")
        self.detector = YOLOv5Detector(model_path, device=device, img_size=img_size)

        # Initialize tracker
        self.tracker = ByteTracker()

        # Initialize camera calibration with improved parameters
        # Video dimensions will be set after loading video
        self.calibrator = CameraCalibrator(video_width=1920, video_height=1080)

        # Visualization colors
        self.colors = self._generate_colors(100)

        # Statistics
        self.frame_count = 0
        self.total_violation_frames = 0
        self.last_alert_time = 0

    def _generate_colors(self, n):
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(n):
            r = int(np.sin(0.3 * i) * 127 + 128)
            g = int(np.sin(0.3 * i + 2) * 127 + 128)
            b = int(np.sin(0.3 * i + 4) * 127 + 128)
            colors.append((b, g, r))
        return colors

    def detect_objects(self, img):
        """
        Detect objects using YOLOv5s

        Args:
            img: Input image (BGR)

        Returns:
            List of detections as [tlwh, score, class_id]
        """
        detections = self.detector.detect(img,
                                         conf_threshold=self.conf_threshold)
        return detections

    def calculate_distance(self, person1: STrack, person2: STrack) -> Tuple[float, float]:
        """
        Calculate distance between two tracked persons

        Returns:
            (pixel_distance, real_distance)
        """
        # Get bottom center points (feet location for distance measurement)
        box1 = person1.tlwh
        box2 = person2.tlwh

        # Calculate center points
        center1 = np.array([box1[0] + box1[2] / 2, box1[1] + box1[3]])
        center2 = np.array([box2[0] + box2[2] / 2, box2[1] + box2[3]])

        # Pixel distance
        pixel_dist = np.linalg.norm(center1 - center2)

        # Estimate real distance
        # Average distance of both persons from camera
        avg_dist1 = self.calibrator.estimate_distance(box1[2], box1[3])
        avg_dist2 = self.calibrator.estimate_distance(box2[2], box2[3])
        avg_object_distance = (avg_dist1 + avg_dist2) / 2

        real_dist = self.calibrator.pixel_to_real_distance(pixel_dist, avg_object_distance)

        return pixel_dist, real_dist

    def detect_violations(self, tracked_objects: List[STrack]) -> List[Tuple[int, int, float]]:
        """
        Detect safety distance violations

        Returns:
            List of (id1, id2, real_distance) for violations
        """
        violations = []

        for i, person1 in enumerate(tracked_objects):
            for j, person2 in enumerate(tracked_objects):
                if i >= j:  # Avoid duplicate pairs
                    continue

                pixel_dist, real_dist = self.calculate_distance(person1, person2)

                if real_dist < self.min_distance:
                    violations.append((person1.track_id, person2.track_id, real_dist))

        return violations

    def draw_detections(self, img, tracked_objects, violations):
        """
        Draw detections, tracks, and violation warnings on image
        """
        # Draw tracked objects
        for obj in tracked_objects:
            tlwh = obj.tlwh
            track_id = obj.track_id

            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h

            # Get color based on track ID
            color = self.colors[track_id % len(self.colors)]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw track ID
            label = f'ID:{track_id}'
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw violation lines and warnings
        violation_pairs = set()
        for id1, id2, real_dist in violations:
            # Avoid drawing duplicate pairs
            pair = tuple(sorted([id1, id2]))
            if pair in violation_pairs:
                continue
            violation_pairs.add(pair)

            # Find objects by ID
            obj1 = next((o for o in tracked_objects if o.track_id == id1), None)
            obj2 = next((o for o in tracked_objects if o.track_id == id2), None)

            if obj1 and obj2:
                # Draw line between violating persons
                center1 = [int(obj1.tlwh[0] + obj1.tlwh[2] / 2),
                          int(obj1.tlwh[1] + obj1.tlwh[3])]
                center2 = [int(obj2.tlwh[0] + obj2.tlwh[2] / 2),
                          int(obj2.tlwh[1] + obj2.tlwh[3])]

                cv2.line(img, tuple(center1), tuple(center2), (0, 0, 255), 2)

                # Draw distance label
                mid_point = [(center1[0] + center2[0]) // 2,
                            (center1[1] + center2[1]) // 2]
                label = f'{real_dist:.1f}m'
                cv2.putText(img, label, tuple(mid_point),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw statistics
        stats_text = [
            f'Persons: {len(tracked_objects)}',
            f'Violation Pairs: {len(violation_pairs)}',
            f'Min Safe Distance: {self.min_distance}m'
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(img, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return img

    def run(self, source='camera', video_path=None):
        """
        Run the monitoring system

        Args:
            source: 'camera' or 'video'
            video_path: Path to video file if source is 'video'
        """
        # Initialize video source
        if source == 'camera':
            cap = cv2.VideoCapture(self.camera_id)
        else:
            if not video_path:
                raise ValueError("video_path must be provided when source='video'")
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {source}")

        print(f"Monitoring started. Press 'q' to quit.")
        print(f"Video window should open. If not, check if your environment supports GUI display.")

        fps_counter = time.time()
        fps = 0
        fps_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            fps_frames += 1

            # Detect objects
            detections = self.detect_objects(frame)

            # Update tracker
            tracked_objects = self.tracker.update(detections, self.frame_count)

            # Detect safety violations
            violations = self.detect_violations(tracked_objects)

            # Count unique violation pairs in this frame
            violation_pairs = set(tuple(sorted([v[0], v[1]])) for v in violations)
            if len(violation_pairs) > 0:
                self.total_violation_frames += 1

            # Alert for violations
            if violations and time.time() - self.last_alert_time > 2:
                num_violations = len(violation_pairs)
                num_people = len(tracked_objects)
                if num_people > 0:
                    # Calculate maximum possible pairs: C(n,2) = n*(n-1)/2
                    max_pairs = num_people * (num_people - 1) // 2
                    print(f"WARNING: {num_violations}/{max_pairs} violation pairs ({len(tracked_objects)} people)")
                self.last_alert_time = time.time()

            # Draw results
            frame = self.draw_detections(frame, tracked_objects, violations)

            # Calculate and display FPS
            current_time = time.time()
            if current_time - fps_counter > 1:
                fps = fps_frames / (current_time - fps_counter)
                fps_frames = 0
                fps_counter = current_time

            cv2.putText(frame, f'FPS: {fps:.1f}', (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display result
            cv2.imshow('Safety Distance Monitor', frame)

            # Save frame every 30 frames with detections
            # if self.frame_count % 30 == 0 and len(tracked_objects) > 0:
            #     output_file = f'result_frame_{self.frame_count:04d}.jpg'
            #     cv2.imwrite(output_file, frame)
            #     print(f"✓ Saved frame {self.frame_count} to {output_file}")

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Monitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Factory Worker Safety Distance Monitor')
    parser.add_argument('--model', type=str, default='yolov5s.pt',
                       help='Path to YOLOv5s model')
    parser.add_argument('--source', type=str, default='camera',
                       choices=['camera', 'video'],
                       help='Video source')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (lower = more detections)')
    parser.add_argument('--min-distance', type=float, default=1.5,
                       help='Minimum safe distance in meters')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--img-size', type=int, default=1280,
                       help='Input image size (640, 1280, etc. Larger = more accurate but slower)')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please ensure yolov5s.pt is in the current directory")
        return

    # Initialize monitor
    monitor = SafetyDistanceMonitor(
        model_path=args.model,
        conf_threshold=args.conf,
        min_distance=args.min_distance,
        device=args.device,
        camera_id=args.camera_id,
        img_size=args.img_size
    )

    # Run monitoring
    monitor.run(source=args.source, video_path=args.video)


if __name__ == '__main__':
    main()

