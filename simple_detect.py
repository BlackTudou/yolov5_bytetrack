"""
简单的YOLOv5s人员检测Demo
只检测视频中的人员，不包含跟踪和距离计算
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class SimplePersonDetector:
    """简单的人员检测器"""

    def __init__(self, model_path='yolov5s.pt', conf_threshold=0.25, device='cpu'):
        """
        初始化检测器

        Args:
            model_path: YOLOv5模型路径
            conf_threshold: 置信度阈值
            device: 运行设备 'cpu' 或 'cuda'
        """
        self.conf_threshold = conf_threshold
        self.device = device

        print("Loading YOLOv5s model...")
        self.model = self._load_model(model_path)
        print(f"Model loaded successfully! Device: {device}")

    def _load_model(self, model_path):
        """加载YOLOv5模型"""
        import torch.hub

        try:
            # 从torch.hub加载YOLOv5s
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                                   pretrained=True, trust_repo=True)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nPlease ensure you have internet connection to download model.")
            sys.exit(1)

    def detect_people(self, img):
        """
        检测图像中的人员

        Args:
            img: BGR图像

        Returns:
            检测结果列表: [[x1, y1, x2, y2, confidence], ...]
        """
        # 转RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 推理
        with torch.no_grad():
            results = self.model(img_rgb)

        detections = []

        # 解析结果
        if hasattr(results, 'pandas'):
            df = results.pandas().xyxy[0]

            for _, row in df.iterrows():
                if float(row['confidence']) < self.conf_threshold:
                    continue

                # 只检测人员（class 0）
                if int(row['class']) != 0:
                    continue

                # 获取坐标
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = float(row['confidence'])

                detections.append([x1, y1, x2, y2, conf])

        return detections

    def draw_detections(self, img, detections):
        """在图像上绘制检测结果"""
        for x1, y1, x2, y2, conf in detections:
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f'Person {conf:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(label,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # 标签背景
            cv2.rectangle(img, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), (0, 255, 0), -1)

            # 标签文字
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 显示统计信息
        cv2.putText(img, f'Detected: {len(detections)} people',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return img

    def process_video(self, video_path, output_path=None):
        """
        处理视频文件

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"\nPress 'q' to quit, 's' to save screenshot\n")

        # 如果需要保存输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 检测人员
            detections = self.detect_people(frame)

            # 绘制结果
            frame = self.draw_detections(frame, detections)

            # 显示帧数
            cv2.putText(frame, f'Frame: {frame_count}/{total_frames}',
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 显示
            cv2.imshow('Person Detection', frame)

            # 保存输出
            if output_path:
                out.write(frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('s'):
                screenshot_path = f'screenshot_{frame_count}.jpg'
                cv2.imwrite(screenshot_path, frame)
                print(f"Saved screenshot to {screenshot_path}")

        cap.release()
        if output_path:
            out.release()
            print(f"\nOutput saved to {output_path}")

        cv2.destroyAllWindows()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Simple Person Detection Demo')
    parser.add_argument('--video', type=str, default='palace.mp4',
                       help='Input video path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device (default: cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')

    args = parser.parse_args()

    # 检查视频文件
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    # 创建检测器
    detector = SimplePersonDetector(
        conf_threshold=args.conf,
        device=args.device
    )

    # 处理视频
    detector.process_video(args.video, args.output)


if __name__ == '__main__':
    main()

