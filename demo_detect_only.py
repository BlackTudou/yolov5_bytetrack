"""
简单的YOLOv5s人体检测Demo
只检测视频中的人，不进行跟踪、距离计算等功能
"""

import cv2
import time
from detection import YOLOv5Detector


def detect_persons_in_video(video_path, model_path='yolov5s.pt', device='cpu'):
    """
    在视频中检测人员

    Args:
        video_path: 视频文件路径
        model_path: YOLOv5模型路径
        device: 使用设备 ('cpu' 或 'cuda')
    """
    print("正在加载YOLOv5模型...")
    detector = YOLOv5Detector(model_path, device=device)
    print("模型加载完成！")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"\n按 'q' 键退出播放\n")

    frame_count = 0
    total_persons = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 检测人员
        detections = detector.detect(frame, conf_threshold=0.5, iou_threshold=0.45)

        # 绘制检测框
        person_count = 0
        for detection in detections:
            tlwh, score, class_id = detection
            x, y, w, h = tlwh

            # 只处理人体检测 (class_id=0)
            if class_id == 0:
                person_count += 1
                total_persons += 1

                # 绘制边界框 (绿色)
                cv2.rectangle(frame, (int(x), int(y)),
                            (int(x + w), int(y + h)),
                            (0, 255, 0), 2)

                # 绘制标签
                label = f"Person {person_count} {score:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x), int(y) - label_size[1] - 10),
                            (int(x) + label_size[0], int(y)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x), int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 在左上角显示信息
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Persons: {person_count}",
            f"FPS: {fps}"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        # 显示帧
        cv2.imshow('YOLOv5 Person Detection', frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 打印进度
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"已处理 {frame_count}/{total_frames} 帧, 检测到 {person_count} 人, 速度: {fps_actual:.1f} FPS")

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()

    # 打印统计信息
    elapsed = time.time() - start_time
    avg_persons = total_persons / frame_count if frame_count > 0 else 0
    print(f"\n检测完成！")
    print(f"总帧数: {frame_count}")
    print(f"总检测人数: {total_persons}")
    print(f"平均每帧人数: {avg_persons:.2f}")
    print(f"处理时间: {elapsed:.2f} 秒")
    print(f"平均速度: {frame_count/elapsed:.2f} FPS")


if __name__ == '__main__':
    # 配置参数
    VIDEO_PATH = 'palace.mp4'  # 视频文件路径
    MODEL_PATH = 'yolov5s.pt'  # YOLOv5模型路径
    DEVICE = 'cpu'  # 使用 'cuda' 如果有GPU

    print("=" * 60)
    print("YOLOv5s 人体检测 Demo")
    print("=" * 60)
    print()

    try:
        detect_persons_in_video(VIDEO_PATH, MODEL_PATH, DEVICE)
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

