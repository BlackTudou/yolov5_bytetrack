"""
YOLOv5s + ByteTrack 检测与跟踪 Demo
检测视频中的人并使用ByteTrack进行多目标跟踪
为每个人分配唯一ID并持续跟踪
"""

import cv2
import time
import numpy as np
from detection import YOLOv5Detector
from tracker import ByteTracker

# ==================== 可调整的参数 ====================
# 修改这些参数来优化检测和跟踪效果

# YOLOv5 检测参数
CONF_THRESHOLD = 0.55        # 置信度阈值 (0.3-0.8) 提高=更严格，降低=更宽松
                              # 人群密集场景推荐：0.55（平衡检测精度和漏检）
IOU_THRESHOLD = 0.35         # IoU阈值 (0.3-0.6) 降低=允许更密集，提高=更稀疏
                              # 人群密集场景推荐：0.35（处理重叠）

# ByteTrack 跟踪参数
TRACK_THRESH = 0.5           # 低置信度阈值 (0.4-0.6) 跟踪框的最低置信度
HIGH_THRESH = 0.6            # 高置信度阈值 (0.5-0.7) 区分高低置信度
                              # 人群密集场景推荐：0.6（平衡筛选）
MATCH_THRESH = 0.75          # 匹配阈值 (0.6-0.9) 提高=更稳定，降低=更灵活
                              # 人群密集场景推荐：0.75（平衡稳定性和灵活性）
                              # 如果丢失目标太多，降低到0.7或0.65
TRACK_BUFFER = 50            # 跟踪缓冲帧数 (20-70) 丢失目标后保留的帧数
                              # 人群密集场景推荐：50（处理频繁遮挡，减少ID切换）
FRAME_RATE = 30              # 视频帧率

# 其他参数
IMG_SIZE = 640               # 模型输入尺寸 (416/640/800) 越大越准确但更慢
DEVICE = 'cpu'               # 'cpu' 或 'cuda'

# ==================== 场景预设 (取消注释使用) ====================
# 场景1: 人群密集环境（当前默认）
# CONF_THRESHOLD = 0.55
# IOU_THRESHOLD = 0.35
# HIGH_THRESH = 0.6
# MATCH_THRESH = 0.75
# TRACK_BUFFER = 50

# 场景2: 人群密集且目标丢失严重
# CONF_THRESHOLD = 0.5
# MATCH_THRESH = 0.7
# TRACK_BUFFER = 60

# 场景3: 空旷环境
# CONF_THRESHOLD, IOU_THRESHOLD = 0.45, 0.5
# MATCH_THRESH = 0.8
# TRACK_BUFFER = 30

# 场景4: 快速运动
# IOU_THRESHOLD, TRACK_THRESH, HIGH_THRESH = 0.4, 0.45, 0.55
# MATCH_THRESH = 0.65
# TRACK_BUFFER = 50

# 场景5: 低光照
# CONF_THRESHOLD, IOU_THRESHOLD = 0.4, 0.35
# TRACK_THRESH, HIGH_THRESH, MATCH_THRESH = 0.4, 0.5, 0.7
# TRACK_BUFFER = 60

# ===============================================================


def get_color_by_id(track_id):
    """
    根据跟踪ID生成不同的颜色

    Args:
        track_id: 跟踪ID

    Returns:
        BGR颜色元组
    """
    # 使用HSV颜色空间生成不同颜色
    np.random.seed(track_id)
    hue = (track_id * 31) % 180
    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, color))


def detect_and_track(video_path, model_path='yolov5s.pt', device=None):
    """
    在视频中检测并跟踪人员

    Args:
        video_path: 视频文件路径
        model_path: YOLOv5模型路径
        device: 使用设备 ('cpu' 或 'cuda')，None时使用全局设置
    """
    # 使用全局设备设置
    if device is None:
        device = DEVICE

    print("正在加载YOLOv5模型...")
    print(f"使用设备: {device}, 图像尺寸: {IMG_SIZE}")
    detector = YOLOv5Detector(model_path, device=device)
    print("模型加载完成！")

    # 初始化ByteTracker（使用全局参数）
    tracker = ByteTracker(
        frame_rate=FRAME_RATE,
        track_thresh=TRACK_THRESH,
        high_thresh=HIGH_THRESH,
        match_thresh=MATCH_THRESH,
        track_buffer=TRACK_BUFFER
    )
    print(f"跟踪器配置: track_thresh={TRACK_THRESH}, high_thresh={HIGH_THRESH}, "
          f"match_thresh={MATCH_THRESH}, track_buffer={TRACK_BUFFER}")

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
    print(f"\n按 'q' 键退出, 's' 键暂停/继续\n")

    frame_count = 0
    total_tracks = 0
    track_ids_in_frame = set()
    start_time = time.time()
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 检测人员（使用全局参数）
            detections = detector.detect(frame, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)

            # 更新跟踪器
            tracked_objects = tracker.update(detections, frame_id=frame_count)

            # 统计本帧的跟踪ID
            track_ids_in_frame.clear()
            for obj in tracked_objects:
                track_ids_in_frame.add(obj.track_id)

            # 绘制检测和跟踪结果
            person_count = 0
            for obj in tracked_objects:
                person_count += 1
                total_tracks += 1

                # 获取跟踪信息
                tlwh = obj.tlwh
                track_id = obj.track_id
                score = obj.score
                x, y, w, h = tlwh

                # 获取该ID的颜色
                color = get_color_by_id(track_id)

                # 绘制边界框（根据track_id分配不同颜色）
                cv2.rectangle(frame, (int(x), int(y)),
                            (int(x + w), int(y + h)),
                            color, 3)

                # 绘制标签背景
                label = f"ID:{track_id}  Conf:{score:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x), int(y) - label_size[1] - 10),
                            (int(x) + label_size[0] + 5, int(y)), color, -1)

                # 绘制标签文字
                cv2.putText(frame, label, (int(x) + 2, int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 在左上角显示信息
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Tracks: {person_count}",
                f"Track IDs: {sorted(list(track_ids_in_frame))}",
                f"FPS: {fps}"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # 显示帧
            cv2.imshow('YOLOv5 + ByteTrack Tracking', frame)

        # 按键处理
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            paused = not paused
            if paused:
                print("已暂停，按 's' 继续")
            else:
                print("继续播放")

        # 打印进度
        if frame_count % 30 == 0 and not paused:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"已处理 {frame_count}/{total_frames} 帧, "
                  f"当前 {person_count} 个目标, 速度: {fps_actual:.1f} FPS")

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()

    # 打印统计信息
    elapsed = time.time() - start_time
    avg_tracks = total_tracks / frame_count if frame_count > 0 else 0
    print(f"\n跟踪完成！")
    print(f"总帧数: {frame_count}")
    print(f"总跟踪次数: {total_tracks}")
    print(f"平均每帧跟踪数: {avg_tracks:.2f}")
    print(f"处理时间: {elapsed:.2f} 秒")
    print(f"平均速度: {frame_count/elapsed:.2f} FPS")


if __name__ == '__main__':
    # 配置参数
    VIDEO_PATH = 'palace.mp4'  # 视频文件路径
    MODEL_PATH = 'yolov5s.pt'  # YOLOv5模型路径

    print("=" * 60)
    print("YOLOv5s + ByteTrack 检测与跟踪 Demo")
    print("=" * 60)
    print()
    print("当前参数配置:")
    print(f"  检测阈值: conf={CONF_THRESHOLD}, iou={IOU_THRESHOLD}")
    print(f"  跟踪阈值: track={TRACK_THRESH}, high={HIGH_THRESH}, match={MATCH_THRESH}")
    print(f"  缓冲帧数: {TRACK_BUFFER}")
    print(f"  图像尺寸: {IMG_SIZE}")
    print()
    print("提示: 修改文件顶部的参数来优化效果，查看 调参指南.md 获取详细信息")
    print("=" * 60)
    print()

    try:
        detect_and_track(VIDEO_PATH, MODEL_PATH, DEVICE)
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

