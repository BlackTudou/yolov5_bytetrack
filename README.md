# Factory Worker Safety Distance Monitoring System

## 概述
基于YOLOv5s + ByteTrack的工厂人员安全距离监测与预警系统，用于智慧工厂、建筑工地等安全生产场景。

## 功能特性
- ✅ 实时人体检测（YOLOv5s）
- ✅ 多目标跟踪（ByteTrack算法）
- ✅ 像素距离计算与真实距离估算
- ✅ 安全距离违规检测与预警
- ✅ 可视化显示追踪轨迹和违规情况
- ✅ 支持摄像头和视频文件输入

## 系统要求
- Python 3.7+
- PyTorch 1.9+ (兼容 2.6+)
- OpenCV 4.5+
- pandas 1.3+
- CPU或CUDA加速（推荐GPU）

## 安装步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

**注意**: 如果使用 PyTorch 2.6+ 版本，请确保安装 pandas：
```bash
pip install pandas
```

或在问题查看 `FIX_PYTORCH_2.6.md` 了解详细修复方法。

### 2. 准备模型
确保 `yolov5s.pt` 文件在项目目录下。如果使用预训练模型，可以下载：
```bash
# YOLOv5s模型已包含在项目中
```

## 使用方法

### 使用摄像头实时监测
```bash
python main.py --source camera --device cuda
```

### 使用视频文件
```bash
python main.py --source video --video path/to/video.mp4 --device cuda
```

### 参数说明
- `--source`: 视频源，`camera` 或 `video`
- `--video`: 视频文件路径（当source=video时）
- `--conf`: 检测置信度阈值（默认0.5）
- `--min-distance`: 最小安全距离，单位米（默认1.5米）
- `--device`: 运行设备，`cpu` 或 `cuda`（默认cpu）
- `--camera-id`: 摄像头ID（默认0）

### 示例命令
```bash
# 使用CPU，最低1.8米安全距离
python main.py --source camera --min-distance 1.8 --device cpu

# 使用视频文件，2米安全距离
python main.py --source video --video test.mp4 --min-distance 2.0 --device cuda
```

## 工作原理

### 1. 目标检测
使用YOLOv5s检测图像中的人员，过滤出置信度高于阈值的检测框。

### 2. 多目标跟踪
使用ByteTrack算法对检测到的目标进行跟踪，为每个人分配唯一ID，维持跨帧轨迹连续性。

### 3. 距离计算
- **像素距离**：计算两人底部中心点之间的像素距离
- **真实距离估算**：根据目标尺寸估算与相机的距离，结合像素距离计算真实世界距离

### 4. 违规检测
当两人之间的真实距离小于设定阈值时，触发违规告警。

### 5. 可视化
- 绿色框：正常状态
- 红色连线：违规聚集
- 显示距离标签和违规统计

## 相机标定
当前使用默认标定参数。为获得更准确的真实距离测量，建议进行相机标定：

1. 拍摄已知尺寸的参考物体
2. 调整 `camera_calibrate.py` 中的标定参数
3. 保存标定文件供系统使用

## 系统架构

```
main.py                 # 主程序
tracker.py             # ByteTrack跟踪算法
camera_calibrate.py    # 相机标定与距离估算
requirements.txt       # 依赖包
yolov5s.pt            # 检测模型
```

## ARM平台部署
本系统设计时考虑了ARM平台的兼容性：
1. 使用PyTorch Mobile支持可部署到ARM设备
2. 模型可以转换为ONNX格式进行优化
3. 可配置为使用CPU推理（无需GPU）

### ARM部署步骤
1. 安装ARM版本的PyTorch
2. 将模型转换为ONNX或TorchScript
3. 在ARM设备上运行Python脚本

## 注意事项
1. **标定准确性**：距离估算的准确性依赖于相机标定，建议根据实际环境进行校准
2. **光照条件**：确保良好的光照条件以获得准确的检测
3. **相机角度**：俯视角度可获得更准确的距离测量
4. **性能优化**：对于ARM平台，考虑模型量化或使用更小的模型（如YOLOv5n）

## 故障排除

### CUDA错误
如果遇到CUDA相关错误，使用CPU模式：
```bash
python main.py --device cpu
```

### 模型加载失败
确保 `yolov5s.pt` 文件完整且路径正确

### 摄像头无法打开
检查摄像头ID是否正确，或尝试其他ID：
```bash
python main.py --camera-id 1
```

## 未来改进
- [ ] 自动相机标定功能
- [ ] 轨迹历史记录与回放
- [ ] 数据导出与统计分析
- [ ] Web界面支持
- [ ] 多相机支持与场景拼接
- [ ] 更精确的距离估计算法

## 许可证
MIT License

