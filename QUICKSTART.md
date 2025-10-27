# 快速开始指南

## 安装环境

### Windows
```bash
# 1. 创建虚拟环境（可选但推荐）
python -m venv venv
venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 如果使用GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Linux/Mac
```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip3 install -r requirements.txt
```

## 快速测试

### 1. 测试摄像头（可选）
```bash
python test_camera.py
```

这将扫描并测试可用的摄像头设备。

### 2. 运行监测系统

#### 使用内置摄像头
```bash
python main.py
```

#### 使用CPU（如果没有GPU）
```bash
python main.py --device cpu
```

#### 指定安全距离（例如2米）
```bash
python main.py --min-distance 2.0
```

#### 使用视频文件
```bash
python main.py --source video --video your_video.mp4
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 视频源: camera/video | camera |
| `--video` | 视频文件路径 | None |
| `--conf` | 检测置信度阈值 | 0.5 |
| `--min-distance` | 最小安全距离（米） | 1.5 |
| `--device` | 运行设备: cpu/cuda | cpu |
| `--camera-id` | 摄像头ID | 0 |

## 操作说明

### 启动后
1. 系统会自动开始检测和跟踪
2. 显示实时画面和统计信息
3. 红色线条表示安全距离违规
4. 每个人员显示唯一ID

### 控制
- 按 `q` 键退出程序

## 效果示意

```
正常状态：
┌─────────┐     ┌─────────┐
│ Person1 │     │ Person2 │  [绿色框，无连线]
└─────────┘     └─────────┘
   ID: 1           ID: 2

违规状态：
┌─────────┐ ──── ┌─────────┐
│ Person1 │  1.2m│ Person2 │  [红色连线标注距离]
└─────────┘ ──── └─────────┘
   ID: 1           ID: 2
```

## 故障排除

### 问题1: 摄像头无法打开
```bash
# 尝试不同的摄像头ID
python main.py --camera-id 1
```

### 问题2: 模型加载失败
确保 `yolov5s.pt` 文件在项目目录下。如果没有：
```bash
# 程序会自动从网络下载模型
python main.py
```

### 问题3: 检测不到人员
- 检查光照条件
- 尝试降低置信度阈值：`--conf 0.3`
- 确保人员完整出现在画面中

### 问题4: 距离测量不准确
当前使用默认相机标定参数。对于更准确的测量：
- 放置已知尺寸的参考物体
- 调整 `camera_calibrate.py` 中的参数
- 使用俯视角度可获得更准确结果

## 性能优化

### CPU优化
```bash
# 使用较小模型（需要下载YOLOv5n）
python main.py --device cpu
```

### GPU加速
```bash
python main.py --device cuda
```

### 降低精度提升速度
```bash
# 降低置信度阈值
python main.py --conf 0.3 --device cpu
```

## 使用示例场景

### 工厂车间
```bash
python main.py --min-distance 1.8 --device cuda
```

### 建筑工地
```bash
python main.py --min-distance 2.0 --conf 0.4
```

### 会议室
```bash
python main.py --min-distance 1.0 --device cpu
```

## 下一步

1. **校准相机** - 进行实际场景标定
2. **录制测试** - 使用视频文件验证
3. **调整参数** - 根据场景微调阈值
4. **ARM部署** - 转换为ONNX或使用TorchScript

## 技术支持

遇到问题？检查：
- Python版本 >= 3.7
- 所有依赖包已正确安装
- 摄像头驱动正常
- 模型文件完整

