"""
YOLOv5 detection wrapper for better model loading
"""

import torch
import numpy as np
import cv2
from pathlib import Path


class YOLOv5Detector:
    """
    YOLOv5s detector wrapper
    """
    def __init__(self, model_path, device='cpu', img_size=640):
        """
        Initialize YOLOv5 detector

        Args:
            model_path: Path to .pt model file
            device: 'cpu' or 'cuda'
            img_size: Input image size
        """
        self.model_path = model_path
        self.device = device
        self.img_size = img_size

        # Load model
        self._load_model()

    def _load_model(self):
        """Load YOLOv5 model"""
        import torch

        # Try method 1: Load local model file
        try:
            # Get torch.load function
            load_func = torch.load

            # Check if weights_only parameter exists (PyTorch 2.6+)
            import inspect
            sig = inspect.signature(load_func)
            if 'weights_only' in sig.parameters:
                # Use weights_only=False for PyTorch 2.6+
                checkpoint = load_func(self.model_path, map_location=self.device, weights_only=False)
            else:
                # Older PyTorch doesn't have this parameter
                checkpoint = load_func(self.model_path, map_location=self.device)

            # YOLOv5 model is typically stored as a full model object
            if isinstance(checkpoint, dict):
                # If it's a dict, it might contain 'model' key
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                else:
                    # It might be a state dict, try to load it directly
                    raise ValueError("Model checkpoint format not recognized")
            else:
                # It's the model object directly
                self.model = checkpoint

            self.model = self.model.to(self.device)
            self.model.eval()

            # Make sure model is in eval mode
            if hasattr(self.model, 'model'):
                # YOLOv5 models have nested structure
                self.model.model.eval()
            if hasattr(self.model, 'module'):
                # If wrapped in DataParallel or similar
                self.model.module.eval()

            print("Successfully loaded model from local file")
            return

        except Exception as e:
            print(f"Error loading local model: {e}")

            # Try method 2: Load from torch.hub
            try:
                import torch.hub
                print("Trying torch.hub...")
                # Use default yolo repo from ultralytics
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                                            pretrained=True, trust_repo=True, _verbose=False)
                self.model = self.model.to(self.device)
                self.model.eval()
                print("Successfully loaded YOLOv5s from torch.hub")
                return
            except Exception as e2:
                print(f"Error loading from torch.hub: {e2}")
                import traceback
                traceback.print_exc()

        # If all methods failed
        print("\nTroubleshooting:")
        print("1. Ensure yolov5s.pt file exists in current directory")
        print("2. Install missing packages: pip install requests pandas")
        print("3. For network issues, download model manually from:")
        print("   https://github.com/ultralytics/yolov5/releases")
        raise RuntimeError("Unable to load YOLOv5 model. Please check your setup.")

    def detect(self, img, conf_threshold=0.5, iou_threshold=0.45):
        """
        Detect objects in image

        Args:
            img: BGR image (numpy array)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detections as [tlwh, score, class_id]
        """
        # Get original dimensions
        h, w = img.shape[:2]

        # Run inference - YOLOv5 model expects images in RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # YOLOv5 models typically have a forward method that returns detections
            # The model might have an __call__ method that returns inference results
            if hasattr(self.model, '__call__'):
                results = self.model(img_rgb, size=640)
            else:
                # Fallback: try direct inference
                img_tensor = self._preprocess(img_rgb)
                results = self.model(img_tensor)

        # Parse results
        detections = []

        # YOLOv5 might return different formats
        # Handle pandas DataFrame or tensor
        if hasattr(results, 'pandas'):
            # This is a YOLOv5 Results object
            df = results.pandas().xyxy[0]  # Get pandas DataFrame with boxes

            for _, row in df.iterrows():
                if float(row['confidence']) < conf_threshold:
                    continue

                if int(row['class']) != 0:  # Only persons
                    continue

                # Get coordinates
                x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                conf = row['confidence']

                # Ensure coordinates are within bounds
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Convert to tlwh format
                tlwh = [x1, y1, x2 - x1, y2 - y1]

                detections.append([tlwh, float(conf), 0])

        else:
            # Fallback: try to parse as tensor
            if isinstance(results, (list, tuple)):
                results = results[0]

            if isinstance(results, torch.Tensor):
                results = results.cpu().numpy()
                if len(results.shape) == 3:
                    results = results[0]

                for det in results:
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, cls = det[:6]

                        if conf < conf_threshold:
                            continue

                        if int(cls) != 0:
                            continue

                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        tlwh = [x1, y1, x2 - x1, y2 - y1]
                        detections.append([tlwh, float(conf), 0])

        return detections

    def _preprocess(self, img_rgb):
        """
        Preprocess image for YOLOv5

        Args:
            img_rgb: RGB image

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Resize
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

        # Normalize
        img_float = img_resized.astype(np.float32) / 255.0

        # Convert to tensor [C, H, W]
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)


if __name__ == '__main__':
    # Test detector
    detector = YOLOv5Detector('yolov5s.pt', device='cpu')
    print("Detector loaded successfully!")

