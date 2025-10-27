"""
Configuration file for Safety Distance Monitor
"""

class Config:
    """System configuration"""

    # Model settings
    MODEL_PATH = 'yolov5s.pt'
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    IMG_SIZE = 640

    # Safety distance settings
    MIN_DISTANCE_METERS = 1.5  # Minimum safe distance in meters

    # Tracker settings
    TRACK_THRESH = 0.5
    HIGH_THRESH = 0.6
    MATCH_THRESH = 0.8
    TRACK_BUFFER = 30
    FRAME_RATE = 30
    MIN_BOX_AREA = 10

    # Camera calibration settings
    FOCAL_LENGTH = 800  # pixels (adjust based on your camera)
    KNOWN_REAL_WIDTH = 0.4  # meters (average person width)
    KNOWN_REAL_HEIGHT = 1.7  # meters (average person height)

    # Visualization settings
    BOX_THICKNESS = 2
    LINE_THICKNESS = 2
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    # Colors (BGR format)
    COLOR_NORMAL = (0, 255, 0)  # Green
    COLOR_VIOLATION = (0, 0, 255)  # Red
    COLOR_WARNING = (0, 165, 255)  # Orange
    COLOR_TEXT = (255, 255, 255)  # White

    # Alert settings
    ALERT_COOLDOWN = 2  # seconds
    MAX_DISTANCE_ESTIMATE = 10.0  # meters
    MIN_DISTANCE_ESTIMATE = 0.5  # meters

    # Video settings
    CAMERA_ID = 0
    OUTPUT_WIDTH = 1280
    OUTPUT_HEIGHT = 720

    # Performance settings
    FPS_DISPLAY_INTERVAL = 1  # seconds
    MAX_TRACKS_TO_DISPLAY = 100

    # System settings
    DEVICE = 'cpu'  # 'cpu' or 'cuda'
    VERBOSE = True

