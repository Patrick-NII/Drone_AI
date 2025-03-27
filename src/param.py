# param.py : Configuration parameters for YOLO tracking

# ==========================
#  YOLO PARAMETERS
# ==========================

# YOLO model to use (optimized for tracking)
YOLO_MODEL = "yolov8n.pt"  # Using nano model for better performance on macOS

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for better detection

# Intersection over Union threshold
IOU_THRESHOLD = 0.5  # Higher IOU for better tracking stability

# Tracker configuration
TRACKER = "bytetrack.yaml"  # Best tracker for general purpose tracking

# Device configuration (optimized for macOS)
USE_CUDA = False  # Disabled for macOS compatibility
USE_MPS = True    # Enable Metal Performance Shaders for macOS

# ==========================
#  SOURCES & OUTPUT
# ==========================

# Default source (0 = Webcam)
DEFAULT_SOURCE = 1

# Output directory for processed media
OUTPUT_DIR = "output"

# ==========================
#  DISPLAY PARAMETERS
# ==========================

# Box colors for different object types (BGR format for OpenCV)
BOX_COLORS = {
    "person": (0, 255, 0),         # Green
    "car": (255, 0, 0),            # Blue
    "truck": (0, 0, 255),          # Red
    "motorcycle": (255, 165, 0),   # Orange
    "bicycle": (128, 0, 128),      # Purple
    "dog": (0, 255, 255),          # Cyan
    "cat": (255, 192, 203),        # Pink
    "bird": (75, 0, 130),          # Indigo
    "boat": (255, 255, 0),         # Yellow
    "traffic light": (0, 128, 128) # Dark Blue
}

# Default box color for unlisted objects
DEFAULT_BOX_COLOR = (200, 200, 200)  # Light Gray

# Text parameters
TEXT_COLOR = (255, 255, 255)  # White text for better visibility
TEXT_FONT = 0.6
TEXT_THICKNESS = 2

# ==========================
#  OBJECT LABELS
# ==========================

# Dictionary for object labels (English)
OBJECT_LABELS = {
    "person": "Person",
    "car": "Car",
    "truck": "Truck",
    "motorcycle": "Motorcycle",
    "bicycle": "Bicycle",
    "bus": "Bus",
    "train": "Train",
    "boat": "Boat",
    "airplane": "Airplane",
    "traffic light": "Traffic Light",
    "stop sign": "Stop Sign",
    "parking meter": "Parking Meter",
    "bench": "Bench",
    "dog": "Dog",
    "cat": "Cat",
    "horse": "Horse",
    "bird": "Bird",
    "cow": "Cow",
    "sheep": "Sheep",
    "elephant": "Elephant",
    "zebra": "Zebra",
    "giraffe": "Giraffe",
    "backpack": "Backpack",
    "umbrella": "Umbrella",
    "handbag": "Handbag",
    "tie": "Tie",
    "suitcase": "Suitcase",
    "frisbee": "Frisbee",
    "skis": "Skis",
    "snowboard": "Snowboard",
    "sports ball": "Sports Ball",
    "kite": "Kite",
    "baseball bat": "Baseball Bat",
    "baseball glove": "Baseball Glove",
    "skateboard": "Skateboard",
    "surfboard": "Surfboard",
    "tennis racket": "Tennis Racket",
    "bottle": "Bottle",
    "wine glass": "Wine Glass",
    "cup": "Cup",
    "fork": "Fork",
    "knife": "Knife",
    "spoon": "Spoon",
    "bowl": "Bowl",
    "banana": "Banana",
    "apple": "Apple",
    "sandwich": "Sandwich",
    "orange": "Orange",
    "broccoli": "Broccoli",
    "carrot": "Carrot",
    "hot dog": "Hot Dog",
    "pizza": "Pizza",
    "donut": "Donut",
    "cake": "Cake",
    "chair": "Chair",
    "couch": "Couch",
    "potted plant": "Potted Plant",
    "bed": "Bed",
    "dining table": "Dining Table",
    "toilet": "Toilet",
    "tv": "TV",
    "laptop": "Laptop",
    "mouse": "Mouse",
    "remote": "Remote",
    "keyboard": "Keyboard",
    "cell phone": "Cell Phone",
    "microwave": "Microwave",
    "oven": "Oven",
    "toaster": "Toaster",
    "sink": "Sink",
    "refrigerator": "Refrigerator",
    "book": "Book",
    "clock": "Clock",
    "vase": "Vase",
    "scissors": "Scissors",
    "teddy bear": "Teddy Bear",
    "hair drier": "Hair Drier",
    "toothbrush": "Toothbrush"
}

# ==========================
#  ADDITIONAL OPTIONS
# ==========================

# Enable real-time display
DISPLAY_REAL_TIME = True

# Save processed video
SAVE_PROCESSED_VIDEO = True

# Save detected object images
SAVE_OBJECT_IMAGES = False

# Resize output images
RESIZE_OUTPUT = True
OUTPUT_IMAGE_SIZE = (640, 480)  # Optimized size for better performance

# Frame processing parameters
MAX_FRAME_WIDTH = 1280
MAX_FRAME_HEIGHT = 720
PROCESS_EVERY_N_FRAMES = 2  # Process every nth frame for better performance
