from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolo12x") 

# Perform tracking with the model
results = model.track("https://www.youtube.com/watch?v=a6MsxYJWE1c", show=True, tracker="bytetrack.yaml")  # with ByteTrack