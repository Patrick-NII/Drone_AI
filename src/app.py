from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Track the video source 0, save the output video in the project folder, name the output video as test_yolo, and show the output video
test = model.track(source=0, save=True, project="./output", name="test_yolo", show=True)