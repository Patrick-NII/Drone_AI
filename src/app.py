from ultralytics import YOLO

# Load the model yolov Extra large model with high accuracy
model = YOLO("yolov11x.pt")

# Track the video source 0, save the output video in the project folder, name the output video as test_yolo, and show the output video
source_0_ = 0
test = model.track(source=source_0_, save=True, project="./output", name="test_yolo", show=True)
