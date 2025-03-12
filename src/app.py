import sys
sys.path.append('/home/kimuntu/Documents/Programming2025/Drone_AI/my-env/lib/python3.12/site-packages')

from ultralytics import YOLO


model = YOLO("yolov8n.pt")
test = model.track(source=0, save=True, project="./output", name="test_yolo", show=True)