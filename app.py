from ultralytics import YOLO


model= YOLO("yolov11s.pt")

test = model.track(source=0, save=True, project="./output", name="test_yolo", show=True)
