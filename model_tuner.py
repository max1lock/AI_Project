import os
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11n.pt")

# Tune hyperparameters
model.tune(
    data=os.path.abspath("./config.yaml"),
    epochs=30,
    iterations=300,
    optimizer="AdamW",
    imgsz=640,
)
