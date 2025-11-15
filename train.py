from ultralytics import YOLO

# Choose a base model: yolov8n.pt (small) is good for CPU or low-GPU
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', 'yolov8m.pt' depending on GPU

# Train - change epochs/batch/imgsz according to your machine
model.train(
    data='data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='yash_experiment'
)
