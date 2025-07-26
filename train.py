from ultralytics import YOLO

# Load the last trained model (best.pt or last.pt)
model = YOLO("runs/segment/train/weights/last.pt")  # Change to best.pt if needed

# Resume training with fine-tuning
model.train(
    data="C:/Users/divya/Desktop/Chip_CNN/data.yaml",  # Path to your data.yaml
    epochs=20,                 # Fine-tune for 20 more epochs
    imgsz=640,                 # Image size
    batch=4,                   # Batch size
    workers=2,                 # Number of workers
    device="cpu",              # Use "cuda" if GPU is available
    resume=True,               # Continue training from the previous checkpoint
    lr0=0.001,                 # Lower learning rate for fine-tuning
    hsv_h=0.015,               # Hue augmentation
    hsv_s=0.5,                 # Saturation augmentation
    hsv_v=0.4,                 # Brightness augmentation
    fliplr=0.5,                # Horizontal flip
    flipud=0.0,                # Vertical flip
    scale=0.1,                 # Small scaling
    mosaic=0.0,                # Disable mosaic
    mixup=0.0                  # Disable mixup
)
