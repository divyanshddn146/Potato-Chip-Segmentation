import cv2
import os
import numpy as np
from ultralytics import YOLO


model = YOLO("runs/segment/train/weights/best.pt")

input_folder = "C:/Users/divya/Desktop/Chip_CNN/Pepsico/Test/Defective"
output_folder = "C:/Users/divya/Desktop/Chip_CNN/predictions"
os.makedirs(output_folder, exist_ok=True)

CLASS_COLORS = {"damaged": (0, 0, 255), "semi-damaged": (0, 165, 255)}  
ALPHA = 0.3

for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_folder, file_name)
    print("Reading:", img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue

    results = model.predict(img, verbose=False)
    for r in results:
        if r.masks is not None:
            for mask, cls_id, box in zip(r.masks.data.cpu().numpy(),
                                         r.boxes.cls.cpu().numpy(),
                                         r.boxes.xyxy.cpu().numpy()):
                cls_name = model.names[int(cls_id)]
                if cls_name == "chip":
                    continue

                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_resized = (mask_resized > 0.5).astype(np.uint8)

                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[:] = color
                img = np.where(mask_resized[..., None] > 0,
                               (img * (1 - ALPHA) + colored_mask * ALPHA).astype(np.uint8),
                               img)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_folder, file_name), img)
