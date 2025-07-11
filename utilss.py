from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def generate_bbox(model, img):
    # Convert to RGB only once
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Run inference
    results = model(pil_img)

    # Use tensor output directly to skip Pandas overhead
    boxes = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

    # Draw boxes on the original image (BGR)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img
"""
img = cv2.imread("./received_frame_10.jpg")
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model = model
ok = generate_bbox(model, img)
print(ok.shape)
"""