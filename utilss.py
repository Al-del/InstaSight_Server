from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
device = "cuda" if torch.cuda.is_available() else "cpu"

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
def process_image(img_rgb, yolo_model, feature_extractor, device, depth_model):
    # Load image if path
    img_pil = Image.fromarray(img_rgb)

    # Get bounding boxes from YOLOv5
    results = yolo_model(img_pil)
    bboxes = results.xyxy[0].cpu().numpy()  # (x1,y1,x2,y2,conf,cls)

    # Depth estimation with MiDaS
    inputs = feature_extractor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img_pil.size[::-1],  # (height, width)
        mode="bicubic",
        align_corners=False,
    )
    depth_map = prediction.squeeze().cpu().numpy()
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

    return img_rgb, bboxes, depth_norm


img = cv2.imread("./received_frame_10.jpg")
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s").to(device)
feature_extractor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
old_img, bbox, latest_frame = process_image(img, yolo_model, feature_extractor, device, depth_model)
plt.imshow(latest_frame, cmap="inferno")
plt.colorbar()
plt.title("Estimated Depth Map")
plt.axis("off")
plt.show()
cv2.imwrite("depth_map_debug.jpg", (latest_frame * 255).astype(np.uint8))