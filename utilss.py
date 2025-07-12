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

    # Depth estimation
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

    close_objects = []  # List to store close object info
    
    # Check object proximity
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Extract depth values within the bounding box
        box_depth = depth_norm[y1:y2, x1:x2]
        
        if box_depth.size == 0:
            continue
            
        # Calculate average depth in the bounding box
        avg_depth = np.mean(box_depth)
        
        # Define threshold (adjust this value based on your needs)
        CLOSE_THRESHOLD = 0.7
        
        if avg_depth > CLOSE_THRESHOLD:
            print(f"WARNING: Object at ({x1},{y1})-({x2},{y2}) is too close! (depth: {avg_depth:.2f})")
            
            # Add to close objects list
            close_objects.append({
                'bbox': [x1, y1, x2, y2],
                'depth': float(avg_depth)
            })
            
            # Draw warning on the image
            cv2.putText(img_rgb, "TOO CLOSE!", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img_rgb, bboxes, depth_norm, close_objects