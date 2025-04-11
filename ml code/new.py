from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from collections import deque

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Configuration parameters - adjust these for your specific case
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for detections
IOU_THRESHOLD = 0.45        # NMS IOU threshold
RESIZE_WIDTH = 1280         # Higher resolution input for better detection
RESIZE_HEIGHT = 720
SKIP_FRAMES = 2             # Process every nth frame
USE_TRACKER = True          # Enable tracking for smoother results
TRACK_HISTORY = 20          # Number of frames to keep track history

# Set up video input
video_path = '4.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Optional: Get video info for output video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Colors for visualization
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

# Track history for each tracked object
track_history = {}

# Function to apply overlay on segmentation mask
def apply_mask_overlay(image, mask, color, alpha=0.3):
    # Create color overlay for the mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color
    
    # Blend the image and the colored mask
    masked_image = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    return masked_image

count = 0
processing_times = deque(maxlen=50)  # To calculate average FPS

# Main processing loop
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    count += 1
    if count % (SKIP_FRAMES + 1) != 0:
        continue  # Skip frames for speed
    
    # Resize frame for consistent processing
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    h, w, _ = frame.shape
    
    # Run inference with higher confidence
    if USE_TRACKER:
        results = model.track(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, persist=True, verbose=False)
    else:
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    
    # Make a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Process each detected object
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        # Process tracking information if available
        if hasattr(boxes, 'id') and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)
        
        # Process each detection
        for i, (mask, box, track_id) in enumerate(zip(masks, boxes, track_ids)):
            # Get class details
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = class_names[class_id]
            color = COLORS[class_id].tolist()
            
            # Resize mask to frame size
            mask = cv2.resize(mask, (w, h))
            mask = (mask > 0.5).astype(np.uint8)  # Threshold mask
            
            # Find contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Apply mask overlay
                vis_frame = apply_mask_overlay(vis_frame, mask, color, alpha=0.3)
                
                # Draw contours
                cv2.drawContours(vis_frame, contours, -1, color, 2)
                
                # Get bounding box from contours
                all_points = np.concatenate(contours)
                x, y, w_box, h_box = cv2.boundingRect(all_points)
                
                # Add tracking history
                if USE_TRACKER and track_id is not None:
                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=TRACK_HISTORY)
                    
                    centroid = (int(x + w_box/2), int(y + h_box/2))
                    track_history[track_id].append(centroid)
                    
                    # Draw tracking lines
                    for j in range(1, len(track_history[track_id])):
                        if track_history[track_id][j-1] is None or track_history[track_id][j] is None:
                            continue
                        cv2.line(vis_frame, 
                                track_history[track_id][j-1],
                                track_history[track_id][j], 
                                color, 2)
                
                # Add text with confidence score
                label = f"{class_name} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
                cv2.putText(vis_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Calculate and display FPS
    processing_time = time.time() - start_time
    processing_times.append(processing_time)
    avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
    cv2.putText(vis_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the processed frame
    cv2.imshow('YOLOv8 Segmentation', vis_frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()