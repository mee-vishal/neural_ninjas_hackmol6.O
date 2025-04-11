from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from collections import deque, Counter

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Configuration parameters - adjust these for your specific case
CONFIDENCE_THRESHOLD = 0.55  # Slightly higher confidence threshold for more reliable counting
IOU_THRESHOLD = 0.45         # NMS IOU threshold
RESIZE_WIDTH = 1280          # Higher resolution input for better detection
RESIZE_HEIGHT = 720
SKIP_FRAMES = 2              # Process every nth frame
MIN_POTHOLE_AREA = 100       # Minimum area of pothole to count (in pixels)
MIN_DETECTION_FRAMES = 3     # Minimum number of frames a pothole must be detected to be counted
USE_TRACKER = True           # Enable tracking for more stable detections
TRACK_HISTORY = 20           # Number of frames to keep track history

# Set up video input
video_path = 'p.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Colors for visualization
POTHOLE_COLOR = (0, 0, 255)  # Red in BGR
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
for i in range(len(class_names)):
    if 'pothole' in class_names[i].lower():
        COLORS[i] = np.array(POTHOLE_COLOR)

# Track history and pothole tracking
track_history = {}
pothole_detections = {}  # Track ID -> count of detections
confirmed_potholes = set()  # Set of confirmed pothole track IDs
pothole_areas = {}  # Track ID -> list of recent area measurements
total_unique_potholes = 0
current_frame_pothole_count = 0

# Global counters for display
total_frame_count = 0
processed_frame_count = 0

# Dictionary to store pothole locations (for duplicate detection)
pothole_locations = {}

# Function to apply overlay on segmentation mask
def apply_mask_overlay(image, mask, color, alpha=0.5):
    # Create color overlay for the mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color
    
    # Blend the image and the colored mask
    masked_image = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    return masked_image

# Function to determine if a pothole is a duplicate (too close to existing one)
def is_duplicate_pothole(centroid, existing_locations, min_distance=50):
    x, y = centroid
    for loc_id, loc in existing_locations.items():
        loc_x, loc_y = loc
        distance = np.sqrt((x - loc_x)**2 + (y - loc_y)**2)
        if distance < min_distance:
            return True, loc_id
    return False, None

# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    # box format: (x, y, w, h)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # No intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

count = 0
processing_times = deque(maxlen=50)  # To calculate average FPS

# Initialize video writer for saving output (optional)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('pothole_detection_output.mp4', fourcc, fps, (RESIZE_WIDTH, RESIZE_HEIGHT))

# Main processing loop
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    count += 1
    total_frame_count += 1
    if count % (SKIP_FRAMES + 1) != 0:
        continue  # Skip frames for speed
    
    processed_frame_count += 1
    
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
    
    # Reset current frame pothole count
    current_frame_pothole_count = 0
    current_frame_potholes = []
    
    # Process each detected object
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        # Process tracking information if available
        if hasattr(boxes, 'id') and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)
        
        # Track active potholes in this frame
        active_pothole_ids = set()
        
        # Process each detection
        for i, (mask, box, track_id) in enumerate(zip(masks, boxes, track_ids)):
            # Get class details
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = class_names[class_id]
            color = COLORS[class_id].tolist()
            
            # Skip if not a pothole (assuming we have multiple classes)
            if 'pothole' not in class_name.lower():
                continue
                
            # Resize mask to frame size
            mask = cv2.resize(mask, (w, h))
            mask = (mask > 0.5).astype(np.uint8)  # Threshold mask
            
            # Find contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Calculate area and filter out small detections
                all_points = np.concatenate(contours)
                area = cv2.contourArea(all_points)
                x, y, w_box, h_box = cv2.boundingRect(all_points)
                
                # Skip small detections that are likely noise
                if area < MIN_POTHOLE_AREA:
                    continue
                
                # Apply mask overlay
                vis_frame = apply_mask_overlay(vis_frame, mask, color, alpha=0.5)
                
                # Draw contours with thicker lines for better visibility
                cv2.drawContours(vis_frame, contours, -1, color, 3)
                
                # Calculate pothole centroid
                centroid = (int(x + w_box/2), int(y + h_box/2))
                
                # Tracking logic
                if track_id is not None:
                    # Track pothole history
                    if track_id not in pothole_detections:
                        pothole_detections[track_id] = 1
                        pothole_areas[track_id] = [area]
                    else:
                        pothole_detections[track_id] += 1
                        pothole_areas[track_id].append(area)
                        # Keep only the last 5 area measurements
                        if len(pothole_areas[track_id]) > 5:
                            pothole_areas[track_id] = pothole_areas[track_id][-5:]
                    
                    # Check if this is a confirmed pothole
                    if (pothole_detections[track_id] >= MIN_DETECTION_FRAMES and 
                        track_id not in confirmed_potholes):
                        # Check if this is a duplicate of an existing pothole
                        is_duplicate = False
                        for existing_id in confirmed_potholes:
                            if existing_id in pothole_locations:
                                dist = np.sqrt((centroid[0] - pothole_locations[existing_id][0])**2 + 
                                               (centroid[1] - pothole_locations[existing_id][1])**2)
                                if dist < 50:  # 50 pixel threshold for duplicate detection
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            confirmed_potholes.add(track_id)
                            pothole_locations[track_id] = centroid
                            total_unique_potholes += 1
                    
                    # Mark as active in this frame
                    active_pothole_ids.add(track_id)
                
                # Consider as a current frame pothole if it's confirmed or high confidence
                if (track_id in confirmed_potholes) or (conf > 0.7):
                    current_frame_pothole_count += 1
                    current_frame_potholes.append((centroid, area, conf))
                
                # Add text with confidence score and area information
                avg_area = sum(pothole_areas.get(track_id, [area])) / len(pothole_areas.get(track_id, [area]))
                label = f"Pothole {conf:.2f} ({int(avg_area)}px²)"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
                cv2.putText(vis_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw a circle at centroid for visualization
                cv2.circle(vis_frame, centroid, 4, (0, 255, 255), -1)
        
        # Clean up old tracks that are no longer active
        inactive_tracks = set(pothole_detections.keys()) - active_pothole_ids
        for track_id in inactive_tracks:
            # Only remove if it's been absent for several frames
            pothole_detections[track_id] -= 0.2  # Gradual decay of confidence
            if pothole_detections[track_id] <= 0:
                if track_id in pothole_detections:
                    del pothole_detections[track_id]
                if track_id in pothole_areas:
                    del pothole_areas[track_id]
    
    # Draw counters and stats on the frame
    info_bg = np.zeros((120, RESIZE_WIDTH, 3), dtype=np.uint8)
    cv2.putText(info_bg, f"Total Unique Potholes: {total_unique_potholes}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(info_bg, f"Current Frame Potholes: {current_frame_pothole_count}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Calculate and display FPS
    processing_time = time.time() - start_time
    processing_times.append(processing_time)
    avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
    cv2.putText(info_bg, f"FPS: {avg_fps:.2f} (Processing {100*(processed_frame_count/total_frame_count):.1f}% of frames)", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Combine info panel with main frame
    display_frame = np.vstack([info_bg, vis_frame])
    
    # Write frame to output video if desired
    # out.write(vis_frame)
    
    # Display the processed frame
    cv2.imshow('Accurate Pothole Detection & Counting', display_frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
# out.release()  # Uncomment if saving output video
cv2.destroyAllWindows()

print(f"Total unique potholes detected: {total_unique_potholes}")