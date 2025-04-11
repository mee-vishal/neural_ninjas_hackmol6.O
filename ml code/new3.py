from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from collections import deque, Counter
import datetime
import matplotlib.pyplot as plt
import csv

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.55  # Confidence threshold for detections
IOU_THRESHOLD = 0.45         # NMS IOU threshold
RESIZE_WIDTH = 1280          # Input resolution
RESIZE_HEIGHT = 720
SKIP_FRAMES = 2              # Process every nth frame
MIN_POTHOLE_AREA = 100       # Minimum area to count
MIN_DETECTION_FRAMES = 3     # Minimum detection persistence
USE_TRACKER = True           # Enable tracking
TRACK_HISTORY = 20           # Tracking history length

# Road quality assessment parameters
ROAD_SEGMENT_LENGTH = 50     # Frames per road segment for assessment
ROAD_QUALITY_LEVELS = {
    0: {"label": "Excellent", "color": (0, 255, 0)},     # Green
    1: {"label": "Good", "color": (0, 255, 255)},        # Yellow
    2: {"label": "Fair", "color": (0, 165, 255)},        # Orange
    3: {"label": "Poor", "color": (0, 0, 255)},          # Red
    4: {"label": "Very Poor", "color": (0, 0, 128)}      # Dark Red
}

# Set up video input
video_path = 'https://res.cloudinary.com/dx3chjisf/video/upload/v1744305628/n33l3xe0hji3tdxgxsc0.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Colors for visualization
POTHOLE_COLOR = (0, 0, 255)  # Red in BGR
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
for i in range(len(class_names)):
    if 'pothole' in class_names[i].lower():
        COLORS[i] = np.array(POTHOLE_COLOR)

# Tracking variables
track_history = {}
pothole_detections = {}  # Track ID -> count of detections
confirmed_potholes = set()  # Set of confirmed pothole track IDs
pothole_areas = {}  # Track ID -> list of recent area measurements
total_unique_potholes = 0
current_frame_pothole_count = 0

# Road quality assessment variables
segment_id = 0
segment_potholes = []  # List to store pothole counts per segment
segment_pothole_density = []  # Potholes per frame in each segment
segment_pothole_sizes = []  # Average pothole size per segment
segment_quality_scores = []  # Quality score for each segment
current_segment_potholes = 0
current_segment_frames = 0
current_segment_pothole_sizes = []
road_quality_history = deque(maxlen=10)  # Store recent quality levels for smoothing

# Create directory for saving results
output_dir = "road_quality_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up CSV for logging results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(output_dir, f"road_quality_{timestamp}.csv")
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Segment', 'Frames', 'Potholes', 'Avg Size (px²)', 'Density', 'Quality Score', 'Quality Label'])

# Function to calculate road quality score
def calculate_road_quality(pothole_count, frames_count, avg_size):
    if frames_count == 0:
        return 0
    
    # Calculate density (potholes per frame)
    density = pothole_count / frames_count
    
    # Normalize size factor (larger potholes = worse road)
    size_factor = min(1.0, avg_size / 2000) if avg_size > 0 else 0
    
    # Combined score (0 = excellent, 4 = very poor)
    if density == 0:
        base_score = 0  # Excellent - no potholes
    elif density < 0.05:
        base_score = 1  # Good - very few potholes
    elif density < 0.1:
        base_score = 2  # Fair - some potholes
    elif density < 0.2:
        base_score = 3  # Poor - many potholes
    else:
        base_score = 4  # Very poor - extensive pothole damage
    
    # Adjust score based on pothole size
    adjusted_score = min(4, base_score + (size_factor * 1.5))
    
    return adjusted_score

# Function to apply overlay on segmentation mask
def apply_mask_overlay(image, mask, color, alpha=0.5):
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color
    masked_image = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    return masked_image

# For visualization - create a quality bar for display
def create_quality_bar(quality_score, width=200, height=30):
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    # Create gradient colors for quality levels
    for i in range(width):
        level = min(4, max(0, int(i / width * 5)))
        color = ROAD_QUALITY_LEVELS[level]["color"]
        bar[:, i] = color
    
    # Draw marker for current quality
    marker_pos = int(min(quality_score, 4) / 4 * (width - 10))
    cv2.rectangle(bar, (marker_pos, 0), (marker_pos + 10, height), (255, 255, 255), -1)
    
    return bar

# Main variable tracking
count = 0
processing_times = deque(maxlen=50)  # For FPS calculation
total_frame_count = 0
processed_frame_count = 0

# Initialize video writer for saving output
output_video_path = os.path.join(output_dir, f"road_quality_output_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps/3, (RESIZE_WIDTH, RESIZE_HEIGHT + 250))

# Prepare plot areas
plot_background = np.ones((250, RESIZE_WIDTH, 3), dtype=np.uint8) * 255
plot_width = RESIZE_WIDTH - 40
plot_height = 150
plot_x = 20
plot_y = 20

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
    current_segment_frames += 1
    
    # Resize frame for consistent processing
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    h, w, _ = frame.shape
    
    # Run inference
    if USE_TRACKER:
        results = model.track(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, persist=True, verbose=False)
    else:
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    
    # Make a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Reset current frame pothole count
    current_frame_pothole_count = 0
    current_frame_pothole_areas = []
    
    # Process detection results
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
            
            # Skip if not a pothole (in case of multiple classes)
            if 'pothole' not in class_name.lower():
                continue
                
            # Resize and process mask
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
                
                # Draw contours
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
                        confirmed_potholes.add(track_id)
                        total_unique_potholes += 1
                        current_segment_potholes += 1
                    
                    # Mark as active in this frame
                    active_pothole_ids.add(track_id)
                
                # Consider as a current frame pothole if it's confirmed or high confidence
                if (track_id in confirmed_potholes) or (conf > 0.7):
                    current_frame_pothole_count += 1
                    current_frame_pothole_areas.append(area)
                    
                    # Add to segment data for road quality assessment
                    current_segment_pothole_sizes.append(area)
                
                # Add text with confidence score and area information
                avg_area = sum(pothole_areas.get(track_id, [area])) / len(pothole_areas.get(track_id, [area]))
                label = f"Pothole {conf:.2f} ({int(avg_area)}px²)"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
                cv2.putText(vis_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw a circle at centroid
                cv2.circle(vis_frame, centroid, 4, (0, 255, 255), -1)
        
        # Clean up old tracks
        inactive_tracks = set(pothole_detections.keys()) - active_pothole_ids
        for track_id in inactive_tracks:
            pothole_detections[track_id] -= 0.2  # Gradual decay
            if pothole_detections[track_id] <= 0:
                if track_id in pothole_detections:
                    del pothole_detections[track_id]
                if track_id in pothole_areas:
                    del pothole_areas[track_id]
    
    # Check if we need to calculate road quality for this segment
    if current_segment_frames >= ROAD_SEGMENT_LENGTH or total_frame_count >= total_frames:
        # Calculate average pothole size in this segment
        avg_size = sum(current_segment_pothole_sizes) / len(current_segment_pothole_sizes) if current_segment_pothole_sizes else 0
        
        # Calculate road quality score
        quality_score = calculate_road_quality(current_segment_potholes, current_segment_frames, avg_size)
        
        # Calculate pothole density
        density = current_segment_potholes / current_segment_frames if current_segment_frames > 0 else 0
        
        # Save segment data
        segment_potholes.append(current_segment_potholes)
        segment_pothole_density.append(density)
        segment_pothole_sizes.append(avg_size)
        segment_quality_scores.append(quality_score)
        
        # Get quality label
        quality_level = min(4, int(round(quality_score)))
        quality_label = ROAD_QUALITY_LEVELS[quality_level]["label"]
        
        # Add to road quality history for smoothing
        road_quality_history.append(quality_score)
        
        # Log to CSV
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                segment_id, 
                current_segment_frames, 
                current_segment_potholes, 
                int(avg_size) if avg_size > 0 else 0,
                f"{density:.4f}",
                f"{quality_score:.2f}",
                quality_label
            ])
        
        # Reset segment counters
        segment_id += 1
        current_segment_potholes = 0
        current_segment_frames = 0
        current_segment_pothole_sizes = []
    
    # Calculate current road quality (smoothed)
    current_quality = sum(road_quality_history) / len(road_quality_history) if road_quality_history else 0
    quality_level = min(4, int(round(current_quality)))
    quality_label = ROAD_QUALITY_LEVELS[quality_level]["label"]
    quality_color = ROAD_QUALITY_LEVELS[quality_level]["color"]
    
    # Create info panel with road quality
    info_bg = np.zeros((120, RESIZE_WIDTH, 3), dtype=np.uint8)
    cv2.putText(info_bg, f"Road Quality: {quality_label}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, quality_color, 2)
    cv2.putText(info_bg, f"Total Potholes: {total_unique_potholes} | Current Frame: {current_frame_pothole_count}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw quality bar
    quality_bar = create_quality_bar(current_quality)
    info_bg[80:110, 10:210] = quality_bar
    
    # Calculate and display FPS
    processing_time = time.time() - start_time
    processing_times.append(processing_time)
    avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
    cv2.putText(info_bg, f"FPS: {avg_fps:.2f} | Progress: {100*(total_frame_count/total_frames):.1f}%", 
                (RESIZE_WIDTH - 400, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Create plot area for road quality history
    plot_bg = plot_background.copy()
    
    # If we have segments, plot them
    if segment_quality_scores:
        # Define plot area
        cv2.rectangle(plot_bg, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), (0, 0, 0), 1)
        
        # Plot labels
        cv2.putText(plot_bg, "Road Quality History", (plot_x, plot_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(plot_bg, "Excellent", (plot_x - 15, plot_y + plot_height + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.putText(plot_bg, "Very Poor", (plot_x - 15, plot_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        
        # Plot quality scores
        max_points = min(len(segment_quality_scores), 50)  # Show last 50 segments max
        start_idx = max(0, len(segment_quality_scores) - max_points)
        for i in range(start_idx, len(segment_quality_scores)):
            if i == start_idx:
                continue
                
            rel_i = i - start_idx
            x1 = plot_x + int((rel_i - 1) * plot_width / max_points)
            y1 = plot_y + int((4 - segment_quality_scores[i-1]) * plot_height / 4)
            x2 = plot_x + int(rel_i * plot_width / max_points)
            y2 = plot_y + int((4 - segment_quality_scores[i]) * plot_height / 4)
            
            # Get color based on quality level
            level = min(4, int(round(segment_quality_scores[i])))
            line_color = ROAD_QUALITY_LEVELS[level]["color"]
            
            # Draw line
            cv2.line(plot_bg, (x1, y1), (x2, y2), line_color, 2)
            
            # Mark point
            cv2.circle(plot_bg, (x2, y2), 3, line_color, -1)
    
    # Combine everything
    display_frame = np.vstack([info_bg, vis_frame, plot_bg])
    resized_display = cv2.resize(display_frame, (RESIZE_WIDTH, RESIZE_HEIGHT + 250))
    
    # Write frame to output video
    out.write(resized_display[:RESIZE_HEIGHT + 250])
    
    # Display the processed frame
    cv2.imshow('Road Quality Assessment', resized_display)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Generate final report and charts
if segment_quality_scores:
    # Create summary chart
    plt.figure(figsize=(12, 6))
    
    # Plot road quality scores
    plt.subplot(2, 1, 1)
    plt.plot(segment_quality_scores, 'r-')
    plt.fill_between(range(len(segment_quality_scores)), segment_quality_scores, alpha=0.3, color='red')
    plt.title('Road Quality Assessment')
    plt.ylabel('Quality Score (0=Excellent, 4=Very Poor)')
    plt.grid(True)
    
    # Plot pothole density
    plt.subplot(2, 1, 2)
    plt.bar(range(len(segment_pothole_density)), segment_pothole_density, color='blue', alpha=0.7)
    plt.title('Pothole Density by Segment')
    plt.xlabel('Road Segment')
    plt.ylabel('Potholes per Frame')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"road_quality_chart_{timestamp}.png"))
    
    # Calculate overall road quality
    avg_quality = sum(segment_quality_scores) / len(segment_quality_scores)
    overall_level = min(4, int(round(avg_quality)))
    overall_label = ROAD_QUALITY_LEVELS[overall_level]["label"]
    
    # Create summary report
    with open(os.path.join(output_dir, f"road_quality_summary_{timestamp}.txt"), 'w') as f:
        f.write("Road Quality Assessment Summary\n")
        f.write("=============================\n\n")
        f.write(f"Video Analyzed: {video_path}\n")
        f.write(f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Frames Analyzed: {processed_frame_count}\n")
        f.write(f"Total Road Segments: {len(segment_quality_scores)}\n")
        f.write(f"Total Potholes Detected: {total_unique_potholes}\n\n")
        f.write(f"Overall Road Quality: {overall_label} (Score: {avg_quality:.2f}/4.00)\n\n")
        f.write("Segment Breakdown:\n")
        f.write("=================\n")
        
        for i in range(len(segment_quality_scores)):
            quality_level = min(4, int(round(segment_quality_scores[i])))
            quality_label = ROAD_QUALITY_LEVELS[quality_level]["label"]
            f.write(f"Segment {i+1}: {quality_label} (Score: {segment_quality_scores[i]:.2f}, " +
                   f"Potholes: {segment_potholes[i]}, Density: {segment_pothole_density[i]:.4f})\n")

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Analysis complete. Results saved to {output_dir} directory.")
print(f"Total unique potholes detected: {total_unique_potholes}")
print(f"Video output saved to: {output_video_path}")