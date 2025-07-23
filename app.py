# app.py
import gradio as gr
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from moviepy.editor import ImageSequenceClip
import time
from collections import deque

print("Loading models...")
# Initialize models once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = YOLO('yolov8n.pt')
detector.to(device)
tracker = DeepSort(max_age=30)
print("Models loaded.")

# --- NEW: Dictionary to store object trajectories ---
trajectory_history = {}

def process_video(video_path, start_time, end_time, progress=gr.Progress()):
    # Reset history for each new video
    global trajectory_history
    trajectory_history.clear()
    
    start_time = int(start_time)
    end_time = int(end_time)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    start_frame = start_time * fps
    end_frame = end_time * fps
    total_processing_frames = end_frame - start_frame
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    output_frames = []
    progress(0, desc="Starting...")
    
    start_timestamp = time.time()
    
    for frame_count in range(total_processing_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # --- Run perception pipeline ---
        results = detector(frame)[0]
        yolo_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = results.names[cls]
            if class_name in ['car', 'truck', 'person', 'bicycle', 'motorbike']:
                yolo_detections.append(([int(x1), int(y1), int(w), int(h)], conf, class_name))
        
        tracks = tracker.update_tracks(yolo_detections, frame=frame)
        
        # --- Draw results and trajectories ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- NEW: Trajectory Logic ---
            # Get center point of the bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Initialize history for new tracks
            if track_id not in trajectory_history:
                trajectory_history[track_id] = deque(maxlen=30) # Store last 30 points
            
            # Append current center point
            trajectory_history[track_id].append((center_x, center_y))
            
            # Draw the historical path (tail)
            points = list(trajectory_history[track_id])
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)

            # Draw predicted future path
            if track.time_since_update == 0 and len(points) > 1:
                # Get velocity (pixels per frame)
                vx = (points[-1][0] - points[-2][0])
                vy = (points[-1][1] - points[-2][1])
                
                # Predict 10 frames into the future
                predicted_x = points[-1][0] + vx * 10
                predicted_y = points[-1][1] + vy * 10
                
                cv2.line(frame, points[-1], (int(predicted_x), int(predicted_y)), (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frames.append(frame_rgb)
        
        progress((frame_count + 1) / total_processing_frames, desc=f"Processing clip...")

    end_timestamp = time.time()
    elapsed_time = end_timestamp - start_timestamp
    processing_fps = 0
    if elapsed_time > 0:
        processing_fps = total_processing_frames / elapsed_time
    fps_string = f"{processing_fps:.2f} FPS"

    cap.release()
    
    if not output_frames:
        raise gr.Error("No frames were processed. Check start/end times.")

    output_path = "output.mp4"
    clip = ImageSequenceClip(output_frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", logger=None)
    
    print(f"Video processing complete. Speed: {fps_string}")
    
    return output_path, fps_string

# Interface definition remains the same
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Driving Video"),
        gr.Number(label="Start Time (seconds)", value=0),
        gr.Number(label="End Time (seconds)", value=10)
    ],
    outputs=[
        gr.Video(label="Tracked Video Clip"),
        gr.Textbox(label="Processing Speed")
    ],
    title="Autonomous Vehicle Perception Demo",
    description="Upload a video to see object tracking with trajectory prediction.",
    examples=[['data/test_video.mp4', 5, 15]]
)

if __name__ == "__main__":
    iface.launch()