import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from moviepy import VideoFileClip
from tqdm import tqdm

# How to use
# app.py input/path/for/videos output/path/for/videos

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

def get_face_center(frame):
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        return bbox.xmin + (bbox.width / 2), bbox.ymin + (bbox.height / 2)
    return 0.5, 0.5

def process_video(input_path, output_path):
    clip = VideoFileClip(input_path)
    w, h = clip.size

    # --- SMOOTHING CONFIG ---
    # Lower 'smoothing' = more "lazy" camera (smoother). Try 0.05 for buttery movement.
    state = {"curr_x": 0.5, "curr_y": 0.5, "curr_zoom": 1.1}
    smoothing = 0.05 
    base_zoom = 1.12
    zoom_pulse = 0.04 # How much it "breathes" in and out

    def smooth_filter(get_frame, t):
        frame = get_frame(t)
        target_x, target_y = get_face_center(frame)

        # 1. Smooth Coordinate Movement (The "Gimbal" Effect)
        state["curr_x"] += (target_x - state["curr_x"]) * smoothing
        state["curr_y"] += (target_y - state["curr_y"]) * smoothing

        # 2. Subtle Zoom Pulse (Simulates a breathing camera)
        # Uses a sine wave based on time 't' to move zoom between 1.12 and 1.16
        dynamic_zoom = base_zoom + (zoom_pulse * np.sin(t * 1.5))
        state["curr_zoom"] += (dynamic_zoom - state["curr_zoom"]) * smoothing

        # 3. Calculate Dimensions
        z = state["curr_zoom"]
        new_h, new_w = int(h / z), int(w / z)

        # 4. Center & Clamp
        center_x_px = int(state["curr_x"] * w)
        center_y_px = int(state["curr_y"] * h)

        x1 = max(0, min(w - new_w, center_x_px - new_w // 2))
        y1 = max(0, min(h - new_h, center_y_px - new_h // 2))

        cropped = frame[y1:y1+new_h, x1:x1+new_w]
        
        # INTER_LANCZOS4 is higher quality than INTER_AREA for slight zooms
        # This reduces the "blurriness" you saw.
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

    print(f"--- Rendering Smooth Motion: {output_path}")
    tracked_clip = clip.transform(smooth_filter)
    tracked_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=clip.fps,
        preset="slow", # 'slow' improves quality and reduces blur/artifacts
        logger=None,
        threads=4
    )
    
    clip.close()
    tracked_clip.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.mp4', '.mov'))]

    for filename in tqdm(video_files, desc="Batch Progress"):
        process_video(os.path.join(args.input_dir, filename), 
                      os.path.join(args.output_dir, f"{Path(filename).stem}_pro_smooth.mp4"))

if __name__ == "__main__":
    main()
