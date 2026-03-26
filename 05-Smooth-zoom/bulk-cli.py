import whisper
import re
import os
import cv2
import sys
import argparse
import numpy as np
from collections import Counter
from moviepy import VideoFileClip
from pathlib import Path
# How to use
# app.py input/path/for/videos output/path/for/videos

# Load Whisper Model
print("Loading Whisper model... please wait.")
model = whisper.load_model("base")

def apply_zoom_logic(video_path, output_path):
    """Processes video frame-by-frame to add soft zoom during speech."""
    
    # 1. Transcribe to get speech segments
    print(f"--- Transcribing: {video_path}")
    result = model.transcribe(video_path, language="en")
    segments = result['segments']
    transcript = result['text']

    # 2. Open Video for Transformation
    clip = VideoFileClip(video_path)
    w, h = clip.size 

    def zoom_filter(get_frame, t):
        frame = get_frame(t)
        
        # Check if current timestamp 't' is within any speech segment
        is_speaking = any(s['start'] <= t <= s['end'] for s in segments)
        
        if is_speaking:
            try:
                current_s = next(s for s in segments if s['start'] <= t <= s['end'])
                duration = max(current_s['end'] - current_s['start'], 0.1)
                progress = (t - current_s['start']) / duration
                # Zoom scale: Start at 1.0, end at 1.1 (10% zoom)
                zoom = 1.0 + (0.1 * progress)
            except StopIteration:
                zoom = 1.0
        else:
            zoom = 1.0

        if zoom > 1.0:
            new_h, new_w = int(h / zoom), int(w / zoom)
            y1, x1 = (h - new_h) // 2, (w - new_w) // 2
            cropped = frame[y1:y1+new_h, x1:x1+new_w]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return frame

    # 3. Write the new video file
    print(f"--- Rendering: {output_path}")
    zoomed_clip = clip.transform(zoom_filter)
    zoomed_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac", 
        fps=clip.fps, 
        logger=None
    )
    
    clip.close()
    zoomed_clip.close()
    return transcript

def generate_titles(text):
    """Extracts keywords and creates viral titles."""
    words = re.findall(r'\w+', text.lower())
    boring = {'the', 'and', 'this', 'that', 'with', 'from', 'video', 'just', 'like', 'about', 'would'}
    keywords = [w for w in words if len(w) > 4 and w not in boring]
    
    counts = Counter(keywords).most_common(1)
    topic = counts[0][0].capitalize() if counts else "This Video"
    
    hooks = [
        f"The Truth About {topic} You Need To Know",
        f"I Tried {topic} For 30 Days (My Results)",
        f"Why {topic} Is Actually A Huge Mistake",
        f"5 Secrets To Master {topic} In Minutes",
        f"Is {topic} Dead In 2026? Honest Review"
    ]
    return "\n".join([f"{i+1}. {t}" for i, t in enumerate(hooks)])

def main():
    parser = argparse.ArgumentParser(description="Batch process videos with auto-zoom.")
    parser.add_argument("input_dir", help="Path to folder containing source videos")
    parser.add_argument("output_dir", help="Path to folder where processed videos will be saved")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Supported video formats
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    
    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]

    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting process...")

    for filename in video_files:
        input_path = os.path.join(args.input_dir, filename)
        
        # Create output filename: original_name_zoom.mp4
        file_stem = Path(filename).stem
        output_filename = f"{file_stem}_zoom.mp4"
        output_path = os.path.join(args.output_dir, output_filename)

        print(f"\n>> Processing: {filename}")
        try:
            transcript = apply_zoom_logic(input_path, output_path)
            
            # Optional: Save titles to a text file next to the video
            titles = generate_titles(transcript)
            title_file = os.path.join(args.output_dir, f"{file_stem}_titles.txt")
            with open(title_file, "w") as f:
                f.write(titles)
                
            print(f"Done! Saved to {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
