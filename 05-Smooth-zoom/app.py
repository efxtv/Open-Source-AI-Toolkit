import gradio as gr
import whisper
import re
import os
import cv2
import numpy as np
from collections import Counter
from moviepy import VideoFileClip

print("Loading Whisper model... please wait.")
model = whisper.load_model("base")

def apply_zoom_logic(video_path):
    """Processes video frame-by-frame to add soft zoom during speech."""
    output_path = "zoomed_output.mp4"
    
    # 1. Transcribe to get speech segments
    print(f"--- Transcribing: {video_path}")
    result = model.transcribe(video_path, language="en")
    segments = result['segments']
    transcript = result['text']

    # 2. Open Video for Transformation
    clip = VideoFileClip(video_path)
    w, h = clip.size # MoviePy uses (width, height)

    def zoom_filter(get_frame, t):
        frame = get_frame(t)
        
        # Check if current timestamp 't' is within any speech segment
        is_speaking = any(s['start'] <= t <= s['end'] for s in segments)
        
        if is_speaking:
            # Find the specific segment to calculate smooth progress
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
            # Calculate new dimensions
            new_h, new_w = int(h / zoom), int(w / zoom)
            y1, x1 = (h - new_h) // 2, (w - new_w) // 2
            
            # Crop and Resize
            cropped = frame[y1:y1+new_h, x1:x1+new_w]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return frame

    # 3. Write the new video file
    print("--- Applying Zoom & Rendering (This may take a few minutes)...")
    zoomed_clip = clip.transform(zoom_filter)
    zoomed_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac", 
        fps=clip.fps, 
        logger=None
    )
    
    # Close clips to free up memory
    clip.close()
    zoomed_clip.close()
    
    return transcript, output_path

def generate_titles(text):
    """Extracts keywords and creates 5 viral titles."""
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

def process_all(video):
    if video is None:
        return "No video uploaded.", "", None
    
    transcript, zoomed_video = apply_zoom_logic(video)
    titles = generate_titles(transcript)
    
    return transcript, titles, zoomed_video

# --- Gradio Web Interface ---
with gr.Blocks(title="Transcribe-English-Videos") as demo:
    gr.Markdown("# 🎥 Smooth Auto-Zoom (English)")
    # Added Buy Me a Coffee link
    gr.Markdown("[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/efxtv)")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Original Video")
            run_btn = gr.Button("🚀 Start Analysis", variant="primary")
            
        with gr.Column():
            video_output = gr.Video(label="Zoomed Video Result")
            titles_output = gr.Textbox(label="Viral Title Ideas", lines=5)
            transcript_output = gr.Textbox(label="Full Transcript", lines=8)

    run_btn.click(
        fn=process_all,
        inputs=video_input,
        outputs=[transcript_output, titles_output, video_output]
    )

if __name__ == "__main__":
    # Launch on 0.0.0.0 to allow network/Docker access
    demo.launch(server_name="0.0.0.0", server_port=7860)
