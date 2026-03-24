import whisper
import os
import subprocess
import gradio as gr
import shutil

def process_video_source(num_clips, duration, mode, url, uploaded_file, progress=gr.Progress()):
    try:
        progress(0, desc="Initializing...")
        output_template = "source_video.mp4"
        
        # Determine Source: File Upload takes priority over URL
        if uploaded_file is not None:
            progress(0.1, desc="Processing uploaded file...")
            if os.path.exists(output_template):
                os.remove(output_template)
            shutil.copy(uploaded_file.name, output_template)
            video_path = output_template
        elif url:
            progress(0.1, desc="Downloading from YouTube...")
            video_path = download_youtube(url)
        else:
            return "❌ Error: Please provide either a YouTube URL or upload a video file.", []

        if video_path is None or not os.path.exists(video_path):
            return "❌ Source acquisition failed. Check URL or File.", []

        # 2. Transcribe
        progress(0.3, desc="Analyzing Audio (Whisper)...")
        model = whisper.load_model("base")
        result = model.transcribe(video_path, fp16=False)
        
        # 3. Logic
        progress(0.6, desc="Finding best segments...")
        best_parts = get_best_segments(result['segments'], int(num_clips), int(duration))
        
        output_dir = f"clips_{mode}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 4. Export
        clip_paths = []
        for i, clip in enumerate(best_parts):
            progress(0.6 + (0.3 * (i/len(best_parts))), desc=f"Exporting Clip {i+1}...")
            output_name = os.path.abspath(f"{output_dir}/clip_{i+1}.mp4")
            start_time = clip['start']
            
            if mode.lower() == "short":
                vf_filter = "crop=ih*(9/16):ih"
                cmd = f"ffmpeg -ss {start_time} -t {int(duration)} -i {video_path} -vf '{vf_filter}' -c:v libx264 -crf 23 -c:a aac -y {output_name}"
            else:
                cmd = f"ffmpeg -ss {start_time} -t {int(duration)} -i {video_path} -c:v libx264 -c:a aac -y {output_name}"
                
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clip_paths.append(output_name)

        progress(1.0, desc="Done!")
        return f"✅ Success! Created {len(clip_paths)} clips.", clip_paths
    
    except Exception as e:
        return f"❌ Error: {str(e)}", []

def download_youtube(url):
    output_template = "source_video.mp4"
    cookie_arg = "--cookies cookies.txt" if os.path.exists("cookies.txt") else ""
    cmd = f'yt-dlp {cookie_arg} -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" --no-check-certificate -o "{output_template}" "{url}"'
    result = subprocess.run(cmd, shell=True)
    return output_template if result.returncode == 0 else None

def get_best_segments(segments, num_clips, duration):
    scored_segments = []
    for s in segments:
        score = len(s['text'].split()) / (s['end'] - s['start'] + 0.1)
        scored_segments.append({'start': s['start'], 'score': score})
    
    scored_segments.sort(key=lambda x: x['score'], reverse=True)
    final = []
    for s in scored_segments:
        if len(final) >= int(num_clips): break
        if not any(abs(s['start'] - f['start']) < int(duration) for f in final):
            final.append(s)
    return final

# --- Gradio Interface ---
with gr.Blocks(title="Bulk Viral Video Generator") as demo:
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown("# 🎥 Bulk Viral Video Generator")
        with gr.Column(scale=2):
            # Added Buy Me a Coffee link
            gr.Markdown("[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/efxtv)")
    
    with gr.Row():
        with gr.Column():
            yt_url = gr.Textbox(label="YouTube URL (Optional if uploading file)", placeholder="Enter URL here...")
            file_input = gr.File(label="OR Upload Video File", file_types=["video"])
            
            with gr.Row():
                num_shorts = gr.Number(label="Number of Shorts", value=5, precision=0)
                duration = gr.Number(label="Seconds per Clip", value=60, precision=0)
            
            mode = gr.Radio(choices=["short", "long"], label="Format (short = 9:16 Vertical, long = 16:9 Landscape)", value="short")
            btn = gr.Button("🚀 Generate Viral Clips", variant="primary")
        
        with gr.Column():
            status = gr.Textbox(label="Status")
            gallery = gr.Gallery(label="Generated Clips", columns=2, height="auto")

    btn.click(
        fn=process_video_source, 
        inputs=[num_shorts, duration, mode, yt_url, file_input], 
        outputs=[status, gallery]
    )

if __name__ == "__main__":
    # Force 0.0.0.0 for Docker/Local Network access
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
