import gradio as gr
import whisper
import re
import os
from collections import Counter

print("Loading Whisper model... please wait.")
model = whisper.load_model("base")

def process_video(video_path):
    if video_path is None:
        return "Please upload a video file.", ""

    # 1. Transcribe (Forcing English)
    print(f"Processing: {video_path}")
    result = model.transcribe(video_path, language="en")
    transcript = result["text"]

    # 2. Extract Keywords for Titles
    content = transcript.lower()
    words = re.findall(r'\w+', content)
    boring = {'the', 'and', 'this', 'that', 'with', 'from', 'video', 'just', 'like', 'about', 'would'}
    keywords = [w for w in words if len(w) > 4 and w not in boring]
    
    counts = Counter(keywords).most_common(5)
    main_topic = counts[0][0].capitalize() if counts else "This Video"

    # 3. Generate Viral Titles
    hooks = [
        f"I Tried {main_topic} For 30 Days (And It Changed Everything)",
        f"Stop Doing This If You Want Better {main_topic}",
        f"The Truth About {main_topic} That No One Tells You",
        f"Why {main_topic} Is Actually A Waste Of Time",
        f"5 Simple Secrets To Master {main_topic} Fast",
        f"I Tested Every {main_topic} Method So You Don't Have To",
        f"How To {main_topic} Like A Pro (Step-By-Step)",
        f"The Only {main_topic} Guide You Will Ever Need",
        f"Everything I Wish I Knew Before Starting {main_topic}",
        f"Is {main_topic} Dead in 2026? (Honest Review)"
    ]
    
    formatted_titles = "\n".join([f"{i+1}. {t}" for i, t in enumerate(hooks)])
    
    return transcript, formatted_titles

# Create the Web Interface
with gr.Blocks(title="Transcribe any video into text in seconds") as demo:
    gr.Markdown("# 🎥 Transcribe any video into text in seconds")
    # Added Buy Me a Coffee link
    gr.Markdown("[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/efxtv)")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            btn = gr.Button("Analyze & Transcribe", variant="primary")
            
        with gr.Column():
            titles_output = gr.Textbox(label="Viral Title Ideas", lines=10)
            transcript_output = gr.Textbox(label="Full Transcript (CC)", lines=10)

    btn.click(
        fn=process_video, 
        inputs=video_input, 
        outputs=[transcript_output, titles_output]
    )

# Launch on 0.0.0.0 to make it accessible locally
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
