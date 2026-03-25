import gradio as gr
import whisper_timestamped as whisper
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw
import numpy as np
import os
import re
import time

def to_hex(color_val):
    if not color_val: return "#FFFFFF"
    if str(color_val).startswith("#"): return color_val
    matches = re.findall(r"(\d+\.?\d*)", str(color_val))
    if len(matches) >= 3:
        r, g, b = map(lambda x: int(float(x)), matches[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#FFFFFF"

def make_rounded_rect(w, h, color, radius):
    image = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=color)
    return np.array(image)

def create_styled_word(text_str, f_size, t_color, s_color, s_width, bg_color, use_bg, p_l, p_r, p_t, p_b, radius, v_shift, duration):
    txt = TextClip(
        text=text_str,
        font_size=int(f_size),
        color=to_hex(t_color),
        stroke_color=to_hex(s_color),
        stroke_width=float(s_width),
        method='label',
        duration=duration,
        margin=(20, 20) 
    )

    mask_array = (txt.get_frame(0) * 255).astype('uint8')
    if mask_array.ndim == 3: mask_array = mask_array.mean(axis=2).astype('uint8')
    mask_im = Image.fromarray(mask_array)
    bbox = mask_im.getbbox() 

    if bbox:
        # Safety Buffer to prevent cutting
        safe_bbox = (
            max(0, bbox[0] - 10), 
            max(0, bbox[1] - 10), 
            min(txt.w, bbox[2] + 10), 
            min(txt.h, bbox[3] + 15) 
        )
        txt = txt.cropped(x1=safe_bbox[0], y1=safe_bbox[1], x2=safe_bbox[2], y2=safe_bbox[3])

    if not use_bg:
        return txt

    bg_w = int(txt.w + p_l + p_r)
    bg_h = int(txt.h + p_t + p_b)
    
    bg_img = make_rounded_rect(bg_w, bg_h, to_hex(bg_color), int(radius))
    bg_clip = ImageClip(bg_img).with_duration(duration)
    
    final_pos = ('center', (bg_h - txt.h)//2 + v_shift)
    return CompositeVideoClip([bg_clip, txt.with_position(final_pos)])

cache = {"result": None, "video_path": None}

def get_preview(video_path, main_c, stroke_c, stroke_w, bg_c, use_bg, f_size, v_type, p_l, p_r, p_t, p_b, radius, v_shift):
    if not video_path: return None
    video = VideoFileClip(video_path)
    frame_t = min(1.0, video.duration / 2)
    h = video.size[1]
    v_pos = int(h * 0.55) if v_type == "Short (9:16)" else int(h * 0.75)
    
    word_clip = create_styled_word("STYLE PREVIEW", f_size, main_c, stroke_c, stroke_w, bg_c, use_bg, p_l, p_r, p_t, p_b, radius, v_shift, 0.1)
    word_clip = word_clip.with_position(('center', v_pos))
    
    final_preview = CompositeVideoClip([video.subclipped(frame_t, frame_t + 0.05), word_clip])
    preview_path = f"live_preview_{int(time.time() * 1000)}.png"
    final_preview.save_frame(preview_path, t=0)
    video.close()
    return preview_path

def process_full_video(video_path, main_c, stroke_c, stroke_w, bg_c, use_bg, f_size, v_type, p_l, p_r, p_t, p_b, radius, v_shift, progress=gr.Progress()):
    if not video_path: return None
    video = VideoFileClip(video_path)
    if cache["video_path"] != video_path:
        progress(0.2, desc="Transcribing...")
        model = whisper.load_model("base", device="cpu")
        cache["result"] = whisper.transcribe(model, video_path)
        cache["video_path"] = video_path

    v_pos = int(video.size[1] * 0.55) if v_type == "Short (9:16)" else int(video.size[1] * 0.75)
    all_clips = [video]

    progress(0.5, desc="Rendering...")
    for segment in cache["result"]["segments"]:
        for word in segment["words"]:
            text_str = word["text"].upper().strip().replace(".", "").replace(",", "")
            word_clip = create_styled_word(text_str, f_size, main_c, stroke_c, stroke_w, bg_c, use_bg, p_l, p_r, p_t, p_b, radius, v_shift, word["end"]-word["start"])
            word_clip = word_clip.with_start(word["start"]).with_position(('center', v_pos))
            animated = word_clip.resized(lambda t: 0.6 + (t/0.05)*0.7 if t < 0.05 else (1.3 - ((t-0.05)/0.07)*0.3 if t < 0.12 else 1.0))
            all_clips.append(animated)

    output = "final_export.mp4"
    CompositeVideoClip(all_clips).write_videofile(output, codec="libx264", audio_codec="aac", fps=24, logger=None)
    video.close()
    return output

# --- UI (Fixed for Gradio 6.0 Warnings) ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎬 BEST CC Editor")
    # Added Buy Me a Coffee link
    gr.Markdown("[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/efxtv)")
    
    
    with gr.Row():
        with gr.Column(scale=1): # Fixed: integer scale
            video_input = gr.Video(label="Upload", height=230)
            with gr.Tabs():
                with gr.TabItem("Style"):
                    v_type = gr.Radio(["Short (9:16)", "Long-form (16:9)"], value="Short (9:16)", label="Format")
                    f_size = gr.Slider(20, 250, value=90, label="Size")
                    s_w = gr.Slider(0, 10, value=2, step=0.5, label="Stroke")
                    with gr.Row():
                        m_c = gr.ColorPicker(label="Text", value="#FFFFFF")
                        s_c = gr.ColorPicker(label="Stroke", value="#000000")
                
                with gr.TabItem("Box Padding"):
                    u_bg = gr.Checkbox(label="Enable Box", value=True)
                    b_c = gr.ColorPicker(label="Box Color", value="#000000")
                    rad = gr.Slider(0, 150, value=100, label="Corners")
                    v_shift = gr.Slider(-50, 50, value=0, label="Nudge")
                    with gr.Row():
                        p_t = gr.Slider(0, 150, value=20, label="Top")
                        p_b = gr.Slider(0, 150, value=20, label="Bottom")
                    with gr.Row():
                        p_l = gr.Slider(0, 200, value=50, label="Left")
                        p_r = gr.Slider(0, 200, value=50, label="Right")

            btn_gen = gr.Button("🚀 EXPORT", variant="primary")
            final_out = gr.Video(label="Result", height=200)

        with gr.Column(scale=2): # Fixed: integer scale
            preview_out = gr.Image(label="Live Preview", height=850)

    trig = [video_input, m_c, s_c, s_w, b_c, u_bg, f_size, v_type, p_l, p_r, p_t, p_b, rad, v_shift]
    for c in trig:
        c.change(fn=get_preview, inputs=trig, outputs=preview_out)

    btn_gen.click(fn=process_full_video, inputs=trig, outputs=final_out)

if __name__ == "__main__":
    # Fixed: moved css and theme here
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 98% !important}"
    )
