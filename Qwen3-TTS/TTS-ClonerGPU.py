# coding=utf-8
import os
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, login
from qwen_tts import Qwen3TTSModel

# 1. AUTHENTICATION
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    login(token=HF_TOKEN)

# 2. CONFIGURATION
MODEL_SIZES = ["0.6B", "1.7B"]
SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]

loaded_models = {}

def get_model_path(model_type: str, model_size: str) -> str:
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")

def get_lazy_model(model_key):
    global loaded_models
    
    if model_key not in loaded_models:
        # GPU handles multiple models better, but we still clear cache for efficiency
        torch.cuda.empty_cache()
        
        print(f"--- Loading {model_key} to GPU... ---")
        m_type, m_size = model_key.split('_')
        
        # GPU loading parameters: Using bfloat16 and FlashAttention 2
        model = Qwen3TTSModel.from_pretrained(
            get_model_path(m_type, m_size),
            device_map="cuda",            # Automatic mapping to available GPU
            torch_dtype=torch.bfloat16,   # High-precision speed on GPU
            token=HF_TOKEN,
            attn_implementation="flash_attention_2", # Use Flash-Attn for 3x speed
        )
        loaded_models[model_key] = model
        print(f"--- {model_key} is now ready on GPU. ---")
        
    return loaded_models[model_key]

# 3. AUDIO UTILS (Simplified normalization for speed)
def _normalize_audio(wav):
    y = np.asarray(wav).astype(np.float32)
    m = np.max(np.abs(y))
    if m > 1.0: y /= m
    return np.clip(y, -1.0, 1.0)

def _audio_to_tuple(audio):
    if audio is None: return None
    if isinstance(audio, tuple):
        return _normalize_audio(audio[1]), int(audio[0])
    return None

# 4. INFERENCE FUNCTIONS (Redirected to GPU Model)
def generate_voice_design(text, language, voice_description):
    try:
        model = get_lazy_model("VoiceDesign_1.7B")
        wavs, sr = model.generate_voice_design(
            text=text.strip(), language=language, instruct=voice_description.strip(),
            non_streaming_mode=True, max_new_tokens=512,
        )
        return (sr, wavs[0]), "Success"
    except Exception as e: return None, f"Error: {e}"

def generate_voice_clone(ref_audio, ref_text, target_text, language, model_size):
    audio_tuple = _audio_to_tuple(ref_audio)
    if not audio_tuple: return None, "Upload reference audio!"
    try:
        model = get_lazy_model(f"Base_{model_size}")
        wavs, sr = model.generate_voice_clone(
            text=target_text.strip(), language=language, ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            max_new_tokens=512,
        )
        return (sr, wavs[0]), "Success"
    except Exception as e: return None, f"Error: {e}"

def generate_custom_voice(text, language, speaker, instruct, model_size):
    try:
        model = get_lazy_model(f"CustomVoice_{model_size}")
        wavs, sr = model.generate_custom_voice(
            text=text.strip(), language=language, speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True, max_new_tokens=512,
        )
        return (sr, wavs[0]), "Success"
    except Exception as e: return None, f"Error: {e}"

# 5. UI (Streamlined for GPU Speed)
def build_ui():
    with gr.Blocks(title="Qwen3-TTS GPU Edition") as demo:
        gr.Markdown("# Qwen3-TTS GPU-Accelerated")
        
        with gr.Tabs():
            with gr.Tab("Voice Design (Prompt)"):
                with gr.Row():
                    with gr.Column():
                        d_txt = gr.Textbox(label="Text", value="The speed on GPU is incredible!")
                        d_lang = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                        d_inst = gr.Textbox(label="Description", value="An energetic, youthful voice.")
                        d_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        d_aud = gr.Audio(label="Output")
                        d_sts = gr.Textbox(label="Status")
                d_btn.click(generate_voice_design, [d_txt, d_lang, d_inst], [d_aud, d_sts])

            with gr.Tab("Voice Clone"):
                with gr.Row():
                    with gr.Column():
                        c_aud_in = gr.Audio(label="Reference Voice")
                        c_ref_t = gr.Textbox(label="Reference Text (Optional)")
                        c_tar_t = gr.Textbox(label="New Text")
                        c_size = gr.Dropdown(label="Model", choices=MODEL_SIZES, value="1.7B")
                        c_btn = gr.Button("Clone", variant="primary")
                    with gr.Column():
                        c_aud_out = gr.Audio(label="Output")
                        c_sts = gr.Textbox(label="Status")
                c_btn.click(generate_voice_clone, [c_aud_in, c_ref_t, c_tar_t, gr.State("Auto"), c_size], [c_aud_out, c_sts])

    return demo

if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
