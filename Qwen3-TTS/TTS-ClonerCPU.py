# coding=utf-8
# Qwen3-TTS CPU-Optimized (RAM-SAFE VERSION)
import os
import gc
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

# Global dictionary to track the active model
loaded_models = {}

def get_model_path(model_type: str, model_size: str) -> str:
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")

def clear_ram():
    """Explicitly deletes models and clears cache to free up system RAM."""
    global loaded_models
    print("--- Cleaning RAM... ---")
    for key in list(loaded_models.keys()):
        del loaded_models[key]
    loaded_models = {}
    gc.collect() # Force Python garbage collection
    print("--- RAM Cleared. ---")
    return "RAM Cleared Successfully"

def get_lazy_model(model_key):
    global loaded_models
    
    # If the requested model isn't the one currently in RAM, clear memory first
    if model_key not in loaded_models:
        clear_ram()
        
        print(f"--- Loading {model_key} (This may take a moment)... ---")
        m_type, m_size = model_key.split('_')
        
        # CPU loading parameters
        model = Qwen3TTSModel.from_pretrained(
            get_model_path(m_type, m_size),
            device_map="cpu",
            torch_dtype=torch.float32, 
            token=HF_TOKEN,
            attn_implementation="sdpa",
        )
        loaded_models[model_key] = model
        print(f"--- {model_key} is now ready in RAM. ---")
        
    return loaded_models[model_key]

# 3. AUDIO UTILS
def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float32) / max(abs(info.min), info.max) if info.min < 0 else (x.astype(np.float32) - (info.max + 1) / 2.0) / ((info.max + 1) / 2.0)
    else:
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6: y /= (m + eps)
    if clip: y = np.clip(y, -1.0, 1.0)
    return np.mean(y, axis=-1).astype(np.float32) if y.ndim > 1 else y

def _audio_to_tuple(audio):
    if audio is None: return None
    if isinstance(audio, tuple):
        return _normalize_audio(audio[1]), int(audio[0])
    if isinstance(audio, dict):
        return _normalize_audio(audio["data"]), int(audio["sampling_rate"])
    return None

# 4. INFERENCE FUNCTIONS
def generate_voice_design(text, language, voice_description):
    try:
        model = get_lazy_model("VoiceDesign_1.7B")
        wavs, sr = model.generate_voice_design(
            text=text.strip(), language=language, instruct=voice_description.strip(),
            non_streaming_mode=True, max_new_tokens=512,
        )
        return (sr, wavs[0]), "Success"
    except Exception as e: return None, f"Error: {e}"

def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None: return None, "Upload reference audio!"
    try:
        model = get_lazy_model(f"Base_{model_size}")
        wavs, sr = model.generate_voice_clone(
            text=target_text.strip(), language=language, ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only, max_new_tokens=512,
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

# 5. UI CONSTRUCTION
def build_ui():
    with gr.Blocks(title="Qwen3-TTS RAM-SAFE CPU") as demo:
        gr.Markdown("# Qwen3-TTS RAM-Optimized (CPU)")
        
        with gr.Row():
            mem_btn = gr.Button("🗑️ Clear RAM Manually", variant="stop")
            mem_status = gr.Textbox(label="Memory Status", placeholder="Click to clear RAM if system gets slow")
        mem_btn.click(clear_ram, outputs=mem_status)

        with gr.Tabs():
            with gr.Tab("Voice Design (Prompt)"):
                with gr.Row():
                    with gr.Column():
                        d_txt = gr.Textbox(label="Text", value="It is amazing how clear this voice is!")
                        d_lang = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                        d_inst = gr.Textbox(label="Description", value="A warm, professional male voice.")
                        d_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        d_aud = gr.Audio(label="Output")
                        d_sts = gr.Textbox(label="Status")
                d_btn.click(generate_voice_design, [d_txt, d_lang, d_inst], [d_aud, d_sts])

            with gr.Tab("Voice Clone (Reference)"):
                with gr.Row():
                    with gr.Column():
                        c_aud_in = gr.Audio(label="Upload Voice to Clone")
                        c_ref_t = gr.Textbox(label="What is being said in the audio?")
                        c_tar_t = gr.Textbox(label="New Text to Say")
                        c_size = gr.Dropdown(label="Model Size (0.6B is safer for RAM)", choices=MODEL_SIZES, value="0.6B")
                        c_btn = gr.Button("Clone & Generate", variant="primary")
                    with gr.Column():
                        c_aud_out = gr.Audio(label="Output")
                        c_sts = gr.Textbox(label="Status")
                c_btn.click(generate_voice_clone, [c_aud_in, c_ref_t, c_tar_t, gr.State("Auto"), gr.State(False), c_size], [c_aud_out, c_sts])

            with gr.Tab("TTS (Fixed Speakers)"):
                with gr.Row():
                    with gr.Column():
                        t_txt = gr.Textbox(label="Text", value="Hello there, I am a predefined speaker.")
                        t_spk = gr.Dropdown(label="Speaker", choices=SPEAKERS, value="Ryan")
                        t_size = gr.Dropdown(label="Model Size", choices=MODEL_SIZES, value="0.6B")
                        t_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        t_aud = gr.Audio(label="Output")
                        t_sts = gr.Textbox(label="Status")
                t_btn.click(generate_custom_voice, [t_txt, gr.State("English"), t_spk, gr.State(""), t_size], [t_aud, t_sts])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
