import os
import torch
import gradio as gr
from diffusers import DiffusionPipeline

# 1. STORAGE CONFIGURATION
# This tells Hugging Face where to download and store the large model files.
# Change "./model_storage" to your preferred path (e.g., "/mnt/data/models")
os.environ["HF_HOME"] = "./model_storage"
os.environ["HF_HUB_CACHE"] = "./model_storage"

# 2. LOAD PIPELINE FOR CPU
print("Loading Z-Image-Turbo pipeline on CPU...")
print(f"Models will be stored in: {os.path.abspath('./model_storage')}")

pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.float32,  # float32 is most stable for CPU
    low_cpu_mem_usage=True,
)

# Move to CPU explicitly
pipe.to("cpu")

print("Pipeline loaded successfully!")

# 3. GENERATION LOGIC
def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    # Use CPU Generator
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    return image, seed

# 4. EXAMPLES & THEME
examples = [
    ["Young Chinese woman in red Hanfu, intricate embroidery, neon lightning-bolt lamp above palm."],
    ["A majestic dragon soaring through clouds at sunset, iridescent colors."],
    ["Cozy coffee shop interior, warm lighting, rain on windows, vintage aesthetic."],
    ["Astronaut riding a horse on Mars, cinematic lighting, sci-fi concept art."],
]

custom_theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="amber",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    radius_size="lg"
).set(
    button_primary_background_fill="*primary_500",
)

# 5. GRADIO INTERFACE
with gr.Blocks(theme=custom_theme, fill_height=True) as demo:
    gr.Markdown("# 🎨 Z-Image-Turbo (CPU Edition)")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="✨ Your Prompt", placeholder="Describe your image...", lines=4)
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                height = gr.Slider(512, 1024, value=512, step=64, label="Height")
                width = gr.Slider(512, 1024, value=512, step=64, label="Width")
                num_inference_steps = gr.Slider(1, 12, value=8, step=1, label="Steps (8 is best)")
                randomize_seed = gr.Checkbox(label="🎲 Random Seed", value=True)
                seed = gr.Number(label="Seed", value=42, visible=False)

            generate_btn = gr.Button("🚀 Generate Image", variant="primary")
            gr.Examples(examples=examples, inputs=[prompt])
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Result", type="pil")
            used_seed = gr.Number(label="Seed Used", interactive=False)

    # Interactions
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed],
        outputs=[output_image, used_seed],
    )

# 6. RUN SERVER
if __name__ == "__main__":
    # server_name="0.0.0.0" makes it accessible on your local network
    # server_port=7860 is the default Gradio port
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False 
    )
