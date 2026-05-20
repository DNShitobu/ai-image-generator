"""
AI Image Generator — Gradio App
==================================
Text-to-image generation using Stable Diffusion XL via the
Hugging Face Inference API. No GPU required on the server side.
"""

import gradio as gr
import requests
import os
from PIL import Image
import io

# ── HF Inference API ───────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
API_URL   = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

STYLE_PREFIXES = {
    "None":              "",
    "Photorealistic":    "photorealistic, 8k resolution, highly detailed, ",
    "Oil Painting":      "oil painting, classical art style, brushstrokes, canvas texture, ",
    "Anime":             "anime style, vibrant colors, Studio Ghibli, detailed, ",
    "Watercolor":        "watercolor painting, soft colors, artistic, ",
    "Cyberpunk":         "cyberpunk, neon lights, futuristic city, dark atmosphere, ",
    "Fantasy":           "epic fantasy art, magical, dramatic lighting, concept art, ",
}

def generate_image(prompt: str, style: str, negative_prompt: str, guidance: float, steps: int):
    if not prompt.strip():
        return None, "Please enter a prompt."
    if not HF_TOKEN:
        return None, "Set the HF_TOKEN environment variable to use the Inference API."

    full_prompt = STYLE_PREFIXES[style] + prompt
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "negative_prompt": negative_prompt or "blurry, low quality, deformed, ugly",
            "guidance_scale":  guidance,
            "num_inference_steps": steps,
        }
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        return img, f"Generated with style: {style}"
    elif response.status_code == 503:
        return None, "Model is loading, please wait 20 seconds and try again."
    else:
        return None, f"API error {response.status_code}: {response.text[:200]}"

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Image Generator") as demo:
    gr.Markdown("# AI Image Generator")
    gr.Markdown(
        "Enter a text prompt and generate an image using **Stable Diffusion XL**. "
        "Add your Hugging Face token as the `HF_TOKEN` environment variable."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="A futuristic city at sunset with flying cars...",
                lines=3,
            )
            style_input = gr.Dropdown(
                choices=list(STYLE_PREFIXES.keys()),
                value="Photorealistic",
                label="Style",
            )
            negative_input = gr.Textbox(
                label="Negative Prompt (optional)",
                placeholder="blurry, low quality, deformed...",
                lines=2,
            )
            with gr.Row():
                guidance_input = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                steps_input    = gr.Slider(10, 50, value=25, step=5, label="Inference Steps")
            generate_btn = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            image_output  = gr.Image(type="pil", label="Generated Image")
            status_output = gr.Textbox(label="Status", interactive=False)

    gr.Examples(
        examples=[
            ["A majestic lion in a golden savanna at sunset", "Photorealistic", "", 7.5, 25],
            ["A wizard casting a spell in a magical forest", "Fantasy", "", 8.0, 30],
            ["Tokyo street at night with neon lights and rain", "Cyberpunk", "", 7.5, 25],
            ["A serene mountain lake with reflections", "Watercolor", "", 7.0, 25],
            ["Portrait of a samurai warrior", "Anime", "", 8.0, 30],
        ],
        inputs=[prompt_input, style_input, negative_input, guidance_input, steps_input],
        label="Example prompts",
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, style_input, negative_input, guidance_input, steps_input],
        outputs=[image_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()
