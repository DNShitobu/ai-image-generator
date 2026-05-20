# AI Image Generator

Text-to-image generation powered by **Stable Diffusion XL** via the Hugging Face Inference API.

## Features
- 7 art styles: Photorealistic, Oil Painting, Anime, Watercolor, Cyberpunk, Fantasy, and more
- Adjustable guidance scale and inference steps
- Negative prompt support
- No GPU required (uses HF Inference API)

## Setup

1. Get a free [Hugging Face token](https://huggingface.co/settings/tokens)
2. Set it as the `HF_TOKEN` environment variable
3. Run the app

## Run Locally
```bash
pip install -r requirements.txt
HF_TOKEN=your_token python app.py
```

## Live Demo
[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Dnshitobu/ai-image-generator)

---
Built by [Dnshitobu](https://github.com/Dnshitobu)
