import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sdxl-turbo"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe.to(device)
pipe.set_progress_bar_config(disable=True)

def generate_poster(prompt: str):
    """
    Generate a marketing poster image from text prompt.
    T4-safe, fast, deterministic enough for ads.
    """
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,   # sd-turbo sweet spot
        guidance_scale=0.0       # turbo is trained without CFG
    ).images[0]

    return image
