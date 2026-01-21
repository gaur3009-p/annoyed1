import torch
from diffusers import StableDiffusion3Pipeline

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)

# T4 memory safety
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

pipe.to(device)

def generate_poster(prompt: str):
    """
    Generate a high-quality marketing poster using SD 3.5 (T4-safe).
    """
    image = pipe(
        prompt,
        num_inference_steps=24,   # optimal for ads
        guidance_scale=4.0        # SD3 prefers low CFG
    ).images[0]

    return image
