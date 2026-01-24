import torch
from diffusers import StableDiffusionXLPipeline

MODEL_ID = "stabilityai/sdxl-turbo"

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe.to("cuda")

pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()

NEGATIVE_PROMPT = (
    "text, letters, typography, words, logo, watermark, "
    "symbols, signage, glyphs, numbers"
)

def generate_poster(prompt: str):
    """
    Generates a high-quality background image ONLY.
    """

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=6,     # Turbo sweet spot
        guidance_scale=0.8,        # Prevents hallucinated text
        height=768,
        width=768
    ).images[0]

    return image
