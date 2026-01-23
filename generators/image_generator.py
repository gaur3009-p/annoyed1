import torch
from diffusers import StableDiffusionXLPipeline

# ==================================================
# 🚀 SDXL TURBO CONFIG (FASTEST)
# ==================================================
MODEL_ID = "stabilityai/sdxl-turbo"

# ==================================================
# LOAD PIPELINE
# ==================================================
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe.to("cuda")

# ==================================================
# PERFORMANCE OPTIMIZATIONS
# ==================================================
pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()

# ==================================================
# NEGATIVE PROMPT (CRITICAL FOR QUALITY)
# ==================================================
NEGATIVE_PROMPT = (
    "text, letters, typography, words, logo, watermark, "
    "low quality, blurry, noise, grain, artifacts, "
    "distorted shapes, deformed, extra elements"
)

# ==================================================
# PUBLIC FUNCTION
# ==================================================
def generate_poster(prompt: str):
    """
    Ultra-fast poster background generation using SDXL Turbo.
    ~4–6 steps, high quality, no text.
    """

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=6,      # 🔥 Turbo sweet spot
        guidance_scale=1.5,         # Turbo requires LOW CFG
        height=768,
        width=768
    ).images[0]

    return image
