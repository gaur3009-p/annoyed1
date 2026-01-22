import os
import torch
from huggingface_hub import login
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline

# ==================================================
# 🔐 LOAD HF TOKEN (MODULE SCOPE — IMPORTANT)
# ==================================================
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN is not set. "
        "Set it in Colab Secrets or as an environment variable."
    )

# Login once (safe to call multiple times)
login(token=HF_TOKEN)

# ==================================================
# MODEL CONFIG
# ==================================================
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ==================================================
# LOAD TRANSFORMER (GATED — TOKEN REQUIRED)
# ==================================================
transformer_nf4 = SD3Transformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)

# ==================================================
# LOAD PIPELINE (GATED — TOKEN REQUIRED)
# ==================================================
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer_nf4,
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)

# ==================================================
# MEMORY SAFETY (T4)
# ==================================================
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# ==================================================
# PUBLIC FUNCTION (USED BY app.py)
# ==================================================
def generate_poster(prompt: str):
    """
    Generates a background image only.
    Text must be overlaid separately using PIL.
    """
    image = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=4.0,
        max_sequence_length=512,
    ).images[0]

    return image
