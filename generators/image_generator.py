import torch
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

# -----------------------------
# NF4 Quantization (Transformer only)
# -----------------------------
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # ✅ FIXED
)

transformer_nf4 = SD3Transformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16,
)

# -----------------------------
# Pipeline
# -----------------------------
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer_nf4,
    torch_dtype=torch.float16,
)

# -----------------------------
# Memory safety (MANDATORY)
# -----------------------------
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# -----------------------------
# Generation
# -----------------------------
def generate_background(prompt: str):
    image = pipe(
        prompt=prompt,
        num_inference_steps=28,     # 24–32 sweet spot
        guidance_scale=4.0,         # SD3 prefers low CFG
        max_sequence_length=512,
    ).images[0]

    return image
