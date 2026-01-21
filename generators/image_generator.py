import os
import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

# =========================
# 🔐 Colab HF Token
# =========================
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise RuntimeError("HF_TOKEN not found. Set it in Colab userdata.")

# =========================
# 🎨 SD 3.5 MEDIUM (T4 SAFE)
# =========================
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
pipe.to("cuda")

def generate_poster(prompt: str):
    image = pipe(
        prompt,
        num_inference_steps=24,
        guidance_scale=4.0
    ).images[0]
    return image
