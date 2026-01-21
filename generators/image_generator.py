import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "Phr00t/Qwen-Image-Edit-Rapid-AIO"

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

def generate_poster(prompt):
    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0
    ).images[0]

    return image
