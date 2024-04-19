from diffusers import PixArtAlphaPipeline
import torch


pipe = None
now_lora_path = None

def inference(prompt: list, model_path="./PixArt-XL-256", dtype=torch.bfloat16, device="cuda"):
    global pipe
    if pipe is None:
        pipe = PixArtAlphaPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    # inference
    images = pipe(prompt=prompt).images
    return images


def inference_with_lora(prompt: list, lora_path, model_path="./PixArt-XL-256", dtype=torch.bfloat16, device="cuda"):
    global pipe, now_lora_path
    if (pipe is None) or (lora_path != now_lora_path):
        pipe = PixArtAlphaPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.load_lora_weights(lora_path)
        now_lora_path = lora_path
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    # inference
    images = pipe(prompt=prompt).images
    return images
