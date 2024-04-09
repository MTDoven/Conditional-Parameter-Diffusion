from diffusers import PixArtAlphaPipeline
import torch

pipe = None


def inference(prompt: list, model_path="./PixArt-XL-256", dtype=torch.bfloat16, device="cuda"):
    global pipe
    if pipe is None:
        pipe = PixArtAlphaPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    # inference
    images = pipe(prompt=prompt).images
    return images


if __name__ == "__main__":
    config = {
        # TODO: dataset to load prompt
        "model_path": "./PixArt-XL-256",
        "dtype": torch.bfloat16,
        "device": "cuda",
    }