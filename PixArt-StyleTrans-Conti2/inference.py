import os.path
import random

from diffusers import PixArtAlphaPipeline, Transformer2DModel
from peft import PeftModel
import numpy as np
import torch


pipe = None
now_lora_path = None


def inference(prompt: list, model_path="./PixArt-XL-256", dtype=torch.bfloat16, device="cuda"):
    global pipe
    if pipe is None:
        pipe = PixArtAlphaPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe = pipe.to(device)
    # inference
    images = pipe(prompt=prompt).images
    return images


def inference_with_lora(prompt: list, lora_path, model_path="./PixArt-XL-256", dtype=torch.bfloat16, device="cuda"):
    if lora_path is None or "None" in lora_path:
        return inference(prompt=prompt, model_path=model_path, dtype=dtype, device=device)
    # load model
    global pipe, now_lora_path
    if (pipe is None) or (lora_path != now_lora_path):
        transformer = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
        transformer = PeftModel.from_pretrained(model=transformer, model_id=lora_path)
        pipe = PixArtAlphaPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype)
        now_lora_path = lora_path
        del transformer
        torch.cuda.empty_cache()
        pipe = pipe.to(device)
    # inference
    images = pipe(prompt=prompt).images
    return images


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # path setting
        "BaseModel_path": "../../datasets/PixArt-XL-256",
        "LoRAModel_path": "./CheckpointGenLoRA/class000",
        "save_sampled_images_path": "./temp",
        # inference setting
        "prompts":  # list(pd.read_csv("./CheckpointStyleDataset/prompts.csv")['caption'])[0: 300],
                   ["a bunch of children walk on a sidewalk next to a school bus",
                    "A giraffe standing in a lush green field",
                    "a talbe sitting on a tiled floor holding four vases with flowers",
                    "Rider guiding elephant with grass in open air area",
                    "Trains are on the tracks near wires and a tower",],
        "dtype": torch.float16,
        "seed": 42
    }

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(config["seed"])

    for i in range(50, 1000, 100):
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(3)}"
        images = inference_with_lora(prompt=config["prompts"],
                                     lora_path=config["LoRAModel_path"],
                                     model_path=config["BaseModel_path"],
                                     dtype=config["dtype"],
                                     device=config["device"])
        for j, im in enumerate(images):
            im.save(os.path.join(config["save_sampled_images_path"], f"{str(i).zfill(3)}{str(j).zfill(3)}.jpg"))