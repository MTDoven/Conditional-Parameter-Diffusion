from diffusers import PixArtAlphaPipeline, Transformer2DModel
from peft import PeftModel
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
    global pipe, now_lora_path
    if (pipe is None) or (lora_path != now_lora_path):
        transformer = Transformer2DModel.from_pretrained(model_path,
                                                         subfolder="transformer",
                                                         torch_dtype=dtype)
        transformer = PeftModel.from_pretrained(model=transformer,
                                                model_id=lora_path)
        pipe = PixArtAlphaPipeline.from_pretrained(model_path,
                                                   transformer=transformer,
                                                   torch_dtype=dtype,
                                                   use_safetensors=False)
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
        "device": "cuda:2",
        # path setting
        "BaseModel_path": "./PixArt-XL-256",
        "LoRAModel_path": "./CheckpointLoRAGen/class00",
        "save_sampled_images_path": "./temp",
        "prompts": ["a strong man with dark skin playing in the beach.",
                    "Two women are rollerblading in front of some buses."],
        "dtype": torch.float16,
    }

    for i in range(10):
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}"
        images = inference_with_lora(prompt=config["prompts"],
                                     lora_path=config["LoRAModel_path"],
                                     model_path=config["BaseModel_path"],
                                     dtype=config["dtype"],
                                     device=config["device"])
        for j, im in enumerate(images):
            im.save(f"./temp/{str(i).zfill(3)}{str(j).zfill(3)}.jpg")