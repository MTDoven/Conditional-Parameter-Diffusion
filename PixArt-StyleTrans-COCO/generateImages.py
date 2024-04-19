import torch
from Inference import inference_with_lora


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:1",
        # path setting
        "BaseModel_path": "./PixArt-XL-256",
        "LoRAModel_path": "./CheckpointLoRAGen/class00.pt",
        "save_sampled_images_path": "./temp",
        "prompts": ["abc", "def"],
        "dtype": torch.float32,
    }

    for i in range(10):
        config["LoRAModel_path"] = config["LoRADDPM_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}.pt"
        images = sample(**config)
        inference_with_lora(prompt=config["prompts"],
                            lora_path=config["LoRAModel_path"],
                            model_path=config["BaseModel_path"],
                            dtype=config["dtype"],
                            device=config["device"])