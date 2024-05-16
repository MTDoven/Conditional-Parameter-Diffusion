import os
import torch
import random
from inference import inference_with_lora, inference
from Dataset import Image2SafetensorsDataset
from torchvision import transforms
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # path setting
        "image_size": 256,
        "prompts_file": "./CheckpointStyleDataset/prompts.csv",
        "BaseModel_path": "../../datasets/PixArt-XL-256",
        "LoRAModel_path": "./CheckpointOriginLoRA/class000",
        "save_sampled_images_path": "../../datasets/ContiStyle2/class000",
        # generating setting
        "batch_size": 35,
        "total_number": 35,
        "dtype": torch.float16,
        # variable setting
        "label": 0,
    }

    prompts_all = list(pd.read_csv(config["prompts_file"])['caption'])

    for i in range(0, 1000, 1):
        print("\nstart:", str(i).zfill(3))
        config["label"] = i
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(3)}"

        for k in range(0, config["total_number"], config["batch_size"]):
            prompts = []
            for j, prompt in enumerate(prompts_all):
                if j in [ 0, 1, 3, 4, 5, 7,12,
                         14,15,16,18,19,21,24,
                         26,30,34,40,41,46,50,
                         51,52,53,54,55,56,59,
                         60,67,71,72,75,83,85]:
                    prompts.append(prompt)
            print("\ngenerating: ", prompts)

            if "None" not in config["LoRAModel_path"]:
                images = inference_with_lora(prompt=prompts,
                                             lora_path=config["LoRAModel_path"],
                                             model_path=config["BaseModel_path"],
                                             dtype=config["dtype"],
                                             device=config["device"])
            else:  # "None" in config["LoRAModel_path"]
                images = inference(prompt=prompts,
                                   model_path=config["BaseModel_path"],
                                   dtype=config["dtype"],
                                   device=config["device"])

            save_folder = config["save_sampled_images_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(3)}"
            for j, (image, prompt) in enumerate(zip(images, prompts)):
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=False)
                try:  # A group of people in the mountains walking/skiing.jpg
                    image.save(os.path.join(save_folder, prompt+".jpg"))
                except FileNotFoundError:
                    continue

