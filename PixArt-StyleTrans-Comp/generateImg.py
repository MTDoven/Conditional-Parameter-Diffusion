import os
import torch
import random
from inference import inference_with_lora
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
        "padding": 1960,
        "prompts_file": "./CheckpointStyleDataset/prompts.csv",
        "BaseModel_path": "../../datasets/PixArt-XL-256",
        "path_to_loras": "./CheckpointTrainLoRA",
        "path_to_images": "../../datasets/MultiStyles",
        "LoRAModel_path": "./CheckpointOriginLoRA/class00",
        "save_sampled_images_path": "../../datasets/Generated/OriginMultiStyles/style00",
        # generating setting
        "batch_size": 100,
        "total_number": 20000,
        "dtype": torch.float16,
        # variable setting
        "label": 0,
    }

    dataset = Image2SafetensorsDataset(path_to_loras=config["path_to_loras"],
                                       path_to_images=config["path_to_images"],
                                       image_size=config["image_size"],
                                       padding=config["padding"]).eval()
    prompts_all = list(pd.read_csv(config["prompts_file"])['caption'])

    for i in [6, 7, 8, 10]:
        config["label"] = i
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}"

        for k in range(0, config["total_number"], config["batch_size"]):
            prompts = prompts_all[k: k+config["batch_size"]]
            print("\ngenerating: ", prompts)

            images = inference_with_lora(prompt=prompts,
                                         lora_path=config["LoRAModel_path"],
                                         model_path=config["BaseModel_path"],
                                         dtype=config["dtype"],
                                         device=config["device"])

            save_folder = os.path.join(config["save_sampled_images_path"][:-2] + str(i).zfill(2))
            for j, (image, prompt) in enumerate(zip(images, prompts)):
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=False)
                image.save(os.path.join(save_folder, prompt+".jpg"))

