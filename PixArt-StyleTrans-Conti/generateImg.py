import os
import pandas as pd

import torch
from inference import inference_with_lora


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # path and dataset setting
        "image_size": 256,
        "prompts_file": "./CheckpointStyleDataset/prompts.csv",
        "BaseModel_path": "../../datasets/PixArt-XL-256",
        "LoRAModel_path": "./CheckpointOriginLoRA/class000",
        "save_sampled_images_path": "../../datasets/ContiStyle2/class000",
        # generating setting
        "batch_size": 35,
        "total_number": 35,
        "dtype": torch.float16,
    }

    # load data
    for i in range(0, 1000, 1):
        print("\nstart:", str(i).zfill(3))
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(3)}"
        for k in range(0, config["total_number"], config["batch_size"]):
            prompts = []
            for j, prompt in enumerate(list(pd.read_csv(config["prompts_file"])['caption'])):
                if j in [ 0,  1,  3,  4,  5,  7, 12,
                         14, 15, 16, 18, 19, 21, 24,
                         26, 30, 34, 40, 41, 46, 50,
                         51, 52, 53, 54, 55, 56, 59,
                         60, 67, 71, 72, 75, 83, 85]:
                    prompts.append(prompt)
                    # these prompts looks better
            print("\ngenerating: ", prompts)

            # inference
            images = inference_with_lora(prompt=prompts,
                                         lora_path=config["LoRAModel_path"],
                                         model_path=config["BaseModel_path"],
                                         dtype=config["dtype"],
                                         device=config["device"])

            # save
            save_folder = config["save_sampled_images_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(3)}"
            for j, (image, prompt) in enumerate(zip(images, prompts)):
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=False)
                try:  # A group of people in the mountains walking/skiing.jpg
                    image.save(os.path.join(save_folder, prompt+".jpg"))
                except FileNotFoundError:
                    continue

