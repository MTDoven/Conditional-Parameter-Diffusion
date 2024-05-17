from inference import inference_with_lora
import torch

import pandas as pd
import os


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:6",
        # path and datasets setting
        "image_size": 256,
        "padding": 0,
        "prompts_file": "./CheckpointStyleDataset/prompts.csv",
        "BaseModel_path": "../../datasets/PixArt-XL-256",
        "LoRAModel_path": "./CheckpointSoupLoRA/class00",
        "save_sampled_images_path": "../../datasets/Generated/SoupStyles/style00",
        # generating setting
        "batch_size": 100,
        "total_number": 20000,
        "dtype": torch.float16,
    }

    # load dataset
    prompts_all = list(pd.read_csv(config["prompts_file"])['caption'])
    for i in [0, 1, 2, 3]:
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}"
        for k in range(0, config["total_number"], config["batch_size"]):
            prompts = prompts_all[k: k+config["batch_size"]]
            # we need to generate a lot to calculate FID.
            print("\ngenerating: ", prompts)

            # inference
            images = inference_with_lora(prompt=prompts,
                                         lora_path=config["LoRAModel_path"],
                                         model_path=config["BaseModel_path"],
                                         dtype=config["dtype"],
                                         device=config["device"])

            # save
            save_folder = os.path.join(config["save_sampled_images_path"][:-2] + str(i).zfill(2))
            for j, (image, prompt) in enumerate(zip(images, prompts)):
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=False)
                try:  # A group of people in the mountains walking/skiing.jpg
                    image.save(os.path.join(save_folder, prompt+".jpg"))
                except FileNotFoundError:
                    continue

