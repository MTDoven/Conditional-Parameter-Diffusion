import os
import torch
import random
from inference import inference_with_lora
from Dataset import Image2SafetensorsDataset
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:2",
        # path setting
        "BaseModel_path": "./PixArt-XL-256",
        "path_to_loras": "../../datasets/PixArt-LoRA-Dataset",
        "path_to_images": "../../datasets/Styles",
        "LoRAModel_path": "./CheckpointLoRAGen/class00",
        "save_sampled_images_path": "../../datasets/Generated/GeneratedStyles/style0",
        # generating setting
        "batch_size": 100,
        "total_number": 400,
        "dtype": torch.float16,
        # variable setting
        "label": 0,
    }

    dataset = Image2SafetensorsDataset(path_to_loras=config["path_to_loras"],
                                       path_to_images=config["path_to_images"])
    dataset.eval()

    for i in range(10):
        lpipss = []
        config["label"] = i
        config["LoRAModel_path"] = config["LoRAModel_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}"

        for k in range(config["total_number"] // config["batch_size"]):

            ori_images, items, prompts = [], [], []
            for _ in range(config["batch_size"]):
                image, param, item, prompt = dataset[random.randint(0, len(dataset)-1)]
                ori_images.append(image)
                items.append(item)
                prompts.append(prompt)

            images = inference_with_lora(prompt=prompts,
                                         lora_path=config["LoRAModel_path"],
                                         model_path=config["BaseModel_path"],
                                         dtype=config["dtype"],
                                         device=config["device"])

            for j, (ori_image, image, item, prompt) in enumerate(zip(ori_images, images, items, prompts)):
                image.save(
                    os.path.join(config["save_sampled_images_path"][:-1] + str(i),
                                 prompt+".jpg"))
                transforms.ToPILImage()(ori_image).save(
                    os.path.join(config["save_sampled_images_path"][:-1] + str(i),
                                 prompt + ".ori" + ".jpg"))

