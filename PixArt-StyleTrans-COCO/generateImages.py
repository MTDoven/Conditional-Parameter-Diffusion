import os
import torch
import random
from inference import inference_with_lora
from Dataset import Image2SafetensorsDataset
import lpips
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
        "batch_size": 60,
        "total_number": 600,
        "dtype": torch.float16,
        # variable setting
        "label": 0,
    }

    dataset = Image2SafetensorsDataset(path_to_loras=config["path_to_loras"],
                                       path_to_images=config["path_to_images"])
    dataset.eval()
    cal_lpips = lpips.LPIPS(net='vgg')

    def image_preprocess(img):
        if torch.is_tensor(img):
            if (img < 0).any():
                img = (img * 0.5) + 0.5
            if (img >= 0).all():
                img = (img * 2) - 1
            if len(img.shape) == 3:
                img = img[None]
            img = torch.clamp(img, min=-1, max=1)
            return img
        else:
            img = transforms.ToTensor()(img)
            return image_preprocess(img)


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

            for j, (ori_image, image, item) in enumerate(zip(ori_images, images, items)):
                image.save(os.path.join(config["save_sampled_images_path"][:-1]+str(i),
                                        str(j + k * config["batch_size"]).zfill(6)+".jpg"))
                if item == i:  # same style as the prompted image
                    lpips = cal_lpips(image_preprocess(ori_image), image_preprocess(image))
                    lpipss.append(lpips)
                    print(f"\rSSIM: {lpips}", end="")

        print(f"SSIM in style{i}:", sum(lpipss) / len(lpipss))
