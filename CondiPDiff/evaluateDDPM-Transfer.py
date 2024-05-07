from Model.DDPM import ODUNetTransfer as UNet
from Model.DDPM import GaussianDiffusionTrainer, GaussianDiffusionSampler
from Model.VAE import OneDimVAE as VAE
from Dataset import Image2SafetensorsDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import os.path
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # paths setting
        "image_size": 256,
        "dataset": Image2SafetensorsDataset,
        "UNet_path": "./CheckpointDDPM/UNet-Transfer.pt",
        "VAE_path": "./CheckpointVAE/VAE-Transfer.pt",
        "path_to_loras": "../PixArt-StyleTrans-Comp/CheckpointTrainLoRA",
        "path_to_images": "../../datasets/MultiStyles",
        "path_to_save": "../PixArt-StyleTrans-Comp/CheckpointGenLoRA",
        "adapter_config_path": "../PixArt-StyleTrans-Comp/CheckpointStyleDataset/adapter_config.json",
        # ddpm structure
        "num_channels": [64, 128, 192, 256, 384, 512, 64],
        "T": 1000,
        "num_class": 10,
        "kernel_size": 3,
        "num_layers_diff": -1,
        "use_softmax": False,
        # model structure
        "d_model": [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1024, 64],
        "d_latent": 64,
        "num_parameters": 860336+1960*2,
        "padding": 1960,
        "last_length": 211,
        "kernel_size_vae": 11,
        "num_layers": -1,
        "not_use_var": True,
        "use_elu_activator": True,
        # training setting
        "batch_size": 10,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        # variable parameters
        "condition": 0
    }

    device = config["device"]
    unet = UNet(d_latent=config["d_latent"],
                num_channels=config["num_channels"],
                T=config["T"],
                num_class=config["num_class"],
                kernel_size=config["kernel_size"],
                num_layers=config["num_layers_diff"],
                use_softmax=config["use_softmax"], )
    unet.load_state_dict(torch.load(config["UNet_path"]))
    unet = unet.to(device)
    vae = VAE(d_model=config["d_model"],
              d_latent=config["d_latent"],
              num_parameters=config["num_parameters"],
              last_length=config["last_length"],
              kernel_size=config["kernel_size_vae"],
              num_layers=config["num_layers"],
              use_elu_activator=config["use_elu_activator"],)
    vae.load_state_dict(torch.load(config["VAE_path"]))
    vae = vae.to(device)
    sampler = GaussianDiffusionSampler(
        model=unet,
        beta_1=config["beta_1"],
        beta_T=config["beta_T"],
        T=config["T"])
    sampler = sampler.to(device)
    dataset = config["dataset"](path_to_loras=config["path_to_loras"],
                                path_to_images=config["path_to_images"],
                                image_size=config["image_size"],
                                padding=config["padding"],).eval()

    unet.eval()
    vae.eval()
    with torch.no_grad():
        condition = []
        for i in range(10):
            for index in range(len(dataset)):
                image, param, item, prompt = dataset[index]
                if item == i:
                    condition.append(image)
                    print("\r", item, prompt, end="")
                    break
        condition = torch.stack(condition)
        noise = torch.randn(size=(config["batch_size"], config["d_latent"]), device=device)
        sampled = sampler(noise, condition.to(device))
        gen_parameters = vae.decode(sampled * 400.0, num_parameters=config["num_parameters"])
        gen_parameters = gen_parameters

    for i, param in enumerate(gen_parameters):
        dataset.save_param_dict(
            save_path=os.path.join(config["path_to_save"], f"class{str(i).zfill(2)}"),
            parameters=param,
            adapter_config_path=config["adapter_config_path"], )
    print(f"Generated parameters saved to {config['path_to_save']}")
