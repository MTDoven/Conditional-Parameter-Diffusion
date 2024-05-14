from Model.DDPM import ODUNetTransfer as UNet
from Model.DDPM import GaussianDiffusionTrainer, GaussianDiffusionSampler
from Model.VAE import OneDimVAE as VAE
from Dataset import ContiImage2SafetensorsDataset
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
        "device": "cuda:5",
        # paths setting
        "image_size": 256,
        "dataset": ContiImage2SafetensorsDataset,
        "UNet_path": "./CheckpointDDPM/UNet-Continue-05.pt",
        "VAE_path": "./CheckpointVAE/VAE-Continue-05.pt",
        "path_to_loras": "../PixArt-StyleTrans-Conti/CheckpointOriginLoRA",
        "path_to_images": "../../datasets/ContiStyles", #"../PixArt-StyleTrans-Conti/CheckpointStyleDataset/evaluateStyles",
        "path_to_save": "../PixArt-StyleTrans-Conti/CheckpointGenLoRA",
        "adapter_config_path": "../PixArt-StyleTrans-Conti/CheckpointStyleDataset/adapter_config.json",
        # ddpm structure
        "num_channels": [64, 128, 256, 384, 512, 768, 1024, 24],
        "T": 1000,
        "num_class": 1000,
        "kernel_size": 5,
        "num_layers_diff": -1,
        "not_use_fc": True,
        "freeze_extractor": False,
        "simple_extractor": True,
        # model structure
        "d_model": [16, 32, 64, 128, 256, 512, 512, 32],
        "d_latent": 1024,
        "num_parameters": 516096,
        "padding": 0,
        "last_length": 2016,
        "kernel_size_vae": 9,
        "num_layers": -1,
        "not_use_var": False,
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
                not_use_fc=config["not_use_fc"],
                freeze_extractor=config["freeze_extractor"],
                simple_extractor=config["simple_extractor"])
    unet.load_state_dict(torch.load(config["UNet_path"]))
    unet = unet.to(device)
    vae = VAE(d_model=config["d_model"],
              d_latent=config["d_latent"],
              num_parameters=config["num_parameters"],
              last_length=config["last_length"],
              kernel_size=config["kernel_size_vae"],
              num_layers=config["num_layers"],
              use_elu_activator=config["use_elu_activator"],)
    diction = torch.load(config["VAE_path"], map_location="cpu")
    new_diction = {}
    for name, param in diction.items():
        if "_orig_mod" in name:
            new_diction[name.split(".", 1)[1]] = param
        else:  # not orig_mod
            new_diction[name] = param
    vae.load_state_dict(new_diction)
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
        for i in range(0, 1000, 100):
            for index in range(len(dataset)):
                image, param, item, prompt = dataset[index]
                if item == i:
                    condition.append(image)
                    print("\r", item, end="")
                    break
        condition = torch.stack(condition)
        noise = torch.randn(size=(config["batch_size"], config["d_latent"]), device=device)
        sampled = sampler(noise, condition.to(device))
        gen_parameters = vae.decode(sampled, num_parameters=config["num_parameters"])
        gen_parameters = gen_parameters

    for i, param in zip(range(0, 1000, 100), gen_parameters):
        dataset.save_param_dict(
            save_path=os.path.join(config["path_to_save"], f"class{str(i).zfill(3)}"),
            parameters=param,
            adapter_config_path=config["adapter_config_path"], )
    print(f"Generated parameters saved to {config['path_to_save']}")
