from Model.DDPM import ODUNet as UNet
from Model.DDPM import GaussianDiffusionTrainer, GaussianDiffusionSampler
from Model.VAE import OneDimVAE as VAE
from Dataset import ClassIndex2ParamDataset
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
        "device": "cuda:1",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "UNet_path": "./CheckpointDDPM/UNet.pt",
        "VAE_path": "./CheckpointVAE/VAE-Classify.pt",
        "path_to_loras": "/data/personal/nus-wk/cpdiff/datasets/CIFAR10-LoRA-Dataset",
        "path_to_save": "../DDPM-Classify-CIFAR100/CheckpointLoRAGen",
        # ddpm structure
        "num_channels": [64, 128, 256, 512, 512, 512],
        "T": 1000,
        "num_class": 100,
        "num_layers_diff": -1,
        # model structure
        "d_model": [64, 128, 256, 512, 1024, 1024, 64],
        "d_latent": 512,
        "num_parameters": 54912,
        "last_length": 429,
        "num_layers": -1,
        # training setting
        "batch_size": 100,
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
                num_layers=config["num_layers_diff"],)
    unet.load_state_dict(torch.load(config["UNet_path"]))
    unet = unet.to(device)
    vae = VAE(d_model=config["d_model"],
              d_latent=config["d_latent"],
              num_parameters=config["num_parameters"],
              last_length=config["last_length"],
              num_layers=config["num_layers"],)
    vae.load_state_dict(torch.load(config["VAE_path"]))
    vae = vae.to(device)
    sampler = GaussianDiffusionSampler(
        model=unet,
        beta_1=config["beta_1"],
        beta_T=config["beta_T"],
        T=config["T"])
    sampler = sampler.to(device)
    dataset = config["dataset"](config["path_to_loras"])


    unet.eval()
    vae.eval()
    with torch.no_grad():
        condition = torch.tensor([config["condition"] for _ in range(config["batch_size"])])
        noise = torch.randn(size=(config["batch_size"], config["d_latent"]), device=device)
        sampled = sampler(noise, condition.to(device))
        gen_parameters = vae.decode(sampled, num_parameters=config["num_parameters"])

    for i, param in enumerate(gen_parameters):
        dataset.save_param_dict(
            save_path=os.path.join(config["path_to_save"], f"class{str(i).zfill(2)}.pt"),
            parameters=param,)
    print(f"Generated parameters saved to {config['path_to_save']}")
