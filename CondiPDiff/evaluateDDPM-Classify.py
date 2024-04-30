from Model.DDPM import ODUNetClassify as UNet
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
        "device": "cuda:5",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "UNet_path": "./CheckpointDDPM/UNet-Classify-1.pt",
        "VAE_path": "./CheckpointVAE/VAE-Classify.pt",
        "path_to_loras": "../DDPM-Classify-CIFAR10/CheckpointTrainLoRA",
        "path_to_save": "../DDPM-Classify-CIFAR10/CheckpointGenLoRA",
        # ddpm structure
        "num_channels": [32, 64, 128, 192, 256, 384, 32],
        "T": 1000,
        "num_class": 10,
        "kernel_size": 3,
        "num_layers_diff": -1,
        # model structure
        "d_model": [64, 96, 128, 192, 256, 384, 512, 768, 128],
        "d_latent": 128,
        "kernel_size_vae": 7,
        "num_parameters": 54912+192*2,
        "half_padding": 192,
        "last_length": 108,
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
                num_layers=config["num_layers_diff"],)
    unet.load_state_dict(torch.load(config["UNet_path"]))
    unet = unet.to(device)
    vae = VAE(d_model=config["d_model"],
              d_latent=config["d_latent"],
              kernel_size=config["kernel_size_vae"],
              num_parameters=config["num_parameters"],
              last_length=config["last_length"],
              use_elu_activator=config["use_elu_activator"],)
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
        condition = torch.tensor([i for i in range(config["batch_size"])])
        noise = torch.randn(size=(config["batch_size"], config["d_latent"]), device=device)
        sampled = sampler(noise, condition.to(device))
        gen_parameters = vae.decode(sampled * 1.0, num_parameters=config["num_parameters"])
        gen_parameters = gen_parameters

    for i, param in enumerate(gen_parameters):
        dataset.save_param_dict(
            save_path=os.path.join(config["path_to_save"], f"class{str(i).zfill(2)}.pt"),
            parameters=param,)
    print(f"Generated parameters saved to {config['path_to_save']}")
