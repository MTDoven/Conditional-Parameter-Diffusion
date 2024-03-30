from Model.TransformerDDPM import UNet, GaussianDiffusionTrainer, GaussianDiffusionSampler
from Model.TransformerVAE import TransformerVAE
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
        "device": "cuda:3",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "UNet_path": "./CheckpointDDPM/UNet.pt",
        "VAE_path": "./CheckpointVAE/VAE.pt",
        "path_to_loras": "path/to/loras",
        # model structure
        "d_model": 1024,
        "d_latent": 256,
        "num_layers": 2,
        "T": 1000,
        "num_class": 100,
        # training setting
        "batch_size": 4,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        # variable parameters
        "num_parameters": 54912,
        "condition": 0
    }

    device = config["device"]
    unet = UNet(d_latent=config["d_latent"],
                num_layers=config["num_layers"],
                T=config["T"],
                num_class=config["num_class"],)
    unet.load_state_dict(torch.load(config["UNet_path"]))
    unet = unet.to(device)
    vae = TransformerVAE(d_model=config["d_model"],
                         d_latent=config["d_latent"],
                         num_layers=config["num_layers"],)
    vae.load_state_dict(torch.load(config["VAE_path"]))
    vae = vae.to(device)
    sampler = GaussianDiffusionSampler(
        model=unet,
        beta_1=modelConfig["beta_1"],
        beta_T=modelConfig["beta_T"],
        T=modelConfig["T"])
    sampler = sampler.to(device)
    dataset = config["dataset"]("path_to_loras")


    unet.eval()
    vae.eval()
    with torch.no_grad():
        condition = torch.tensor([config["condition"] for _ in config["batch_size"]])
        noise = torch.randn(size=(config["batch_size"], config["d_latent"]), device=device)
        sampled = sampler(noise, condition)
        result = vae.decode(sampled)

    dataset.save_param_dict(
        save_path=config["path_to_save"],
        parameters=gen_parameters,)
    print(f"Generated parameters saved to {config['path_to_save']}")
