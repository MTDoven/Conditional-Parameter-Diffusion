from Model.TransformerVAE import TransformerVAE
from Dataset import ClassIndex2ParamDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import os.path
import torch
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:6",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "VAE_path": "./CheckpointVAE/VAE.pt",
        "path_to_loras": "../DDPM-Classify-CIFAR100/CheckpointLoRADDPM",
        "path_to_save": "../DDPM-Classify-CIFAR100/CheckpointGen",
        # model structure
        "d_model": 1024,
        "d_latent": 256,
        "num_layers": 2,
        # training setting
        "batch_size": 4,
        # variable parameters
        "num_parameters": 54912,
    }

    device = config["device"]
    model = TransformerVAE(d_model=config["d_model"],
                           d_latent=config["d_latent"],
                           num_layers=config["num_layers"],)
    model.load_state_dict(torch.load(config["VAE_path"]))
    model = model.to(device)
    dataset = config["dataset"](config["path_to_loras"])

    # evaluate
    model.eval()
    with torch.no_grad():
        gen_parameters = model.sample(
            num_samples=config["batch_size"],
            current_device=config["device"],
            num_parameters=config["num_parameters"],)

    for i, param in enumerate(gen_parameters):
        dataset.save_param_dict(
            save_path=os.path.join(config["path_to_save"], f"{str(i).zfill(4)}.pt"),
            parameters=param,)
    print(f"Generated parameters saved to {config['path_to_save']}")

