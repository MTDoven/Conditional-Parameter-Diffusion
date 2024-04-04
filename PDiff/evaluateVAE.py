from Model.VAE import OneDimVAE as VAE
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
        "device": "cuda:4",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "VAE_path": "./CheckpointVAE/VAE.pt",
        "path_to_loras": "/data/personal/nus-wk/condipdiff/DDPM-LoRA-Dataset",
        "path_to_save": "../DDPM-Classify-CIFAR100/CheckpointLoRAGen",
        # vae structure
        "d_model": [32, 64, 128, 256, 512, 512, 8],
        "d_latent": 256,
        "num_parameters": 54912,
        "last_length": 429,
        "num_layers": -1,
        # test setting
        "batch_size": 4,
    }

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
                num_parameters=config["num_parameters"],
                last_length=config["last_length"],
                num_layers=config["num_layers"],)
    model.load_state_dict(torch.load(config["VAE_path"]))
    model = model.to(device)
    dataset = config["dataset"](config["path_to_loras"])

    # evaluate
    model.eval()
    with torch.no_grad():
        for i in range(100):
            item, param = dataset[i]
            gen_parameter = model.generate(
                x=param[None, :].to(device),
                num_parameters=config["num_parameters"],)
            param = gen_parameter.detach().cpu()[0]

            dataset.save_param_dict(
                save_path=os.path.join(config["path_to_save"], f"{str(i).zfill(4)}.pt"),
                parameters=param,)
    print(f"Generated parameters saved to {config['path_to_save']}")


