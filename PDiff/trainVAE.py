from Model.VAE import OneDimVAE as VAE
from Dataset import ClassIndex2ParamDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import AdamW, SGD
from tqdm.auto import tqdm
import os.path
import torch
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:5",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "lora_data_path": "/data/personal/nus-wk/condipdiff/DDPM-LoRA-Dataset",
        "result_save_path": "./CheckpointVAE/VAE.pt",
        # model structure
        "d_model": [32, 64, 128, 256, 512, 512, 8],
        "d_latent": 256,
        "num_parameters": 54912,
        "last_length": 429,
        "num_layers": -1,
        # training setting
        "lr": 0.001,
        "weight_decay": 2e-6,
        "epochs": 400,
        "eta_min": 0.,
        "batch_size": 32,
        "clip_grad_norm": 1.0,
        "num_workers": 16,
        "kld_weight": 0.0,
        "kld_start_epoch": 200,
        "kld_rise_rate": 0.0001,
        "save_every": 20,
    }

    wandb.init(config=config, project="VanillaVAE-Final")

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
                num_parameters=config["num_parameters"],
                last_length=config["last_length"],
                num_layers=config["num_layers"],)
    model = model.to(device)
    optimizer = AdamW(model.parameters(),
                      lr=config["lr"],
                      weight_decay=config["weight_decay"],)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config["epochs"],
                                  eta_min=config["eta_min"],)
    dataloader = DataLoader(config["dataset"](config["lora_data_path"]),
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            pin_memory=True,
                            shuffle=True,)

    wandb.watch(model)
    for e in tqdm(range(config["epochs"])):
        for condition, parameters in dataloader:
            optimizer.zero_grad()
            parameters = parameters.to(device)
            output = model(parameters)
            losses = model.loss_function(*output, kld_weight=config["kld_weight"])
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            optimizer.step()
            wandb.log(losses)

        scheduler.step()
        if (e+1) > config["kld_start_epoch"]:
            config["kld_weight"] += config["kld_rise_rate"]
            if (e+1) % config["save_every"] == 0:
                torch.save(model.cpu().state_dict(), config["result_save_path"]+f".{e}")
                model.to(device)

    print("Finished Training...")

