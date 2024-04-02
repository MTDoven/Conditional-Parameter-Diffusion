from Model.VAE import VanillaVAE as VAE
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
        "lora_data_path": "../DDPM-Classify-CIFAR100/CheckpointLoRADDPM",
        "result_save_path": "./CheckpointVAE/VAE.pt",
        # model structure
        "d_model": [32, 64, 128, 256, 512, 1024, 2048],
        "d_latent": 1024,
        "num_layers": 7,
        # training setting
        "lr": 0.001,
        "weight_decay": 2e-6,
        "epochs": 100,
        "eta_min": 1e-7,
        "batch_size": 32,
        "num_workers": 4,
        "kld_weight": 0.005
    }

    wandb.init(config=config, project="VanillaVAE")

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
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
            output = model(parameters.to(device))
            losses = model.loss_function(*output, kld_weight=config["kld_weight"])
            losses["loss"].backward()
            optimizer.step()
            wandb.log(losses)
        scheduler.step()

    torch.save(model.cpu().state_dict(), config["result_save_path"])
    print("Finished Training...")

