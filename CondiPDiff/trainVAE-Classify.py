from Model.VAE import OneDimVAE as VAE
from Dataset import ClassIndex2ParamDataset

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.optim import AdamW
import torch

from tqdm.auto import tqdm
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "lora_data_path": "../DDPM-Classify-CIFAR10/CheckpointTrainLoRA",
        "result_save_path": "./CheckpointVAE/VAE-Classify-1.pt",
        # vae structure
        "d_model": [32, 64, 96, 128, 192, 256, 384, 512, 64],
        "d_latent": 64,
        "kernel_size": 7,
        "num_parameters": 54912+192*2,
        "half_padding": 192,
        "last_length": 108,
        "not_use_var": False,
        "use_elu_activator": True,
        # training setting
        "autocast": False,
        "lr": 0.0005,
        "weight_decay": 0.0,
        "epochs": 100,
        "eta_min": 0.,
        "batch_size": 256,
        "num_workers": 32,
        "save_every": 100,
        "kld_weight": 1e-10,
        "kld_start_epoch": 10000,
        "kld_rise_rate": 0.0,
    }

    wandb.login(key="put your wandb api key")
    wandb.init(config=config, project="VanillaVAE")

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
                kernel_size=config["kernel_size"],
                num_parameters=config["num_parameters"],
                last_length=config["last_length"],
                use_elu_activator=config["use_elu_activator"],)
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
                            shuffle=True,
                            persistent_workers=True)
    scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model)
    for e in tqdm(range(config["epochs"])):
        for condition, parameters in dataloader:
            optimizer.zero_grad()
            parameters = parameters.to(device)
            with autocast(enabled = e<config["epochs"]*0.75 and config["autocast"], dtype=torch.bfloat16):
                output = model(parameters, not_use_var=config["not_use_var"])
                losses = model.loss_function(*output,
                                             kld_weight=config["kld_weight"],
                                             not_use_var=config["not_use_var"])
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            wandb.log(losses)

        scheduler.step()
        if (e+1) % config["save_every"] == 0:
            torch.save(model.cpu().state_dict(), config["result_save_path"]+f".{e}")
            model.to(device)
        if (e+1) > config["kld_start_epoch"]:
            config["kld_weight"] += config["kld_rise_rate"]

    print("Finished Training...")

