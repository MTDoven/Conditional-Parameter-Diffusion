from Model.VAE import OneDimVAE as VAE
from Dataset import Image2SafetensorsDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import AdamW, SGD
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
import os.path
import torch
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:5",
        # paths setting
        "image_size": 256,
        "dataset": Image2SafetensorsDataset,
        "image_data_path": "../../datasets/MultiStyles",
        "lora_data_path": "../PixArt-StyleTrans-Comp/CheckpointTrainLoRA",
        "result_save_path": "./CheckpointVAE/VAE-Transfer-3.pt",
        # big model structure
        "d_model": [32, 64, 128, 192, 256, 384, 512, 768, 1024, 64],
        "d_latent": 64,
        "num_parameters": 860336+424*2,
        "padding": 424,
        "last_length": 841,
        "kernel_size": 9,
        "num_layers": -1,
        "not_use_var": False,
        "use_elu_activator": True,
        # training setting
        "autocast": False,
        "lr": 0.0005,
        "weight_decay": 0.0,
        "epochs": 10000,
        "eta_min": 0.,
        "batch_size": 32,
        "num_workers": 8,
        "save_every": 1000,
        "kld_weight": 1e-10,
        "kld_start_epoch": 10000,
        "kld_rise_rate": 0.0,
    }

    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(config=config, project="VanillaVAE-Final")

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
                num_parameters=config["num_parameters"],
                last_length=config["last_length"],
                kernel_size=config["kernel_size"],
                num_layers=config["num_layers"],
                use_elu_activator=config["use_elu_activator"],)
    model = model.to(device)
    optimizer = AdamW(model.parameters(),
                      lr=config["lr"],
                      weight_decay=config["weight_decay"],)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config["epochs"],
                                  eta_min=config["eta_min"],)
    dataloader = DataLoader(config["dataset"](path_to_loras=config["lora_data_path"],
                                              path_to_images=config["image_data_path"],
                                              image_size=config["image_size"],
                                              padding=config["padding"]),
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            persistent_workers=True,
                            pin_memory=True,
                            shuffle=True,)
    scaler = torch.cuda.amp.GradScaler()

    for e in tqdm(range(config["epochs"])):
        for *condition, parameters in dataloader:
            optimizer.zero_grad()
            parameters = parameters.to(device)
            with autocast(enabled=config["autocast"], dtype=torch.bfloat16):
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

