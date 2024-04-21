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
        "device": "cuda:7",
        # paths setting
        "image_size": 256,
        "dataset": Image2SafetensorsDataset,
        "image_data_path": "../../datasets/Styles",
        "lora_data_path": "../../datasets/PixArt-LoRA-Dataset",
        "result_save_path": "./CheckpointVAE/VAE-Transfer-final-notusevar.pt",
        # big model structure
        "d_model": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1024, 64],
        "d_latent": 64,
        "num_parameters": 521888+176*2,
        "last_length": 255,
        "kernel_size": 9,
        "num_layers": -1,
        "not_use_var": True,
        "use_elu_activator": True,
        # training setting
        "lr": 0.0003,
        "weight_decay": 0.0,
        "epochs": 1200,
        "eta_min": 0.,
        "batch_size": 64,
        "num_workers": 32,
        "kld_weight": 0.0,
        "kld_start_epoch": 1201,
        "kld_rise_rate": 1e-6,
        "save_every": 20,
        "norm_weight": 0.0,
        "norm_start_epoch": 1201,
        "norm_rise_rate": 1e-7
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
                                              image_size=config["image_size"],),
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            persistent_workers=True,
                            pin_memory=True,
                            shuffle=True,)
    scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model)
    for e in tqdm(range(config["epochs"])):
        for condition, parameters in dataloader:
            optimizer.zero_grad()
            parameters = parameters.to(device)
            with autocast(enabled = e<config["epochs"]*0.8, dtype=torch.bfloat16):
                output = model(parameters, not_use_var=config["not_use_var"])
                losses = model.loss_function(*output, kld_weight=config["kld_weight"], norm_weight=config["norm_weight"])
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            wandb.log(losses)

        scheduler.step()
        if (e+1) % config["save_every"] == 0:
            torch.save(model.cpu().state_dict(), config["result_save_path"]+f".{e}")
            model.to(device)
        if (e+1) > config["norm_start_epoch"]:
            config["norm_weight"] += config["norm_rise_rate"]
        if (e+1) > config["kld_start_epoch"]:
            config["kld_weight"] += config["kld_rise_rate"]

    print("Finished Training...")

