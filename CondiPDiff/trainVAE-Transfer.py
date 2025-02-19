from Model.VAE import OneDimVAE as VAE
from Dataset import Image2SafetensorsDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from tqdm.auto import tqdm
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda",
        # paths setting
        "image_size": 256,
        "dataset": Image2SafetensorsDataset,
        "checkpoint": None,
        "image_data_path": "../../datasets/FIDStyles",
        "lora_data_path": "../PixArt-StyleTrans-Comp/CheckpointTrainLoRA",
        "result_save_path": "./CheckpointVAE/VAE-Transfer.pt",
        # vae structure
        "d_model": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 64],
        "d_latent": 256,
        "num_parameters": 516096,
        "padding": 0,
        "last_length": 504,
        "kernel_size": 9,
        "num_layers": -1,
        "not_use_var": True,
        "use_elu_activator": True,
        # training setting
        "autocast": True,
        "lr": 0.0002,
        "weight_decay": 0.0,
        "epochs": 30,
        "eta_min": 1e-8,
        "batch_size": 64,
        "num_workers": 8,
        "save_every": 10,
        "kld_weight": 0.0,
        "kld_start_epoch": 0,
        "kld_rise_rate": 0.0,
        "kld_reset_every": 10000,
    }

    wandb.login(key="put your api key here")
    wandb.init(config=config, project="VanillaVAE-Final")

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
                num_parameters=config["num_parameters"],
                last_length=config["last_length"],
                kernel_size=config["kernel_size"],
                num_layers=config["num_layers"],
                use_elu_activator=config["use_elu_activator"],)
    if config.get("checkpoint") is not None:
        model.load_state_dict(torch.load(config["checkpoint"], map_location="cpu"))
    model = model.to(device)

    optimizer = Adam(model.parameters(),
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
            with autocast(enabled= e<config["epochs"]*0.9 and config["autocast"], dtype=torch.bfloat16):
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
            torch.save(model.cpu().state_dict(), config["result_save_path"])
            model.to(device)
        if (e+1) > config["kld_start_epoch"]:
            config["kld_weight"] += config["kld_rise_rate"]

    print("Finished Training...")

