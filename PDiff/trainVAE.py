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
        "lora_data_path": "../DDPM/test",
        "result_save_path": "./CheckpointVAE/VAE.pt",
        # model structure
        "d_model": 1024,
        "d_latent": 256,
        "num_layers": 2,
        # training setting
        "lr": 0.001,
        "weight_decay": 2e-6,
        "epochs": 200,
        "eta_min": 1e-7,
        "batch_size": 4,
        "num_workers": 0,
        "kld_weight": 0.005
    }

    wandb.init(config=config, project="TransformerVAE")

    device = config["device"]
    model = TransformerVAE(d_model=config["d_model"],
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

