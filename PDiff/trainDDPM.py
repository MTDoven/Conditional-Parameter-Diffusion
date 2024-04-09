from Model.DDPM import ODUNet as UNet
from Model.DDPM import GaussianDiffusionTrainer
from Model.VAE import OneDimVAE as VAE
from Dataset import ClassIndex2ParamDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import os.path
import wandb
torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:1",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "lora_data_path": "/data/personal/nus-wk/cpdiff/datasets/CIFAR10-LoRA-Dataset",
        "vae_checkpoint_path": "./CheckpointVAE/VAE-Classify.pt",
        "result_save_path": "./CheckpointDDPM/UNet.pt",
        # diffusion structure
        "num_channels": [64, 128, 256, 512, 1024],
        "T": 1000,
        "num_class": 100,
        "num_layers_diff": -1,
        # vae structure
        "d_model": [64, 128, 256, 512, 1024, 1024, 64],
        "d_latent": 512,
        "num_parameters": 54912,
        "last_length": 429,
        "num_layers": -1,
        # training setting
        "lr": 0.01,
        "weight_decay": 0.0,
        "epochs": 300,
        "eta_min": 0.0,
        "batch_size": 32,
        "num_workers": 32,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "clip_grad_norm": 1.0,
        "save_every": 20,
    }

    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(project="OneDimDDPM-Final", config=config)

    device = config["device"]
    unet = UNet(d_latent=config["d_latent"],
                num_channels=config["num_channels"],
                T=config["T"],
                num_class=config["num_class"],
                num_layers=config["num_layers_diff"],)
    unet = unet.to(device)
    trainer = GaussianDiffusionTrainer(unet,
                                       beta_1=config["beta_1"],
                                       beta_T=config["beta_T"],
                                       T=config["T"])
    trainer = trainer.to(device)
    vae = VAE(d_model=config["d_model"],
              d_latent=config["d_latent"],
              num_parameters=config["num_parameters"],
              last_length=config["last_length"],
              num_layers=config["num_layers"],)
    vae.load_state_dict(torch.load(config["vae_checkpoint_path"]))
    vae = vae.to(device)
    for name, param in vae.named_parameters():
        param.requires_grad = False
    optimizer = AdamW(unet.parameters(),
                      lr=config["lr"],
                      weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config["epochs"],
                                  eta_min=config["eta_min"], )
    dataloader = DataLoader(config["dataset"](config["lora_data_path"]),
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            pin_memory=True,
                            shuffle=True,)

    wandb.watch(unet)
    for e in tqdm(range(config["epochs"])):
        for i, (item, param) in enumerate(dataloader):
            optimizer.zero_grad()
            mu, log_var = vae.encode(param.to(device))
            x_0 = vae.reparameterize(mu, log_var)
            loss = trainer(x_0, item.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config["clip_grad_norm"])
            optimizer.step()
            wandb.log({"epoch": e,
                       "loss: ": loss.item(),
                       "lr": optimizer.state_dict()['param_groups'][0]["lr"],})
        scheduler.step()
        if (e+1) % config["save_every"] == 0:
            torch.save(unet.cpu().state_dict(), config["result_save_path"])
            unet = unet.to(device)

    print("Finished Training...")
