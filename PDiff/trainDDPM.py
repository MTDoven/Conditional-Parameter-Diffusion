from Model.TransformerDDPM import UNet, GaussianDiffusionTrainer
from Model.TransformerVAE import TransformerVAE
from Dataset import ClassIndex2ParamDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import os.path
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:3",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "lora_data_path": "../DDPM-Classify-CIFAR100/CheckpointLoRADDPM",
        "vae_checkpoint_path": "./CheckpointVAE/VAE.pt",
        "result_save_path": "./CheckpointDDPM/UNet.pt",
        # model structure
        "d_model": 1024,
        "d_latent": 256,
        "num_layers": 6,
        "T": 1000,
        "num_class": 100,
        # training setting
        "lr": 0.001,
        "weight_decay": 2e-6,
        "epochs": 1000,
        "eta_min": 1e-7,
        "batch_size": 64,
        "num_workers": 16,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "clip_grad_norm": 1.0
    }

    wandb.init(project="TransformerDDPM")

    device = config["device"]
    unet = UNet(d_latent=config["d_latent"],
                num_layers=config["num_layers"],
                T=config["T"],
                num_class=config["num_class"])
    unet = unet.to(device)
    trainer = GaussianDiffusionTrainer(unet,
                                       beta_1=config["beta_1"],
                                       beta_T=config["beta_T"],
                                       T=config["T"])
    trainer = trainer.to(device)
    vae = TransformerVAE(d_model=config["d_model"],
                         d_latent=config["d_latent"],
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
            x_0 = torch.cat([mu, log_var], dim=1)
            loss = trainer(x_0, item.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config["clip_grad_norm"])
            optimizer.step()
            wandb.log({"epoch": e,
                       "loss: ": loss.item(),
                       "lr": optimizer.state_dict()['param_groups'][0]["lr"],})
        scheduler.step()

    torch.save(unet.cpu().state_dict(), config["result_save_path"])
    print("Finished Training...")
