from Model.DDPM import ODUNetClassify as UNet
from Model.DDPM import GaussianDiffusionTrainer
from Model.VAE import OneDimVAE as VAE
from Dataset import ClassIndex2ParamDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import wandb



if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:5",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "lora_data_path": "../DDPM-Classify-CIFAR10/CheckpointTrainLoRA",
        "vae_checkpoint_path": "./CheckpointVAE/VAE-Classify-1.pt",
        "result_save_path": "./CheckpointDDPM/UNet-Classify-1.pt",
        # diffusion structure
        "num_channels": [32, 64, 128, 192, 256, 384, 512, 64],
        "T": 1000,
        "num_class": 10,
        "kernel_size": 3,
        "num_layers_diff": -1,
        # vae structure
        "d_model": [32, 64, 96, 128, 192, 256, 384, 512, 64],
        "d_latent": 64,
        "kernel_size_vae": 7,
        "num_parameters": 54912+192*2,
        "half_padding": 192,
        "last_length": 108,
        "not_use_var": True,
        "use_elu_activator": True,
        # training setting
        "autocast": True,
        "lr": 0.002,
        "weight_decay": 0.0,
        "epochs": 2000,
        "eta_min": 0.0,
        "batch_size": 256,
        "num_workers": 32,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "clip_grad_norm": 1.0,
        "save_every": 100,
    }

    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(project="OneDimDDPM-Final", config=config)

    device = config["device"]
    unet = UNet(d_latent=config["d_latent"],
                num_channels=config["num_channels"],
                T=config["T"],
                num_class=config["num_class"],
                kernel_size=config["kernel_size"],
                num_layers=config["num_layers_diff"],)
    unet = unet.to(device)
    trainer = GaussianDiffusionTrainer(unet,
                                       beta_1=config["beta_1"],
                                       beta_T=config["beta_T"],
                                       T=config["T"])
    trainer = trainer.to(device)
    vae = VAE(d_model=config["d_model"],
              d_latent=config["d_latent"],
              kernel_size=config["kernel_size_vae"],
              num_parameters=config["num_parameters"],
              last_length=config["last_length"],
              use_elu_activator=config["use_elu_activator"],)
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
                            shuffle=True,
                            drop_last=True)
    scaler = torch.cuda.amp.GradScaler()

    for e in tqdm(range(config["epochs"])):
        for i, (item, param) in enumerate(dataloader):
            optimizer.zero_grad()
            with ((autocast(enabled = e<config["epochs"]*0.75 and config["autocast"], dtype=torch.bfloat16))):
                with torch.no_grad():
                    mu, log_var = vae.encode(param.to(device))
                    x_0 = vae.reparameterize(mu, log_var)
                    x_0 = x_0 * 0.01
                loss = trainer(x_0, item.to(device))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            wandb.log({"epoch": e,
                       "loss: ": loss.item(),
                       "lr": optimizer.state_dict()['param_groups'][0]["lr"],})
        scheduler.step()
        if (e+1) % config["save_every"] == 0:
            torch.save(unet.cpu().state_dict(), config["result_save_path"])
            unet = unet.to(device)

    print("Finished Training...")
