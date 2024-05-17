from Model.DDPM import ODUNetTransfer as UNet
from Model.DDPM import GaussianDiffusionTrainer
from Model.VAE import OneDimVAE as VAE
from Dataset import ContiImage2SafetensorsDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast
import torch
from tqdm.auto import tqdm
import wandb


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # paths setting
        "image_size": 256,
        "dataset": ContiImage2SafetensorsDataset,
        "path_to_images": "../../datasets/ContiStyle2",
        "lora_data_path": "../PixArt-StyleTrans-Conti2/CheckpointOriginLoRA05",
        "vae_checkpoint_path": "./CheckpointVAE/VAE-Continue-05-02.pt",
        "result_save_path": "./CheckpointDDPM/UNet-Continue-05-02.pt",
        # diffusion structure
        "num_channels": [64, 128, 256, 512, 768, 1024, 1024, 32],
        "T": 1000,
        "num_class": 1000,
        "kernel_size": 3,
        "num_layers_diff": -1,
        "not_use_fc": False,
        "freeze_extractor": False,
        "simple_extractor": True,
        # vae structure
        "d_model": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 64],
        "d_latent": 256,
        "num_parameters": 516096,
        "padding": 0,
        "last_length": 504,
        "kernel_size_vae": 9,
        "num_layers": -1,
        "not_use_var": True,
        "use_elu_activator": True,
        # training setting
        "duplicate": 100,
        "autocast": True,
        "lr": 0.0005,
        "weight_decay": 0.1,
        "epochs": 80,
        "eta_min": 0.0,
        "batch_size": 256,
        "num_workers": 32,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "clip_grad_norm": 1.0,
        "save_every": 10,
    }

    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(project="OneDimDDPM-Final", config=config)

    device = config["device"]
    unet = UNet(d_latent=config["d_latent"],
                num_channels=config["num_channels"],
                T=config["T"],
                num_class=config["num_class"],
                kernel_size=config["kernel_size"],
                num_layers=config["num_layers_diff"],
                not_use_fc=config["not_use_fc"],
                freeze_extractor=config["freeze_extractor"],
                simple_extractor=config["simple_extractor"])
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
              kernel_size=config["kernel_size_vae"],
              num_layers=config["num_layers"],
              use_elu_activator=config["use_elu_activator"],)
    diction = torch.load(config["vae_checkpoint_path"], map_location="cpu")
    new_diction = {}
    for name, param in diction.items():
        if "_orig_mod" in name:
            new_diction[name.split(".", 1)[1]] = param
        else:  # not orig_mod
            new_diction[name] = param
    vae.load_state_dict(new_diction)
    vae = vae.to(device)

    for name, param in vae.named_parameters():
        param.requires_grad = False
    params_without_weight_decay, params_with_weight_decay = [], []
    for name, param in unet.named_parameters():
        if 'class_encode' in name:
            params_without_weight_decay.append(param)
        else:  # not class encode
            params_with_weight_decay.append(param)

    optimizer = AdamW(params=[{'params': params_with_weight_decay, 'weight_decay': config["weight_decay"]},
                              {'params': params_without_weight_decay}],
                      lr=config["lr"],)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config["epochs"],
                                  eta_min=config["eta_min"], )
    dataloader = DataLoader(config["dataset"](path_to_loras=config["lora_data_path"],
                                              path_to_images=config["path_to_images"],
                                              image_size=config["image_size"],
                                              padding=config["padding"],
                                              duplicate=config["duplicate"]),
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False,
                            persistent_workers=True,)
    scaler = torch.cuda.amp.GradScaler()

    for e in tqdm(range(config["epochs"])):
        for i, (item, param) in enumerate(dataloader):
            optimizer.zero_grad()
            with autocast(enabled=e<config["epochs"]*0.75 and config["autocast"], dtype=torch.bfloat16):
                with torch.no_grad():
                    mu, log_var = vae.encode(param.to(device))
                    x_0 = vae.reparameterize(mu, log_var, not_use_var=config["not_use_var"])
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
