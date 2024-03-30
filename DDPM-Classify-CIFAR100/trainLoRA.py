
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from LoRA.Dataset import OneClassDataset
from Diffusion.Diffusion import GaussianDiffusionTrainer
from Diffusion.Scheduler import GradualWarmupScheduler
from LoRA.Model import UNet
from tqdm import tqdm
import wandb


def train(**config):
    wandb.init(config=config, project="LoRADDPM")
    device = torch.device(config["device"])

    # dataset
    dataset = OneClassDataset(
        root=config["CIFAR100_path"],
        img_size=config["img_size"],
        label=config["label"],)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=True,
        shuffle=True,)

    # model setup
    unet = UNet(
        T=config["T"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        attn=config["attn"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],)
    unet = unet.to(device)
    for name, param in unet.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, unet.parameters())
    trainer = GaussianDiffusionTrainer(
        model=unet,
        beta_1=config["beta_1"],
        beta_T=config["beta_T"],
        T=config["T"])
    trainer = trainer.to(device)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config["lr"],
        weight_decay=config["weight_decay"],)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config["epoch"],)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=config["multiplier"],
        warm_epoch=config["epochs"] // 10,
        after_scheduler=cosineScheduler)

    # start training
    wandb.watch(unet)
    for e in tqdm(range(config["epochs"])):
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            x_0 = images.to(device)
            loss = trainer(x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config["clip_grad_norm"])
            optimizer.step()
            wandb.log({"epoch": e,
                       "loss: ": loss.item(),
                       "lr": optimizer.state_dict()['param_groups'][0]["lr"]})
        warmUpScheduler.step()

    state_dict = unet.cpu().state_dict()
    lora_state_dict = {}
    for name, param in state_dict.items():
        if "lora" in name:
            lora_state_dict[name] = param
    torch.save(lora_state_dict, config["result_save_path"])


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # path setting
        "CIFAR100_path": "./CIFAR100",
        "result_save_path": "./CheckpointLoRADDPM/LoRA.pt",
        # model structure
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "img_size": 224,
        # training setting
        "lr": 1e-4,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "clip_grad_norm": 1.0,
        "multiplier": 2.0,
        "epochs": 100,
        "batch_size": 4,
        "num_workers": 8,
        "dropout": 0.15,
        "weight_decay": 2e-6,
        # variable parameters
        "label": 0
    }

    for label in range(100):
        config["label"] = label
        config["result_save_path"] = f"./CheckpointLoRADDPM/lora_class_{label}.pt",
        print(f"start training lora_class_{label}.pt")
        train(**config)
