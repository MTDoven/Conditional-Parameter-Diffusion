
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from Diffusion.Diffusion import GaussianDiffusionTrainer
from Diffusion.Scheduler import GradualWarmupScheduler
from Diffusion.Model import UNet
from tqdm import tqdm
import wandb


def train(**config):
    wandb.init(config=config, project="BaseDDPM")
    device = torch.device(config["device"])

    # dataset
    dataset = CIFAR100(
        root=config["CIFAR100_path"],
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize(config["img_size"], antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
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
    trainer = GaussianDiffusionTrainer(
        model=unet,
        beta_1=config["beta_1"],
        beta_T=config["beta_T"],
        T=config["T"])
    trainer = trainer.to(device)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config["epochs"],)
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
            if (i+1) % 50 == 0:
                wandb.log({"epoch": e,
                           "loss: ": loss.item(),
                           "lr": optimizer.state_dict()['param_groups'][0]["lr"]})
        warmUpScheduler.step()
        if (e+1) % 5 == 0:
            torch.save(unet.cpu().state_dict(), config["result_save_path"])
            unet = unet.to(device)


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:5",
        # path setting
        "CIFAR100_path": "./CIFAR100",
        "result_save_path": "./CheckpointBaseDDPM/BaseDDPM.pt",
        # model structure
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "img_size": 32,
        # training setting
        "lr": 1e-4,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "clip_grad_norm": 1.0,
        "multiplier": 2.0,
        "epochs": 200,
        "batch_size": 128,
        "num_workers": 32,
        "dropout": 0.15,
        "weight_decay": 2e-5
    }
    train(**config)
