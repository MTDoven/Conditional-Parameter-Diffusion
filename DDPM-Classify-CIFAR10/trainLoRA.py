
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
    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
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
    unet.load_state_dict(torch.load(config["BaseDDPM_path"]), strict=False)
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
        T_max=config["epochs"],)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=config["multiplier"],
        warm_epoch=config["epochs"] // 40,
        after_scheduler=cosineScheduler)

    # start training
    wandb.watch(unet)
    saved_lora = 0
    for e in tqdm(range(config["epochs"])):
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            x_0 = images.to(device)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                loss = trainer(x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config["clip_grad_norm"])
            optimizer.step()
            wandb.log({"epoch": e,
                       "loss: ": loss.item(),
                       "lr": optimizer.state_dict()['param_groups'][0]["lr"]})
            if e >= config["epochs"]-10 and i % 10 == 0:
                state_dict = unet.state_dict()
                lora_state_dict = {}
                for name, param in state_dict.items():
                    if "lora" in name:
                        lora_state_dict[name] = param
                torch.save(lora_state_dict, config["result_save_path"] + f"/lora_class{config['label']}_number{saved_lora}.pt")
                saved_lora += 1
        warmUpScheduler.step()


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:3",
        # path setting
        "CIFAR100_path": "../../datasets/CIFAR10",
        "BaseDDPM_path": "./CheckpointBaseDDPM/BaseDDPM.pt",
        "result_save_path": "./CheckpointTrainLoRA",
        # model structure
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "attn": [1],
        "num_res_blocks": 2,
        "img_size": 32,
        # training setting
        "lr": 2e-6,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "clip_grad_norm": 1.0,
        "multiplier": 1.0,
        "epochs": 1000,
        "batch_size": 64,
        "num_workers": 24,
        "dropout": 0.0,
        "weight_decay": 0.0,
        # variable parameters
        "label": 0
    }

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    for label in range(0, 1, 1):
        config["label"] = label
        print(f"start training lora_class_{label}.pt")
        train(**config)
