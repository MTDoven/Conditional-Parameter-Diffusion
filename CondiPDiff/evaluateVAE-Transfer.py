from Model.VAE import OneDimVAE as VAE
from Dataset import Image2SafetensorsDataset
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
        "device": "cuda:5",
        # paths setting
        "dataset": Image2SafetensorsDataset,
        "VAE_path": "./CheckpointVAE/VAE-Transfer-2.pt.5999",
        "path_to_loras": "../PixArt-StyleTrans-Comp/CheckpointTrainLoRA",
        "path_to_images": "../../datasets/MultiStyles",
        "path_to_save": "../PixArt-StyleTrans-Comp/CheckpointGenLoRA",
        "adapter_config_path": "../PixArt-StyleTrans-Comp/CheckpointStyleDataset/adapter_config.json",
        # vae structure
        "d_model": [16, 32, 64, 96, 128, 192, 256, 384, 512, 64],
        "d_latent": 64,
        "num_parameters": 860336+424*2,
        "padding": 424,
        "last_length": 841,
        "kernel_size": 9,
        "num_layers": -1,
        "not_use_var": True,
        "use_elu_activator": True,
    }

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
                num_parameters=config["num_parameters"],
                last_length=config["last_length"],
                kernel_size=config["kernel_size"],
                num_layers=config["num_layers"],
                use_elu_activator=config["use_elu_activator"],)
    model.load_state_dict(torch.load(config["VAE_path"]))
    model = model.to(device)
    dataset = config["dataset"](config["path_to_loras"], config["path_to_images"])
    dataset.eval()

    # evaluate
    model.eval()
    with torch.no_grad():
        for i in range(16):
            for index in range(len(dataset)):
                image, param, item, prompt = dataset[index]
                if item == i:
                    print("\r", item, end="")
                    break
            gen_parameter = model.generate(
                x=param[None, :].to(device),
                num_parameters=config["num_parameters"],
                not_use_var=config["not_use_var"],)
            param = gen_parameter.detach().cpu()[0]
            dataset.save_param_dict(
                save_path=os.path.join(config["path_to_save"], f"class{str(i).zfill(2)}"),
                parameters=param,
                adapter_config_path=config["adapter_config_path"],)
    print(f"\nGenerated parameters saved to {config['path_to_save']}")


