from Model.VAE import VanillaVAE as VAE
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
        "device": "cuda:2",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "VAE_path": "./CheckpointVAE/VAE.pt",
        "path_to_loras": "../DDPM-Classify-CIFAR100/CheckpointLoRADDPM",
        "path_to_save": "../DDPM-Classify-CIFAR100/CheckpointLoRAGen",
        # vae structure
        "d_model": [32, 64, 128, 256, 512, 1024, 2048],
        "d_latent": 1024,
        "num_layers": 7,
        # training setting
        "batch_size": 4,
        # variable parameters
        "num_parameters": 54912,
    }

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                           d_latent=config["d_latent"],
                           num_layers=config["num_layers"],)
    model.load_state_dict(torch.load(config["VAE_path"]))
    model = model.to(device)
    dataset = config["dataset"](config["path_to_loras"])

    # evaluate
    model.eval()
    with torch.no_grad():
        # gen_parameters = model.sample(
        #     num_samples=config["batch_size"],
        #     current_device=config["device"],
        #     num_parameters=config["num_parameters"],)
        for i in range(100):
            item, param = dataset[i]
            gen_parameter = model.generate(
                x=param[None, :].to(device),
                num_parameters=config["num_parameters"],)
            param = gen_parameter.detach().cpu()[0]


        # for i, param in enumerate(gen_parameters):
            dataset.save_param_dict(
                save_path=os.path.join(config["path_to_save"], f"{str(i).zfill(4)}.pt"),
                parameters=param,)
    print(f"Generated parameters saved to {config['path_to_save']}")

    # latents = []
    # for i in range(100):
    #     item, param = dataset[i]
    #     latent = model.encode(param[None, :].to(device))[0]
    #     latents.append(latent[0].detach().cpu().numpy())
    #
    # from sklearn.decomposition import PCA
    # from matplotlib import pyplot as plt
    # pca = PCA(n_components=2)
    # X_reduced = pca.fit_transform(latents)
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    # plt.savefig("./result-kld0.05.png")
    # plt.show()


