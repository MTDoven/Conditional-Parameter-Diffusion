from Model.VAE import OneDimVAE as VAE
from Dataset import ClassIndex2ParamDataset
import os.path
import torch


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # paths setting
        "dataset": ClassIndex2ParamDataset,
        "VAE_path": "./CheckpointVAE/VAE-Classify.pt",
        "path_to_loras": "../DDPM-Classify-CIFAR10/CheckpointTrainLoRA",
        "path_to_save": "../DDPM-Classify-CIFAR10/CheckpointGenLoRA",
        # vae structure
        "d_model": [32, 64, 96, 128, 192, 256, 384, 512, 64],
        "d_latent": 64,
        "kernel_size": 7,
        "num_parameters": 54912+192*2,
        "half_padding": 192,
        "last_length": 108,
        "not_use_var": False,
        "use_elu_activator": True,
    }

    device = config["device"]
    model = VAE(d_model=config["d_model"],
                d_latent=config["d_latent"],
                kernel_size=config["kernel_size"],
                num_parameters=config["num_parameters"],
                last_length=config["last_length"],
                use_elu_activator=config["use_elu_activator"],)
    model.load_state_dict(torch.load(config["VAE_path"]))
    model = model.to(device)
    dataset = config["dataset"](config["path_to_loras"])

    # evaluate
    model.eval()
    for i in range(10):
        # load data
        for index in range(len(dataset)):
            item, param = dataset[index]
            if item == i:
                break
        print("\r", item, end="")

        # inference
        with torch.no_grad():
            gen_parameter = model.generate(
                x=param[None, :].to(device),
                num_parameters=config["num_parameters"],
                not_use_var=config["not_use_var"], )
            param = gen_parameter.detach().cpu()[0]

        # save
        dataset.save_param_dict(
            save_path=os.path.join(config["path_to_save"], f"class{str(i).zfill(2)}.pt"),
            parameters=param,)
    print(f"Generated parameters saved to {config['path_to_save']}")


