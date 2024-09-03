from Model.DDPM import ODUNetTransfer as UNet
from Model.DDPM import GaussianDiffusionSampler
from Model.VAE import OneDimVAE as VAE
from Dataset import ContiImage2SafetensorsDataset
from Visualize import visualize_data
import torch



if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # paths setting
        "image_size": 256,
        "dataset": ContiImage2SafetensorsDataset,
        "UNet_path": "./CheckpointDDPM/UNet-Continue-05-01.pt",
        "VAE_path": "./CheckpointVAE/VAE-Continue-05-01.pt",
        "path_to_loras": "../PixArt-StyleTrans-Conti1/CheckpointOriginLoRA",
        "path_to_images": "../../datasets/ContiStyles",
        "path_to_save": "../PixArt-StyleTrans-Conti1/CheckpointGenLoRA",
        "adapter_config_path": "../PixArt-StyleTrans-Conti1/CheckpointStyleDataset/adapter_config.json",
        # ddpm structure
        "num_channels": [64, 128, 256, 512, 768, 1024, 1024, 32],
        "T": 1000,
        "num_class": 1000,
        "kernel_size": 3,
        "num_layers_diff": -1,
        "not_use_fc": False,
        "freeze_extractor": False,
        "simple_extractor": True,
        # model structure
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
        "batch_size": 100,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        # variable parameters
        "condition": 0
    }

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
    unet.load_state_dict(torch.load(config["UNet_path"]))
    unet = unet.to(device)
    vae = VAE(d_model=config["d_model"],
              d_latent=config["d_latent"],
              num_parameters=config["num_parameters"],
              last_length=config["last_length"],
              kernel_size=config["kernel_size_vae"],
              num_layers=config["num_layers"],
              use_elu_activator=config["use_elu_activator"],)
    diction = torch.load(config["VAE_path"], map_location="cpu")
    new_diction = {}
    for name, param in diction.items():
        if "_orig_mod" in name:
            new_diction[name.split(".", 1)[1]] = param
        else:  # not orig_mod
            new_diction[name] = param
    vae.load_state_dict(new_diction)
    vae = vae.to(device)
    sampler = GaussianDiffusionSampler(
        model=unet,
        beta_1=config["beta_1"],
        beta_T=config["beta_T"],
        T=config["T"])
    sampler = sampler.to(device)
    dataset = config["dataset"](path_to_loras=config["path_to_loras"],
                                path_to_images=config["path_to_images"],
                                image_size=config["image_size"],
                                padding=config["padding"],
                                duplicate=1).eval()
    dataset.files_path.sort()
    unet.eval()
    vae.eval()


################################# Code for Visualize ################################

    # prepare data
    inserting = 2
    condition = []
    params = []
    for index in range(0, len(dataset), inserting):
        image, param, item, prompt = dataset[index]
        if (item // 100) % 1 == 0:
            condition.append(image)
        if (item // 100) % 2 == 0 and item % 1 == 0:
            params.append(param)
        print("\r", index, item, prompt)
    condition = torch.stack(condition).to(device)
    params = torch.stack(params).to(device)

    # sample
    with torch.no_grad():
        origin = vae.encode(params)[0] * 0.01
        noise = torch.randn(size=(len(range(0, len(dataset), inserting)), config["d_latent"]), device=device)
        sampled = sampler(noise, condition.to(device))

    # draw
    print("origin:", len(origin), origin.max(), "    sampled:", len(sampled), sampled.max())
    input_data = torch.cat((origin, sampled), dim=0)
    input_label = torch.cat((torch.zeros(size=(len(origin),)), torch.ones(size=(len(sampled),))), dim=0)
    visualize_data(input_data.cpu(), input_label.cpu(), pca=True, tsne=False, save_path="./result.jpg")



