from Model.VAE import OneDimVAE as VAE
from Dataset import ContiImage2SafetensorsDataset
import os.path
import torch


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:4",
        # paths setting
        "image_size": 256,
        "dataset": ContiImage2SafetensorsDataset,
        "VAE_path": "./CheckpointVAE/VAE-Continue-05.pt",
        "path_to_loras": "../PixArt-StyleTrans-Conti/CheckpointOriginLoRA",
        "path_to_images": "../PixArt-StyleTrans-Conti/CheckpointStyleDataset/evaluateStyles",
        "path_to_save": "../PixArt-StyleTrans-Conti/CheckpointGenLoRA",
        "adapter_config_path": "../PixArt-StyleTrans-Conti/CheckpointStyleDataset/adapter_config.json",
        # vae structure
        "d_model": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 64],
        "d_latent": 256,
        "num_parameters": 516096,
        "padding": 0,
        "last_length": 504,
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
    diction = torch.load(config["VAE_path"], map_location="cpu")
    if "_orig_mod" in next(iter(diction.items()))[0]:
        new_diction = {}
        for name, param in diction.items():
            new_diction[name.split(".", 1)[1]] = param
        diction = new_diction
    model.load_state_dict(diction)
    model = model.to(device)
    dataset = config["dataset"](path_to_loras=config["path_to_loras"],
                                path_to_images=config["path_to_images"],
                                image_size=config["image_size"],
                                padding=config["padding"]).eval()

    # evaluate
    model.eval()
    for i in range(50, 1000, 100):
        # load data
        for index in range(len(dataset)):
            image, param, item, prompt = dataset[index]
            if item == i:
                print("\rIndex:", item)
                break

        # inference
        with torch.no_grad():
            gen_parameter = model.generate(
                x=param[None, :].to(device),
                not_use_var=config["not_use_var"],)
            recons = gen_parameter.detach().cpu()[0]
            loss = model.loss_function(recons[None], param[None], None, None, not_use_var=True)

        # save
        print("loss:", loss["MSELoss"])
        print("param:", param.flatten()[2000:2020])
        print("recons:", recons.flatten()[2000:2020])
        print()
        dataset.save_param_dict(
            save_path=os.path.join(config["path_to_save"], f"class{str(i).zfill(3)}"),
            parameters=recons,
            adapter_config_path=config["adapter_config_path"],)

    print(f"\nGenerated parameters saved to {config['path_to_save']}")


