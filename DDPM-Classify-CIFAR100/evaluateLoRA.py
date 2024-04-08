
import torch
from Diffusion.Diffusion import GaussianDiffusionSampler
from LoRA.Model import UNet
from Classifier.inference import inference


def sample(**config):
    device = torch.device(config["device"])

    # model setup
    unet = UNet(
        T=config["T"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        attn=config["attn"],
        num_res_blocks=config["num_res_blocks"],
        dropout=0.,)
    base_model_dict = torch.load(config["BaseDDPM_path"])
    lora_param_dict = torch.load(config["LoRADDPM_path"])
    unet.load_state_dict({**base_model_dict, **lora_param_dict})
    unet = unet.to(device)
    sampler = GaussianDiffusionSampler(
        model=unet,
        beta_1=config["beta_1"],
        beta_T=config["beta_T"],
        T=config["T"])
    sampler = sampler.to(device)

    param_number = 0
    for name, param in unet.named_parameters():
        if "lora" in name:
            # print(name, len(param.flatten()))
            param_number += len(param.flatten())
    # print(f"model load weight done. Lora param: {param_number}")

    # load model and evaluate
    unet.eval()
    with torch.no_grad():
        noisyImage = torch.randn(
            size=[config["batch_size"], 3, config["img_size"], config["img_size"]],
            device=device,)
        sampledImgs = sampler(noisyImage)
        if config["save_sampled_images_path"]:
            from PIL import Image
            import os.path
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            arrays = sampledImgs.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            for i, image in enumerate(arrays):
                im = Image.fromarray(image)
                im.save(os.path.join(config["save_sampled_images_path"], f"{str(i).zfill(6)}.jpg"))
        return sampledImgs


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:3",
        # path setting
        "BaseDDPM_path": "./CheckpointBaseDDPM/BaseDDPM.pt",
        "LoRADDPM_path": "CheckpointLoRAGen/0000.pt",
        "save_sampled_images_path": "./temp",
        # model structure
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "img_size": 32,
        # training setting
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "batch_size": 100,
        # variable setting
        "label": 0,
    }

    top1_accuracys, top5_accuracys, mean_probabilitys = [], [], []
    for i in range(100):
        config["LoRADDPM_path"] = config["LoRADDPM_path"].split("/")[0] + f"/class{str(i).zfill(2)}.pt"
        config["label"] = i
        images = sample(**config)
        result, top1_accuracy, top5_accuracy, mean_probability = inference(images, **config)
        print(f"class{i}_result:", result, "\n"
              "top1_accuracy:", top1_accuracy, "\n"
              "top5_accuracy:", top5_accuracy, "\n"
              "mean_probability:", mean_probability)
        top1_accuracys.append(top1_accuracy)
        top5_accuracys.append(top5_accuracy)
        mean_probabilitys.append(mean_probability)
    print("======================================================================")
    print("top1_accuracy:", sum(top1_accuracys) / len(top1_accuracys), "\n"
          "top5_accuracy:", sum(top5_accuracys) / len(top5_accuracys), "\n"
          "mean_probability:", sum(mean_probabilitys) / len(mean_probabilitys))