import os
import torch
from Diffusion.Diffusion import GaussianDiffusionSampler
from Classifier.inference import inference
from PIL import Image


def sample(**config):
    device = torch.device(config["device"])

    # model setup
    if config.get("LoRADDPM_path"):
        from LoRA.Model import UNet
        unet = UNet(
            T=config["T"],
            ch=config["channel"],
            ch_mult=config["channel_mult"],
            attn=config["attn"],
            num_res_blocks=config["num_res_blocks"],
            dropout=0.,)
        base_model_dict = torch.load(config["BaseDDPM_path"], map_location="cpu")
        lora_param_dict = torch.load(config["LoRADDPM_path"], map_location="cpu")
    else:  # not use lora
        from Diffusion.Model import UNet
        unet = UNet(
            T=config["T"],
            ch=config["channel"],
            ch_mult=config["channel_mult"],
            attn=config["attn"],
            num_res_blocks=config["num_res_blocks"],
            dropout=0.,)
        base_model_dict = torch.load(config["BaseDDPM_path"], map_location="cpu")
        lora_param_dict = {}

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
            if not os.path.exists(config["save_sampled_images_path"]):
                os.makedirs(config["save_sampled_images_path"], exist_ok=False)
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            arrays = sampledImgs.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            for i, image in enumerate(arrays):
                im = Image.fromarray(image)
                if config.get("batch_index") is None:
                    im.save(os.path.join(config["save_sampled_images_path"], f"{str(i).zfill(6)}.jpg"))
                else:  # config exists batch index
                    im.save(os.path.join(config["save_sampled_images_path"],
                                         f"{str(config.get('batch_index')).zfill(2)}{str(i).zfill(4)}.jpg"))
        return sampledImgs


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:7",
        # path setting
        "BaseDDPM_path": "./CheckpointBaseDDPM/BaseDDPM.pt",
        "LoRADDPM_path": "./CheckpointGenLoRA/class00.pt",
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
        "batch_size": 500,
        # variable setting
        "label": 0,
    }

    top1_accuracys, top5_accuracys, mean_probabilitys = [], [], []
    for i in range(10):
        config["LoRADDPM_path"] = config["LoRADDPM_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}.pt"
        config["label"] = i
        images = sample(**config)
        if len(images) > 200:
            assert len(images) % 100 == 0
            this_results, this_top1_accuracys, this_top5_accuracys, this_mean_probabilitys = [], [], [], []
            for i in range(0, len(images), 100):
                result, top1_accuracy, top5_accuracy, mean_probability = inference(images[i: i+100], **config)
                this_results.append(result)
                this_top1_accuracys.append(top1_accuracy * config["batch_size"])
                this_top5_accuracys.append(top5_accuracy * config["batch_size"])
                this_mean_probabilitys.append(mean_probability)
            result = torch.cat(this_results, dim=0)
            top1_accuracy = sum(this_top1_accuracys) / len(this_top1_accuracys)
            top5_accuracy = sum(this_top5_accuracys) / len(this_top5_accuracys)
            mean_probability = sum(this_mean_probabilitys) / len(this_mean_probabilitys)
        else:
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