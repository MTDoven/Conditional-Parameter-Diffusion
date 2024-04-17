import os
import torch
from Diffusion.Diffusion import GaussianDiffusionSampler
from LoRA.Model import UNet
from Classifier.inference import inference
from PIL import Image
from evaluateLoRA import sample


if __name__ == "__main__":
    config = {
        # device setting
        "device": "cuda:1",
        # path setting
        "BaseDDPM_path": "./CheckpointBaseDDPM/BaseDDPM.pt",
        "LoRADDPM_path": "./CheckpointLoRAGen-bs128/class00.pt",
        "save_sampled_images_path": "../../datasets/Generated/GeneratedCIFAR100/class00",
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
        "batch_size": 600,
        # variable setting
        "label": 0,
    }

    for i in range(100):
        config["LoRADDPM_path"] = \
                config["LoRADDPM_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}.pt"
        config["save_sampled_images_path"] = \
                config["save_sampled_images_path"].rsplit("/", 1)[0] + f"/class{str(i).zfill(2)}"
        config["label"] = i
        images = sample(**config)