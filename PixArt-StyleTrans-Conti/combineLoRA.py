from safetensors.torch import load_file, save_file
import shutil
import torch
import os

def transfer_weight(w1):
    w2 = 1 - w1
    w1 = (-0.88 * w1 * w1 + 2.44 * w1) * 2.28 / 1.56
    w2 = (-0.88 * w2 * w2 + 2.44 * w2) * 2.12 / 1.56
    return w1 ** 0.5, w2 ** 0.5

def combine_and_save(lora1_path, lora2_path, out_path, weights: tuple[float, float]):
    lora1_weight, lora2_weight = weights
    rank_2 = {}
    for (name1, param1), (name2, param2) in \
            zip(load_file(lora1_path, device="cpu").items(), load_file(lora2_path, device="cpu").items()):
        if param1.shape[0] == 1:
            param3 = torch.cat((param1 * lora1_weight, param2 * lora2_weight), dim=0)
        elif param1.shape[1] == 1:
            param3 = torch.cat((param1 * lora1_weight, param2 * lora2_weight), dim=1)
        else:  # not dim 1
            raise ValueError
        if name1[:5] == "unet.":
            name1 = name1[5:]
        rank_2[name1] = param3
    if out_path is None:
        return rank_2
    else:  # path exists
        save_file(rank_2, out_path)
        return rank_2


for i in range(1000):
    w1 = i / (1000 - 1)
    save_path = f"./CheckpointOriginLoRA/class{str(i).zfill(3)}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)
    combine_and_save(lora1_path="./CheckpointTrainLoRA/lora_class00_group0_number4/pytorch_lora_weights.safetensors",
                     lora2_path="./CheckpointTrainLoRA/lora_class01_group0_number4/pytorch_lora_weights.safetensors",
                     out_path=os.path.join(save_path, "adapter_model.safetensors"),
                     weights=transfer_weight(w1=w1), )
    shutil.copyfile("./CheckpointStyleDataset/adapter_config.json",
                    os.path.join(save_path, "adapter_config.json"))
