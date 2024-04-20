import os.path
from LoRA.Dataset import OneClassDataset
import torch
from PIL import Image

save_to = "../../datasets/Generated/OriginalCIFAR10"


for i in range(10):
    dataset = OneClassDataset(root="../../datasets/CIFAR10",
                              img_size=32,
                              label=i,
                              train_set=True)
    for index, (img, label) in enumerate(dataset):
        save_path = os.path.join(save_to, f"class{str(i).zfill(2)}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=False)
        img = img * 0.5 + 0.5  # [0 ~ 1]
        arrays = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(arrays)
        im.save(os.path.join(save_path, f"{str(index).zfill(6)}.jpg"))
    assert index == 4999, "last index is "+str(index)

    dataset = OneClassDataset(root="../../datasets/CIFAR10",
                              img_size=32,
                              label=i,
                              train_set=False)
    for index, (img, label) in enumerate(dataset):
        save_path = os.path.join(save_to, f"class{str(i).zfill(2)}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=False)
        img = img * 0.5 + 0.5  # [0 ~ 1]
        arrays = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(arrays)
        im.save(os.path.join(save_path, f"{str(index+500).zfill(6)}.jpg"))

