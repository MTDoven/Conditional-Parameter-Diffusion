from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms
import torch


class OneClassDataset(Dataset):
    def __init__(self, root, img_size, label):
        dataset = CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Resize(config["img_size"], antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        image_list = []
        for image, this_label in dataset:
            if this_label == label:
                image_list.append(image)
        self.resize = transforms.Resize(img_size, antialias=True)
        self.image_list = image_list
        self.label = torch.tensor(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img = self.image_list[item]
        img = self.resize(img)
        return img, self.label

