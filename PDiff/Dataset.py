DATA_PATH = "/path/to/test/dir"
from functools import reduce
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random
import torch
import os
import re


class ClassIndex2ParamDataset(Dataset):
    def __init__(self, path_to_loras=DATA_PATH):
        # print(path_to_loras)
        root, _, files = next(os.walk(path_to_loras))
        self.files_path = [os.path.join(root, file) for file in files if "lora" in file]
        self.length = len(self.files_path)
        # save structure
        self.param_structure = []
        for name, param in torch.load(self.files_path[0], map_location="cpu").items():
            assert "lora" in name
            self.param_structure.append((name, param.shape))
        self.param_structure.sort(key=lambda x: x[0])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        file_path = self.files_path[item]
        # load label
        label = int(re.search(r'class(\d+)', file_path).group(1))
        # load param
        diction = torch.load(file_path, map_location="cpu")
        this_param = []
        for name, shape in self.param_structure:
            param = diction[name]
            assert param.shape == shape
            this_param.append(param.flatten())
        this_param = torch.cat(this_param, dim=0)
        # this_param = self.transform(this_param)
        return torch.tensor(label), this_param

    def save_param_dict(self, parameters, save_path):
        assert len(parameters.shape) == 1
        param_dict_to_save = {}
        for name, shape in self.param_structure:
            length_to_cut = reduce(lambda x, y: x*y, shape)
            param = parameters[:length_to_cut]
            param_dict_to_save[name] = param.view(shape)
            parameters = parameters[length_to_cut:]
        torch.save(param_dict_to_save, save_path)


class _OneClassDataset(Dataset):
    def __init__(self, root, img_size, label):
        dataset = CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
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





