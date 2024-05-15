import shutil
from functools import reduce
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random
import torch
import os
import re
import torch
from safetensors.torch import load_file, save_file
from torchvision import transforms
from PIL import Image
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import PIL
import shutil


class ClassIndex2ParamDataset(Dataset):
    def __init__(self, path_to_loras, padding=192, duplicate=100):
        # print(path_to_loras)
        self.padding = padding
        self.duplicate = duplicate
        root, _, files = next(os.walk(path_to_loras))
        self.files_path = [os.path.join(root, file) for file in files if "lora" in file]
        self.length = len(self.files_path)
        # save structure
        self.param_structure = []
        for name, param in torch.load(self.files_path[0], map_location="cpu").items():
            assert "lora" in name
            self.param_structure.append((name, param.shape))
        self.param_structure.sort(key=lambda x: x[0])
        self.files_path = self.files_path * self.duplicate

    def __len__(self):
        return self.length * self.duplicate

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
        this_param = torch.cat([torch.zeros(self.padding), this_param, torch.zeros(self.padding)], dim=0)
        return torch.tensor(label), this_param

    def save_param_dict(self, parameters, save_path):
        assert len(parameters.shape) == 1
        parameters = parameters[self.padding: -self.padding]
        param_dict_to_save = {}
        for name, shape in self.param_structure:
            length_to_cut = reduce(lambda x, y: x*y, shape)
            param = parameters[:length_to_cut]
            param_dict_to_save[name] = param.view(shape)
            parameters = parameters[length_to_cut:]
        torch.save(param_dict_to_save, save_path)


class Image2SafetensorsDataset(Dataset):
    def __init__(self, path_to_loras, path_to_images, image_size=256, padding=0, duplicate=100):
        self._eval = False
        self.padding = padding
        self.duplicate = duplicate
        self.path_to_images = path_to_images
        root, dirs, _ = next(os.walk(path_to_loras))
        self.files_path = [os.path.join(root, dir, "pytorch_lora_weights.safetensors")
                           for dir in dirs if "lora" in dir]
        self.length = len(self.files_path)
        self.param_structure = []
        for name, param in load_file(self.files_path[0], device='cpu').items():
            assert "lora" in name, "included parameters not marked as lora."
            self.param_structure.append((name, param.shape))
        self.param_structure.sort(key=lambda x: x[0])
        self.transfer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return self.length * self.duplicate

    def __getitem__(self, item):
        item = item % self.length
        file_path = self.files_path[item]
        # load image
        label = file_path.split("class")[1][:1]
        dir = None
        for dir in os.listdir(self.path_to_images):
            if label in dir: break
        image_name = random.choice(next(os.walk(os.path.join(self.path_to_images, dir)))[-1])
        try:
            image = Image.open(os.path.join(self.path_to_images, dir, image_name)).convert("RGB")
        except PIL.UnidentifiedImageError:
            return self[random.randint(0, self.length - 1)]
        image = self.transfer(image)
        # load param
        diction = load_file(file_path, device='cpu')
        this_param = []
        for name, shape in self.param_structure:
            param = diction[name]
            if "lora_A" in name:
                param = param * 0.2
            elif "lora_B" in name:
                param = param * 200.0
            else:  # error
                raise ValueError(f"lora not in {name}")
            assert param.shape == shape
            this_param.append(param.flatten())
        this_param = torch.cat(this_param, dim=0)
        this_param = torch.cat([torch.zeros(self.padding), this_param, torch.zeros(self.padding)], dim=0)
        if self._eval:
            return image, this_param, int(label), image_name[:-4]
        return image, this_param

    def save_param_dict(self, parameters, save_path, adapter_config_path):
        assert len(parameters.shape) == 1
        parameters = parameters[self.padding: len(parameters)-self.padding]
        param_dict_to_save = {}
        for name, shape in self.param_structure:
            length_to_cut = reduce(lambda x, y: x * y, shape)
            param = parameters[:length_to_cut]
            if "lora_A" in name:
                param = param * 5.0
            elif "lora_B" in name:
                param = param * 0.005
            else:  # error
                raise ValueError(f"lora not in {name}")
            if "unet." == name[:5]:
                name = name[5:]
            param_dict_to_save[name] = param.view(shape)
            parameters = parameters[length_to_cut:]
        os.makedirs(save_path, exist_ok=True)
        save_file(param_dict_to_save, os.path.join(save_path, "adapter_model.safetensors"))
        shutil.copyfile(adapter_config_path, os.path.join(save_path, "adapter_config.json"))

    def eval(self):
        self._eval = True
        return self


class ContiImage2SafetensorsDataset(Dataset):
    def __init__(self, path_to_loras, path_to_images, image_size=256, padding=0, duplicate=20):
        self._eval = False
        self.padding = padding
        self.duplicate = duplicate
        self.path_to_images = path_to_images
        root, dirs, _ = next(os.walk(path_to_loras))
        self.files_path = [os.path.join(root, dir, "adapter_model.safetensors") for dir in dirs]
        self.length = len(self.files_path)
        self.param_structure = []
        for name, param in load_file(self.files_path[0], device='cpu').items():
            assert "lora" in name, "included parameters not marked as lora."
            self.param_structure.append((name, param.shape))
        self.param_structure.sort(key=lambda x: x[0])
        self.transfer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return self.length * self.duplicate

    def __getitem__(self, item):
        item = item % self.length
        file_path = self.files_path[item]
        # load image
        label = re.search(r'class(\d+)', file_path).group(1)
        dir = None
        for dir in os.listdir(self.path_to_images):
            if label in dir: break
        image_name = random.choice(next(os.walk(os.path.join(self.path_to_images, dir)))[-1])
        try:
            image = Image.open(os.path.join(self.path_to_images, dir, image_name)).convert("RGB")
        except PIL.UnidentifiedImageError:
            return self[random.randint(0, self.length - 1)]
        image = self.transfer(image)
        #image = torch.empty(size=[3, 16, 16])
        #image_name = "None.jpg"
        # load param
        diction = load_file(file_path, device='cpu')
        this_param = []
        for name, shape in self.param_structure:
            param = diction[name]
            if "lora_A" in name:
                param = param * 0.2
            elif "lora_B" in name:
                param = param * 200.0
            else:  # error
                raise ValueError(f"lora not in {name}")
            assert param.shape == shape
            this_param.append(param.flatten())
        this_param = torch.cat(this_param, dim=0)
        this_param = torch.cat([torch.zeros(self.padding), this_param, torch.zeros(self.padding)], dim=0)
        if self._eval:
            return image, this_param, int(label), image_name[:-4]
        return image, this_param

    def save_param_dict(self, parameters, save_path, adapter_config_path):
        assert len(parameters.shape) == 1
        parameters = parameters[self.padding: len(parameters)-self.padding]
        param_dict_to_save = {}
        for name, shape in self.param_structure:
            length_to_cut = reduce(lambda x, y: x * y, shape)
            param = parameters[:length_to_cut]
            if "lora_A" in name:
                param = param * 5.0
            elif "lora_B" in name:
                param = param * 0.005
            else:  # error
                raise ValueError(f"lora not in {name}")
            if "unet." == name[:5]:
                name = name[5:]
            param_dict_to_save[name] = param.view(shape)
            parameters = parameters[length_to_cut:]
        os.makedirs(save_path, exist_ok=True)
        save_file(param_dict_to_save, os.path.join(save_path, "adapter_model.safetensors"))
        shutil.copyfile(adapter_config_path, os.path.join(save_path, "adapter_config.json"))

    def eval(self):
        self._eval = True
        return self