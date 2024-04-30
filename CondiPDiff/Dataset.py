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
    def __init__(self, path_to_loras, path_to_images, image_size=256, padding=176):
        self._eval = False
        self.padding = padding
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
        return self.length

    def __getitem__(self, item):
        file_path = self.files_path[item]
        # load image
        label = file_path.split("class")[1][:2]
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
            assert param.shape == shape
            this_param.append(param.flatten())
        this_param = torch.cat(this_param, dim=0)
        this_param = torch.cat([torch.zeros(self.padding), this_param, torch.zeros(self.padding)], dim=0)
        if self._eval:
            return image, this_param, int(label), image_name[:-4]
        return image, this_param

    def save_param_dict(self, parameters, save_path, adapter_config_path):
        assert len(parameters.shape) == 1
        parameters = parameters[self.padding: -self.padding]
        param_dict_to_save = {}
        for name, shape in self.param_structure:
            length_to_cut = reduce(lambda x, y: x * y, shape)
            param = parameters[:length_to_cut]
            param_dict_to_save[name[5:]] = param.view(shape)
            parameters = parameters[length_to_cut:]
        os.makedirs(save_path, exist_ok=True)
        save_file(param_dict_to_save, os.path.join(save_path, "adapter_model.safetensors"))
        shutil.copyfile(adapter_config_path, os.path.join(save_path, "adapter_config.json"))

    def eval(self):
        self._eval = True
        return self


class DoubleImage2SafetensorsDataset(Dataset):
    def __init__(self, path_to_loras, path_to_images, image_size=256, padding=176):
        self._eval = False
        self.padding = padding
        self.path_to_images = path_to_images
        self.param_structure = None
        # main
        root, dirs, _ = next(os.walk(path_to_loras))
        self.files_path = [os.path.join(root, dir, "pytorch_lora_weights.safetensors")
                           for dir in dirs if "lora" in dir]
        self.length = len(self.files_path)
        self.transfer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return self.length * self.length

    def __getitem__(self, item):
        return self._get_item(item // self.length, item % self.length)

    def _get_image(self, item):
        file_path = self.files_path[item]
        label = file_path.split("class")[1][:2]
        dir = None
        for dir in os.listdir(self.path_to_images):
            if label in dir: break
        image_name = random.choice(next(os.walk(os.path.join(self.path_to_images, dir)))[-1])
        try:
            image = Image.open(os.path.join(self.path_to_images, dir, image_name)).convert("RGB")
            image = self.transfer(image)
        except PIL.UnidentifiedImageError:
            return self._get_image(item)
        return image, label, image_name[:-4]

    def _get_item(self, item1, item2):
        image1, label1, prompt1 = self._get_image(item1)
        image2, label2, prompt2 = self._get_image(item2)
        if item1 != item2:
            image1_weight = torch.rand(1)
            image2_weight = 1 - image1_weight
            weight1 = torch.clamp(image1_weight*4, min=0, max=2)
            weight2 = torch.clamp(image2_weight*4, min=0, max=2)
        else:  # item1 == item2:
            image1_weight, image2_weight = torch.tensor([0.5]), torch.tensor([0.5])
            weight1, weight2 = torch.tensor([1.0]), torch.tensor([1.0])
        # to combine lora
        combined_lora = {}
        for i, ((name1, param1), (name2, param2)) in enumerate(zip(
                    load_file(self.files_path[item1], device="cpu").items(),
                    load_file(self.files_path[item2], device="cpu").items())):
            if param1.shape[0] == 1:
                param3 = torch.cat((param1, param2), dim=0)
            elif param1.shape[1] == 1:
                param3 = torch.cat((param1 * weight1, param2 * weight2), dim=1)
            else:  # not one dim lora
                raise ValueError("not one dim lora")
            combined_lora[name1[5:]] = param3
        # record structure
        if self.param_structure is None:
            self.param_structure = []
            for name, param in combined_lora.items():
                assert "lora" in name, "included parameters not marked as lora."
                self.param_structure.append((name, param.shape))
            self.param_structure.sort(key=lambda x: x[0])
        # load data
        this_param = []
        for name, shape in self.param_structure:
            param = combined_lora[name]
            assert param.shape == shape
            this_param.append(param.flatten())
        this_param = torch.cat(this_param, dim=0)
        this_param = torch.cat([torch.zeros(self.padding), this_param, torch.zeros(self.padding)], dim=0)
        # to return
        if self._eval:
            return image1, image2, image1_weight, image2_weight, this_param, \
                   (int(label1), int(label2), prompt1, prompt2)
        return image1, image2, image1_weight, image2_weight, this_param

    def save_param_dict(self, parameters, save_path, adapter_config_path):
        assert len(parameters.shape) == 1
        parameters = parameters[self.padding: -self.padding]
        param_dict_to_save = {}
        if self.param_structure is None: self.__getitem__(0)
        for name, shape in self.param_structure:
            length_to_cut = reduce(lambda x, y: x*y, shape)
            param = parameters[:length_to_cut]
            param_dict_to_save[name[5:]] = param.view(shape)  # [5:] is used to drop "unet." prefix
            parameters = parameters[length_to_cut:]
        os.makedirs(save_path, exist_ok=True)
        save_file(param_dict_to_save, os.path.join(save_path, "adapter_model.safetensors"))
        shutil.copyfile(adapter_config_path, os.path.join(save_path, "adapter_config.json"))

    def eval(self):
        self._eval = True
        return self
