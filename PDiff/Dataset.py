DATA_PATH = "/path/to/test/dir"
from functools import reduce
from torch.utils.data import Dataset
import torch
import os


class ClassIndex2ParamDataset(Dataset):
    def __init__(self, path_to_loras=DATA_PATH):
        root, _, files = next(os.walk(path_to_loras))
        files_path = [os.path.join(root, file) for file in files if "lora" in file]
        # save structure
        self.param_structure = []
        for name, param in torch.load(files_path[0]).items():
            assert "lora" in name
            self.param_structure.append((name, param.shape))
        self.param_structure.sort(key=lambda x:x[0])
        # load lora parameters
        self.lora_parameters = {}
        for file in files_path:
            this_param = []
            diction = torch.load(file)
            for name, shape in self.param_structure:
                param = diction[name]
                assert param.shape == shape
                this_param.append(param.cpu().flatten())
            this_param = torch.cat(this_param, dim=0)
            assert len(this_param) == 54912
            condition = int(file.split("_")[-1].split(".")[0])
            self.lora_parameters[condition] = this_param

    def __len__(self):
        return len(self.lora_parameters)

    def __getitem__(self, item):
        return torch.tensor(item), self.lora_parameters[item]

    def save_param_dict(self, parameters, save_path):
        assert len(parameters.shape) == 1 and len(parameters) == 54912
        param_dict_to_save = {}
        for name, shape in self.param_structure:
            length_to_cut = reduce(lambda x, y: x*y, shape)
            param = parameters[:length_to_cut].copy()
            param_dict_to_save[name] = param.view(shape)
            parameters = parameters[length_to_cut:]
        torch.save(param_dict_to_save, save_path)


