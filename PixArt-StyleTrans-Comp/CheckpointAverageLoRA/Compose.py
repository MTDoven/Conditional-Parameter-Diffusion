from safetensors.torch import load_file, save_file
import os, shutil, tqdm

num_classes = 10
num_sample = 10
path_to_save = "./GenAverage"
adapter_config_path = "../CheckpointStyleDataset/adapter_config.json"

for j in tqdm.tqdm(range(num_classes)):  # 10 classes
    new_diction = {}
    for i in range(num_sample):
        x = load_file(f"./Gen{str(i).zfill(1)}/class{str(j).zfill(2)}/adapter_model.safetensors", device="cpu")
        for key, value in x.items():
            if new_diction.get(key) is not None:
                new_diction[key] += value / num_sample
            else:  # first circle
                new_diction[key] = value / num_sample
    save_path = os.path.join(path_to_save, f"class{str(j).zfill(2)}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)
    save_file(new_diction, os.path.join(save_path, "adapter_model.safetensors"))
    shutil.copyfile(adapter_config_path, os.path.join(save_path, "adapter_config.json"))

