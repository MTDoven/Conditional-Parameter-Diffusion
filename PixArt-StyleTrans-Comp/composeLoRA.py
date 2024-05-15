from safetensors.torch import load_file, save_file
import os, shutil

checkpoint_train_lora = "./CheckpointTrainLoRA"
path_to_save = "./CheckpointSoupLoRA"
num_classes = 4
num_checkpoints = 64


loras = os.listdir(checkpoint_train_lora)
result = [{}, {}, {}, {}]

for lora in loras:
    checkpoint = load_file(os.path.join(checkpoint_train_lora, lora, "pytorch_lora_weights.safetensors"))
    index = int(lora.split("class")[1][:1])
    for name, param in checkpoint.items():
        if "unet." == name[:5]:
            name = name[5:]
        if result[index].get(name) is None:
            result[index][name] = param / num_checkpoints
        else:  # name have been in result
            result[index][name] += param / num_checkpoints

for i, diction in enumerate(result):
    dir_to_save = os.path.join(path_to_save, f"class{str(i).zfill(2)}")
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save, exist_ok=False)
    save_file(diction, os.path.join(dir_to_save, "adapter_model.safetensors"))
    shutil.copyfile("./CheckpointStyleDataset/adapter_config.json", os.path.join(dir_to_save, "adapter_config.json"))