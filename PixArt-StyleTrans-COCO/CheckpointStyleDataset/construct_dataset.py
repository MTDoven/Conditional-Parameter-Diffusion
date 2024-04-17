import os
import shutil

style_class = 2
param_group = 3


checkpoint_dir = (f'../lora_result_{style_class}_{param_group}')
destination_folder = '../../../datasets/PixArt-LoRA-Dataset'
classs = str(style_class)
group = str(param_group)

i = 0
for folder in os.listdir(checkpoint_dir):
    if 'checkpoint' in folder:
        file_path = os.path.join(checkpoint_dir, folder, 'transformer_lora')
        if os.path.exists(file_path):
            new_name = f'lora_class{classs}_group{group}_number{i}'
            i += 1
            dest_path = os.path.join(destination_folder, new_name)
            shutil.copytree(file_path, dest_path, dirs_exist_ok=True)
