import os
import shutil


checkpoint_dir = ('./lora_result_2_3')
destination_folder = '../../datasets/PixArt-LoRA-Dataset'
classs = "2"
group = "3"


i = 0
for folder in os.listdir(checkpoint_dir):
    if 'checkpoint' in folder:
        file_path = os.path.join(checkpoint_dir, folder, 'transformer_lora')
        if os.path.exists(file_path):
            new_name = f'lora_class{classs}_group{group}_number{i}'
            i += 1
            dest_path = os.path.join(destination_folder, new_name)
            shutil.copytree(file_path, dest_path, dirs_exist_ok=True)
