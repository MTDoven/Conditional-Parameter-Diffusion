import os
import shutil

checkpoint_dir = '/data/personal/nus-wk/cpdiff/condipdiff/PixArt-StyleTrans-COCO/pixart-pokemon-model'

destination_folder = '/data/personal/nus-wk/cpdiff/condipdiff/PixArt-StyleTrans-COCO/lora_path'

i = 0
for folder in os.listdir(checkpoint_dir):
    if 'checkpoint' in folder:
        file_path = os.path.join(checkpoint_dir, folder, 'transformer_lora')

        if os.path.exists(file_path):
            new_name = f'lora_class9_group0_number{i}'
            i += 1
            dest_path = os.path.join(destination_folder, new_name)
            shutil.copytree(file_path, dest_path, dirs_exist_ok=True)
