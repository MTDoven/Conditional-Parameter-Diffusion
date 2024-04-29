import os
import shutil

input_dir = ("/home/wangkai/cpdiff/datasets/MultiStyles/imagenet-sketch/sketch")
output_dir = "./ls"
diction = {}

with open("ls.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    if len(line) < 3:
        continue
    first, prompt = line.split(",", 1)
    prompt = prompt.replace("\n", "")[1:]
    folder = first.split(" ")[1]
    diction[folder] = prompt

for folder in os.listdir(input_dir):
    prompt = diction[folder]
    for i, file in enumerate(os.listdir(os.path.join(input_dir, folder))):
        file_path = os.path.join(input_dir, folder, file)
        output_path = os.path.join(output_dir, f"{prompt.replace(',', '')} {i}.jpg")
        print(file_path, output_path)
        shutil.move(file_path, output_path)
        #exit()