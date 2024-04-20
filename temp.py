# import torch
# x = torch.load("./DDPM-Classify-CIFAR10/CheckpointBaseDDPM/BaseDDPM.pt")
# x = x["net_model"]
# torch.save(x, "DDPM-Classify-CIFAR10/CheckpointBaseDDPM/BaseDDPM.pt")
#
#
# exit()
#
# import torch
# from safetensors.torch import load_file, save_file
#
# # x = torch.load("./DDPM-Classify-CIFAR100/CheckpointLoRAGen/class00.pt")
# # y = torch.load("../datasets/CIFAR10-LoRA-Dataset/lora_class0_number0.pt")
#
# for i in range(10):
#     for j in range(4):
#         for k in range(128):
#             try:
#                 x = load_file(f"../datasets/PixArt-LoRA-Dataset/lora_class{i}_group{j}_number{k}/adapter_model.safetensors")
#                 for name, param in x.items():
#                     if param.norm() > 1.2:
#                         print(param.norm(), f"lora_class{i}_group{j}_number{k}")
#             except FileNotFoundError as e:
#                 print(e)


# for name, param1 in x.items():
#     print(param1.flatten(), param1.norm())
#     break
#
# for name, param2 in y.items():
#     print(param2.flatten(), param2.norm())
#     break
#
# print((param1.cpu() - param2.cpu()).flatten().mean())
# print(param1.std())
#
#
#
# temp = []
# for name, param in y.items():
#     temp.append(param.cpu().flatten())
# temp = torch.cat(temp, dim=0)
#
# #temp = (temp-temp.mean()) / temp.std()
# #temp = torch.tanh(temp) + temp*0.05
#
# import matplotlib.pyplot as plt
# plt.hist(temp, bins=200, density=True, alpha=0.6, color='g')
# plt.savefig("./show.jpg")

import torch
from safetensors.torch import load_file

x = load_file("/home/wangkai/cpdiff/condipdiff/PixArt-StyleTrans-COCO/lora_result_9_0/adapter_model.safetensors", device="cpu")
for key, value in x.items():
    print(key, value[0])
    break

x = load_file("/home/wangkai/cpdiff/condipdiff/PixArt-StyleTrans-COCO/lora_result_9_0/transformer_lora/pytorch_lora_weights.safetensors", device="cpu")
for key, value in x.items():
    print(key, value[0])


