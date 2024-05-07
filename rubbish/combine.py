from safetensors.torch import load_file, save_file
import torch
from tqdm import tqdm

x = "/home/wangkai/cpdiff/condipdiff/PixArt-StyleTrans-Comp/CheckpointTrainLoRA/lora_class01_group0_number0/adapter_model.safetensors"
y = "/home/wangkai/cpdiff/condipdiff/PixArt-StyleTrans-Comp/CheckpointTrainLoRA/lora_class06_group0_number0/adapter_model.safetensors"

rank_2 = {}
for i, ((name1, param1), (name2, param2)) in enumerate(zip(load_file(x).items(), load_file(y).items())):
    if param1.shape[0] == 1:
        param3 = torch.cat((param1, param2), dim=0)
    elif param1.shape[1] == 1:
        param3 = torch.cat((param1*2, param2*2), dim=1)
    else:
        raise ValueError
    rank_2[name1[5:]] = param3.to("cuda:5")

# rank_1 = {}
# for i, (name, param1) in enumerate(rank_2.items()):
#     if i % 2 == 1: continue
#     param2 = rank_2[name.replace("A", "B")]
#     if "base_model.model.pos_embed.proj" in name:
#         param1 = torch.permute(param1, (2, 3, 0, 1))
#         param2 = torch.permute(param2, (2, 3, 0, 1))
#         x = param2 @ param1
#         U, S, V = torch.svd(x)
#         param1_new = V[:, :, :1, :] * torch.sqrt(S[:, :, :1, None])
#         param2_new = U[:, :, :, :1] * torch.sqrt(S[:, :, None, :1])
#         param1_new = torch.permute(param1_new, (2, 3, 0, 1))
#         param2_new = torch.permute(param2_new, (2, 3, 0, 1))
#         param2_new = param2_new.mean(dim=[-1, -2], keepdim=True)
#     else:
#         x = param2 @ param1
#         U, S, V = torch.svd(x)
#         if max(U.shape) >= max(V.shape):
#             param1_new = V[:1, :] * torch.sqrt(S[:1, None])
#             param2_new = U[:, :1] * torch.sqrt(S[None, :1])
#         else:
#             param1_new = (V[:, :1] * torch.sqrt(S[None, :1])).T
#             param2_new = (U[:1, :] * torch.sqrt(S[:1, None])).T
#     print(param1.shape, param1_new.shape)
#     print(param2.shape, param2_new.shape)
#     rank_1[name] = param1_new.contiguous().cpu()
#     rank_1[name.replace("A", "B")] = param2_new.contiguous().cpu()
#
#     print(param2_new @ param1_new)
#     print(x)
#     exit()

save_file(rank_2, "./PixArt-StyleTrans-Comp/CheckpointGenLoRA/ls/adapter_model.safetensors")



