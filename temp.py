import torch

x = torch.load("/data/personal/nus-wk/condipdiff/DDPM-LoRA-Dataset/lora_class0_number0.pt")
# y = torch.load("/data/personal/nus-wk/condipdiff/DDPM-LoRA-Dataset/lora_class0_number200.pt")
y = torch.load("/home/nus-wk/cpdiff/condipdiff/DDPM-Classify-CIFAR100/CheckpointLoRAGen/0000.pt")

for name, param1 in x.items():
    print(param1.flatten(), param1.norm())
    break

for name, param2 in y.items():
    print(param2.flatten(), param2.norm())
    break

print((param1.cpu() - param2.cpu()).flatten())
print((param1.cpu() - param2.cpu()).flatten().mean())



temp = []
for name, param in y.items():
    temp.append(param.cpu().flatten())
temp = torch.cat(temp, dim=0)

#temp = (temp-temp.mean()) / temp.std()
#temp = torch.tanh(temp) + temp*0.05

import matplotlib.pyplot as plt
plt.hist(temp, bins=200, density=True, alpha=0.6, color='g')
plt.savefig("./show.jpg")


