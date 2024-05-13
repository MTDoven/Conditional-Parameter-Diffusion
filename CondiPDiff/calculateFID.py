import torch_fidelity
import os
os.environ["CUDA_VISIBLE_DEVICE"] = "5"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

fid_list = []
for i in range(1):
    metrics = torch_fidelity.calculate_metrics(
        input1=f'../../datasets/FIDStyles/style0',
        input2=f'/home/wangkai/cpdiff/datasets/Generated/GenStyles/class900',
        fid=True
    )["frechet_inception_distance"]
    print(f"class{str(i).zfill(2)} generated fid:", metrics)
    fid_list.append(metrics)

print("========================================================================")
print("final result:", sum(fid_list) / len(fid_list))