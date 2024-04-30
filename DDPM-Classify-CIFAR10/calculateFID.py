import torch_fidelity
import os
os.environ["CUDA_VISIBLE_DEVICE"] = "5"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

fid_list = []
for i in range(10):
    metrics = torch_fidelity.calculate_metrics(
        input1=f'../../datasets/Generated/OriginCIFAR10/class{str(i).zfill(2)}',
        input2=f'../../datasets/Generated/BaseCIFAR10/class{str(i).zfill(2)}',
        fid=True
    )["frechet_inception_distance"]
    print(f"class{str(i).zfill(2)} generated fid:", metrics)
    fid_list.append(metrics)

print("========================================================================")
print("final result:", sum(fid_list) / len(fid_list))