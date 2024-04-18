import torch_fidelity
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

fid_list = []
for i in range(100):
    metrics = torch_fidelity.calculate_metrics(
        input1=f'../../datasets/Generated/GeneratedCIFAR100/class{str(i).zfill(2)}',
        input2=f'../../datasets/Generated/OriginalCIFAR100/class{str(i).zfill(2)}',
        fid=True
    )["frechet_inception_distance"]
    print(f"class{str(i).zfill(2)} generated fid:", metrics)
    fid_list.append(metrics)

print("========================================================================")
print("final result:", sum(fid_list) / len(fid_list))