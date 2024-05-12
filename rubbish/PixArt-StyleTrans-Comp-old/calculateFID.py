import torch_fidelity
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

fid_list = []
for i in range(10):
    metrics = torch_fidelity.calculate_metrics(
        input1=f'../../datasets/Generated/GeneratedStyles/style{i}',
        input2=f'../../datasets/Styles/style{i}',
        fid=True
    )["frechet_inception_distance"]
    print(f"class{i} generated fid:", metrics)
    fid_list.append(metrics)

print("========================================================================")
print("final result:", sum(fid_list) / len(fid_list))