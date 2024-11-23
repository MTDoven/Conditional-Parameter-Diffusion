import torch_fidelity
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


for j in ["Soup"]:
    fid_list = []

    for i in range(4):
        metrics = torch_fidelity.calculate_metrics(
            input1=f'../../datasets/FIDStyles/style{str(i).zfill(1)}',
            input2=f'../../datasets/Generated/{j}Styles/style{str(i).zfill(2)}',
            fid=True
        )["frechet_inception_distance"]
        print(f"{j} class{str(i).zfill(2)} generated fid:", metrics)
        fid_list.append(metrics)

    print("========================================================================")
    print(f"{j} average result:", sum(fid_list) / len(fid_list))