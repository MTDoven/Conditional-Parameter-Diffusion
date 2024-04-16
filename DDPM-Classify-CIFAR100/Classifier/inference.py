import timm
import torch
from torch import nn
from torch.nn import functional as F


model = None
@torch.no_grad()
def inference(images, **config):
    global model
    if model is None:
        model = timm.create_model("timm/vit_base_patch16_224", pretrained=False)
        model.head = nn.Linear(model.head.in_features, 100)
        model.load_state_dict(torch.load("./Classifier/ViT-CIFAR100/pytorch_model.bin"))
        model = model.to(config["device"])
        model.eval()

    result = model(F.interpolate(images, size=(224, 224), mode="bilinear"))
    probabilities = torch.softmax(result, dim=1)
    _, top5_pred = torch.topk(probabilities, 5, dim=1)
    result = torch.argmax(result, dim=-1)

    # Calculate top-1 and top-5 accuracy
    top1_accuracy = torch.eq(top5_pred[:, 0], config['label']).sum().item() / config['batch_size']
    top5_accuracy = torch.eq(top5_pred, config['label']).sum().item() / config['batch_size']
    # Calculate mean prob
    mean_probability = torch.mean(probabilities, dim=0)
    mean_probability = mean_probability[config["label"]].item()
    return result, top1_accuracy, top5_accuracy, mean_probability


if __name__ == "__main__":
    inference(None)