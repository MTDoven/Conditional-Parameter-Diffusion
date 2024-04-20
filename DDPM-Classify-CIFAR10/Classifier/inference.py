from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


model = None
processor = None

@torch.no_grad()
def inference(images, **config):
    global model, processor
    if model is None:
        processor = ViTImageProcessor.from_pretrained('./Classifier/ViT-CIFAR10')
        model = ViTForImageClassification.from_pretrained('./Classifier/ViT-CIFAR10')
    inputs = processor(images=images, return_tensors="pt", do_rescale=False)
    outputs = model(**inputs).logits
    probabilities = torch.softmax(outputs, dim=1)
    _, top5_pred = torch.topk(probabilities, 5, dim=1)
    result = torch.argmax(outputs, dim=-1)

    # Calculate top-1 and top-5 accuracy
    top1_accuracy = torch.eq(top5_pred[:, 0], config['label']).sum().item() / config['batch_size']
    top5_accuracy = torch.eq(top5_pred, config['label']).sum().item() / config['batch_size']
    # Calculate mean prob
    mean_probability = torch.mean(probabilities, dim=0)
    mean_probability = mean_probability[config["label"]].item()
    return result, top1_accuracy, top5_accuracy, mean_probability


if __name__ == "__main__":
    inference(Image.open("/data/personal/nus-wk/cpdiff/condipdiff/DDPM-Classify-CIFAR10/temp/000000.jpg"))