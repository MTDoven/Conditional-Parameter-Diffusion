import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import os, random
from PIL import Image
from io import BytesIO
import requests
import pandas as pd


imsize = 512
device = "cuda:7"


transform = transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(imsize), transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

detransform = transforms.ToPILImage()

def image_save(tensor, save_name="./output.jpg"):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = detransform(image)
    image.save(save_name, quality=90)

class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]

class Normalization(nn.Module):
    def __init__(self, mean=cnn_normalization_mean, std=cnn_normalization_std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean.to(img.device)) / self.std.to(img.device)

def get_style_model_and_losses(cnn, style_img, content_img, content_layers=None, style_layers=None):
    # normalization module
    if style_layers is None:
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    if content_layers is None:
        content_layers = ['conv_4']
    normalization = Normalization()
    # losses
    content_losses = []
    style_losses = []
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:  # Unrecognized layer
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            return style_score + content_score
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

def download_image(image_url):
    response = requests.get(image_url)
    image_data = response.content
    image_file = BytesIO(image_data)
    return image_file


cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
style_image_path = r"C:\Users\t1526\Desktop\style.jpg"
save_dir = r"C:\Users\t1526\Desktop\abs1"
csv_file_path = 'train.csv'
df = pd.read_csv(csv_file_path)
length = len(df)


i = 0
style_img = image_loader(style_image_path).to(device)
for index, row in df.iterrows():
    prompt = row['caption']
    image_url = row['image']
    content_image_path = download_image(image_url)
    content_img = image_loader(content_image_path).to(device)
    input_img = content_img.clone()
    output = run_style_transfer(cnn=cnn,
                                content_img=content_img.clone(),
                                style_img=style_img.clone(),
                                input_img=content_img.clone(),
                                num_steps=300,
                                style_weight=1e6,
                                content_weight=1)
    save_name = os.path.join(save_dir, f"{prompt}.jpg")
    image_save(output, save_name)

    print(f"{index}/{length}: Style transfer completed for {image_url} and saved as {prompt}.jpg")


print("Batch processing of files for style transfer is complete.")