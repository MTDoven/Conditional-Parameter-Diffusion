
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch import Tensor
import numpy as np


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, condition):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, condition, t), noise, reduction='mean')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (extract(self.coeff1, t, x_t.shape) * x_t -
                extract(self.coeff2, t, x_t.shape) * eps)

    def p_mean_variance(self, x_t, t, condition):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, condition)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, condition):
        x_t = x_T
        for time_step in tqdm(reversed(range(self.T))):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, condition=condition)
            noise = torch.randn_like(x_t) if time_step > 0 else 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, d_model),
            Swish(),
            nn.Linear(d_model, d_model), )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class TransformerUNet(nn.Module):
    def __init__(self, d_latent, num_layers, num_class, T=1000, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.d_latent = d_latent
        self.seq_length = None
        self.num_parameters = None
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.decoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.time_emb = TimeEmbedding(T, d_latent)
        self.condi_emb = nn.Embedding(num_class, d_latent)

    def encode(self, input: Tensor):
        assert len(input.shape) == 2
        batchsize, length = input.shape
        to_cat_length = self.d_latent - length % self.d_latent
        zeros_to_cat = torch.zeros([batchsize, to_cat_length], device=input.device)
        input = torch.cat([input, zeros_to_cat], dim=1)
        input = input.view((batchsize, -1, self.d_latent))
        self.seq_length = input.shape[1]-1
        self.num_parameters = length
        # go ahead into encoder
        result = self.encoder(input)
        return result

    def decode(self, result):
        # go ahead into decoder
        result = self.decoder(result)
        result = torch.flatten(result, start_dim=1)
        result = result[:, :self.num_parameters]
        return result

    def forward(self, input, condition, t, **kwargs):
        result = self.encode(input)
        result += self.time_emb(t)[:, None, :] + self.condi_emb(condition)[:, None, :]
        output = self.decode(result)
        return output


class ODUnetEncoder(nn.Module):
    def __init__(self, in_dim_list, enc_channel_list, kernel_size):
        super(ODUnetEncoder, self).__init__()
        self.in_dim_list = in_dim_list
        self.enc_channel_list = enc_channel_list
        self.kernel_size = kernel_size
        encoder = nn.ModuleList()
        layer_num = len(in_dim_list)
        for i in range(layer_num):
            layer = self.build_layer(in_dim_list[i], enc_channel_list[i], enc_channel_list[i + 1], kernel_size)
            encoder.append(layer)
        self.encoder = encoder

    def build_layer(self, in_dim, in_channel, out_channel, kernel_size):
        layer = nn.Sequential(
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.LeakyReLU(), )
        return layer

    def forward(self, x, condition, time, **kwargs):
        encoder_output = []
        for i, layer in enumerate(self.encoder):
            x = layer(x) + condition[i] + time[i]
            encoder_output.append(x)
        return encoder_output

class ODUnetDecoder(nn.Module):
    def __init__(self, in_dim_list, dec_channel_list, kernel_size):
        super(ODUnetDecoder, self).__init__()
        self.in_dim_list = in_dim_list
        self.dec_channel_list = dec_channel_list
        self.kernel_size = kernel_size

        decoder = nn.ModuleList()
        layer_num = len(in_dim_list)
        for i in range(layer_num):
            layer = self.build_layer(in_dim_list[i], dec_channel_list[i], dec_channel_list[i + 1],
                                     kernel_size, last=(i == layer_num - 1))
            decoder.append(layer)
        self.decoder = decoder

    def build_layer(self, in_dim, in_channel, out_channel, kernel_size, last=False):
        layer = nn.Sequential(
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.LeakyReLU() if not last else nn.Identity(), )
        return layer

    def forward(self, x, condition, time, **kwargs):
        y = x[-1]
        for i, layer in enumerate(self.decoder):
            y = layer(y)
            if i == len(self.decoder) - 1:
                break
            y += x[len(x)-i-2] + condition[len(condition)-i-2] + time[len(time)-i-2]
        return y

class ODUNet(nn.Module):
    def __init__(self, d_latent, num_channels, T, num_class,
                 in_channel=1, fold_rate=1, kernel_size=5, **kwargs):
        super(ODUNet, self).__init__()

        enc_channel_list = num_channels.copy()
        dec_channel_list = list(reversed(num_channels.copy()))
        self.in_dim = d_latent

        enc_dim_list = [d_latent // (fold_rate ** i) for i in range(len(enc_channel_list))]
        dec_dim_list = [d_latent // (fold_rate ** (4-i)) for i in range(len(dec_channel_list))]
        enc_channel_list = [in_channel] + enc_channel_list
        dec_channel_list = dec_channel_list + [in_channel]
        self.encoder = ODUnetEncoder(enc_dim_list, enc_channel_list, kernel_size)
        self.decoder = ODUnetDecoder(dec_dim_list, dec_channel_list, kernel_size)
        self.time_encode, self.class_encode = nn.ModuleList(), nn.ModuleList()
        for channel in num_channels:
            self.time_encode.append(nn.Embedding(T, channel))
            self.class_encode.append(nn.Embedding(num_class, channel))

    def forward(self, input, condition, time, **kwargs):
        condition = [i(condition)[:, :, None] for i in self.class_encode]
        time = [i(time)[:, :, None] for i in self.time_encode]
        x = self.encode(input, condition, time)
        x = self.decode(x, condition, time)
        x = torch.tanh(x)
        return x

    def encode(self, input, condition, time, **kwargs):
        assert input.shape[1] == self.in_dim, f"{input.shape}, {self.in_dim}"
        self.input_shape = input.shape
        input = input[:, None, :]
        enc_output = self.encoder(input, condition, time, **kwargs)
        return enc_output

    def decode(self, x, condition, time, **kwargs):
        dec_output = self.decoder(x, condition, time, **kwargs)
        return dec_output.view(self.input_shape)

