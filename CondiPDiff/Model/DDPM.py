from tqdm.auto import tqdm
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        beta = torch.linspace(beta_1, beta_T, T).double()
        self.register_buffer('betas', beta)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, condition):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, condition, t), noise, reduction='none').sum() * 0.001
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

    def p_mean_variance(self, x_t, condition, t):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, condition, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, condition):
        x_t = x_T
        for time_step in tqdm(reversed(range(self.T))):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, condition=condition, t=t)
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

    def build_layer(self, in_dim, in_channel, out_channel, kernel_size, activator=nn.LeakyReLU):
        layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.BatchNorm1d(out_channel),
            activator(),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.BatchNorm1d(out_channel),
            activator(), )
        return layer

    def forward(self, x, condition, **kwargs):
        encoder_output = []
        for i, layer in enumerate(self.encoder):
            x = layer(x) + condition
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

    def build_layer(self, in_dim, in_channel, out_channel, kernel_size, last=False, activator=nn.ELU):
        layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.BatchNorm1d(out_channel),
            activator(),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1),
            nn.BatchNorm1d(out_channel) if not last else nn.Identity(),
            activator() if not last else nn.Identity(), )
        return layer

    def forward(self, x, condition, **kwargs):
        y = x[-1]
        for i, layer in enumerate(self.decoder):
            y = layer(y)
            if i == len(self.decoder) - 1:
                break
            y += x[len(x)-i-2] + condition
        return y

class MiddleLayers(nn.Module):
    def __init__(self, d_latent, last_channel):
        super().__init__()
        self.layer = nn.Linear(d_latent * last_channel, d_latent * last_channel)

    def forward(self, x0, time_emb, condi_emb, **kwargs):
        x1 = self.layer(torch.flatten(x0 + time_emb, start_dim=1)).view(x0.shape)
        return F.leaky_relu(x1) + time_emb + condi_emb


class ODUNetBase(nn.Module):
    def forward(self, input, condition, time, **kwargs):
        time_emb = self.time_encode(time)[:, :, None]
        condi_emb = self.class_encode(condition)[:, None, :]
        x = self.encode(input, condi_emb)
        x[-1] = self.middle(x[-1], time_emb=time_emb, condi_emb=condi_emb)
        x = self.decode(x, condi_emb)
        return x

    def encode(self, input, condition, **kwargs):
        assert input.shape[1] == self.in_dim, f"{input.shape}, {self.in_dim}"
        self.input_shape = input.shape
        input = input[:, None, :]
        enc_output = self.encoder(input, condition, **kwargs)
        return enc_output

    def decode(self, x, condition, **kwargs):
        dec_output = self.decoder(x, condition, **kwargs)
        return dec_output.view(self.input_shape)


class ODUNetClassify(ODUNetBase):
    def __init__(self, d_latent, num_channels, T, num_class,
                 in_channel=1, fold_rate=1, kernel_size=7, **kwargs):
        super(ODUNetClassify, self).__init__()

        enc_channel_list = num_channels.copy()
        dec_channel_list = list(reversed(num_channels.copy()))
        self.in_dim = d_latent

        enc_dim_list = [d_latent // (fold_rate ** i) for i in range(len(enc_channel_list))]
        dec_dim_list = [d_latent // (fold_rate ** (4-i)) for i in range(len(dec_channel_list))]
        enc_channel_list = [in_channel] + enc_channel_list
        dec_channel_list = dec_channel_list + [in_channel]
        self.encoder = ODUnetEncoder(enc_dim_list, enc_channel_list, kernel_size)
        self.decoder = ODUnetDecoder(dec_dim_list, dec_channel_list, kernel_size)
        self.middle = MiddleLayers(d_latent, num_channels[-1])
        self.time_encode = TimeEmbedding(T, num_channels[-1])
        self.class_encode = nn.Embedding(num_class, d_latent)


class ODUNetTransfer(ODUNetBase):
    def __init__(self, d_latent, num_channels, T, num_class,
                 in_channel=1, fold_rate=1, kernel_size=7, **kwargs):
        super(ODUNetTransfer, self).__init__()

        enc_channel_list = num_channels.copy()
        dec_channel_list = list(reversed(num_channels.copy()))
        self.in_dim = d_latent

        enc_dim_list = [d_latent // (fold_rate ** i) for i in range(len(enc_channel_list))]
        dec_dim_list = [d_latent // (fold_rate ** (4-i)) for i in range(len(dec_channel_list))]
        enc_channel_list = [in_channel] + enc_channel_list
        dec_channel_list = dec_channel_list + [in_channel]
        self.encoder = ODUnetEncoder(enc_dim_list, enc_channel_list, kernel_size)
        self.decoder = ODUnetDecoder(dec_dim_list, dec_channel_list, kernel_size)
        self.middle = MiddleLayers(d_latent, num_channels[-1])
        self.time_encode = TimeEmbedding(T, num_channels[-1])
        self.class_encode = nn.Sequential(
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
            nn.LeakyReLU(),
            nn.Linear(1000, d_latent*2),
            nn.LeakyReLU(),
            nn.Linear(d_latent*2, d_latent),
        )

    def forward(self, input, condition, time, **kwargs):
        time_emb = self.time_encode(time)[:, :, None]
        condi_emb = self.class_encode(condition)[:, None, :]
        x = self.encode(input, condi_emb)
        x[-1] = self.middle(x[-1], time_emb=time_emb, condi_emb=condi_emb)
        x = self.decode(x, condi_emb)
        return x
