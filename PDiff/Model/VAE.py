import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod


class BaseVAE(nn.Module):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def encode(self, input, **kwargs):
        pass

    @abstractmethod
    def decode(self, z, **kwargs):
        pass

    def reparameterize(self, mu, log_var, **kwargs):
        if kwargs.get("not_use_var"):
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)
        recons = self.decode(z)
        return recons, input, mu, log_var

    def loss_function(self, recons, input, mu, log_var, **kwargs):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        recons_norm = recons.norm(dim=1)
        input_norm = input.norm(dim=1)
        norm_loss = F.mse_loss(recons_norm, input_norm).mean()
        loss = recons_loss + kwargs['kld_weight'] * kld_loss + kwargs["norm_weight"] * norm_loss
        return {'loss': loss, 'MSELoss': recons_loss.detach(), 'KLD': kld_loss.detach(),
                "recons_norm": recons_norm.mean().detach(),
                "input_norm": input_norm.mean().detach(),
                "norm_loss": norm_loss}

    def sample(self, num_samples, current_device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x, **kwargs)[0]


class FullConnectVAE(BaseVAE):
    def __init__(self, d_latent, d_model, **kwargs):
        super(FullConnectVAE, self).__init__()
        self.num_parameters = d_model
        self.latent_dim = d_latent

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.LeakyReLU(),
            nn.Linear(d_latent, d_latent),
        )
        self.to_mean = nn.Linear(d_latent, d_latent)
        self.to_var = nn.Linear(d_latent, d_latent)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.LeakyReLU(),
            nn.Linear(d_latent, d_model),
        )

    def encode(self, input, **kwargs):
        input = self.encoder(input)
        mean = self.to_mean(input)
        var = self.to_var(input)
        return mean, var

    def decode(self, z, **kwargs):
        return self.decoder(z)


class OneDimVAE(BaseVAE):
    def __init__(self, d_model, d_latent, kernel_size=5, **kwargs):
        super(OneDimVAE, self).__init__()
        use_elu_activator = ("use_elu_activator" in kwargs) and (kwargs.get("use_elu_activator") is True)

        self.d_model = d_model.copy()
        self.d_latent = d_latent
        self.num_parameters = kwargs["num_parameters"] if "num_parameters" in kwargs else None
        self.last_length = kwargs["last_length"] if "last_length" in kwargs else None

        # Build Encoder
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_dim, h_dim, kernel_size, 2, kernel_size//2),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Sequential(
            nn.Linear(self.last_length * d_model[-1], d_latent),
            nn.LeakyReLU())
        self.fc_mu = nn.Linear(d_latent, d_latent)
        self.fc_var = nn.Linear(d_latent, d_latent)

        # Build Decoder
        modules = []
        self.to_decode = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.LeakyReLU() if not use_elu_activator else nn.ELU(),
            nn.Linear(d_latent, self.last_length * d_model[-1]))
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(d_model[i], d_model[i+1], kernel_size, 2, kernel_size//2, output_padding=1),
                nn.BatchNorm1d(d_model[i + 1]),
                nn.LeakyReLU() if not use_elu_activator else nn.ELU(),))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model[-1], d_model[-1], kernel_size, 2, kernel_size//2, output_padding=1),
            nn.BatchNorm1d(d_model[-1]),
            nn.LeakyReLU() if not use_elu_activator else nn.ELU(),
            nn.Conv1d(d_model[-1], 1, kernel_size, 1, kernel_size//2),
            nn.Tanh() if not use_elu_activator else nn.Identity())

    def encode(self, input, **kwargs):
        # input.shape == [batch_size, num_parameters]
        input = input[:, None, :]
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[-1], self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        assert self.num_parameters == result.shape[-1], \
            f"{self.num_parameters}, {result.shape}"
        assert result.shape[1] == 1, f"{result.shape}"
        return result[:, 0, :]


class TwoDimVAE(BaseVAE):
    def __init__(self, d_model, d_latent, **kwargs):
        super(TwoDimVAE, self).__init__()
        self.d_model = d_model.copy()
        self.d_latent = d_latent
        self.num_parameters = kwargs["num_parameters"] if "num_parameters" in kwargs else None
        self.last_length = kwargs["last_length"] if "last_length" in kwargs else None

        # Build Encoder
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv2d(in_dim, h_dim, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Sequential(
            nn.Linear(self.last_length[0] * self.last_length[1] * d_model[-1], d_latent),
            nn.LeakyReLU())
        self.fc_mu = nn.Linear(d_latent, d_latent)
        self.fc_var = nn.Linear(d_latent, d_latent)

        # Build Decoder
        modules = []
        self.to_decode = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.LeakyReLU(),
            nn.Linear(d_latent, self.last_length[0] * self.last_length[1] * d_model[-1]))
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(d_model[i], d_model[i+1], kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.BatchNorm2d(d_model[i + 1]),
                nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(d_model[-1], d_model[-1], kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(d_model[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(d_model[-1], 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh())

    def encode(self, input, **kwargs):
        # input.shape == [batch_size, num_parameters]
        input = input[:, None, :, :]
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[-1], self.last_length[0], self.last_length[1])
        result = self.decoder(result)
        result = self.final_layer(result)
        assert self.num_parameters == result.shape[-1] * result.shape[-2], \
            f"{self.num_parameters}, {result.shape}"
        assert result.shape[1] == 1, f"{result.shape}"
        return result[:, 0, :, :]


class TransformerVAE(BaseVAE):
    def __init__(self, d_model, d_latent, num_layers, nhead=8, dim_feedforward=2048, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_parameters = kwargs["num_parameters"] if "num_parameters" in kwargs else None
        self.seq_length = kwargs["seq_length"] if "seq_length" in kwargs else None

        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers, norm=nn.LayerNorm(d_model))
        self.decoder = nn.TransformerEncoder(transformer_layer, num_layers, norm=nn.LayerNorm(d_model))

        self.fc_mu = nn.Linear(d_model, d_latent)
        self.fc_var = nn.Linear(d_model, d_latent)
        self.fc_decode = nn.Linear(d_latent, d_model)
        self.fc_out = nn.Sequential(nn.Flatten(start_dim=1), nn.Tanh())

    def encode(self, input, **kwargs):
        assert len(input.shape) == 2
        batchsize, length = input.shape
        to_cat_length = 2 * self.d_model - length % self.d_model
        zeros_to_cat = torch.zeros([batchsize, to_cat_length], device=input.device)
        input = torch.cat([input, zeros_to_cat], dim=1)
        input = input.view((batchsize, -1, self.d_model))
        self.seq_length = input.shape[1]-1
        self.num_parameters = length

        # go ahead into encoder
        result = self.encoder(input)
        result = result[:, -1, :]
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        assert len(z.shape) == 2
        batchsize, d_latent = z.shape
        result = self.fc_decode(z)
        result = result.view(batchsize, 1, self.d_model)
        zeros_to_cat = torch.zeros([batchsize, self.seq_length, self.d_model], device=result.device)
        result = torch.cat([result, zeros_to_cat], dim=1)

        # go ahead into decoder
        result = self.decoder(result)
        result = result[:, 1:, :]
        result = self.fc_out(result)
        result = result[:, :self.num_parameters]
        return result

