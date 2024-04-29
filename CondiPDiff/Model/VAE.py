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
