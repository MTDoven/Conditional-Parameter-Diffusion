import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class TransformerVAE(nn.Module):
    def __init__(self, d_model, d_latent, num_layers, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_parameters = None
        self.seq_length = None

        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.decoder = nn.TransformerEncoder(transformer_layer, num_layers)

        self.fc_mu = nn.Linear(d_model, d_latent)
        self.fc_var = nn.Linear(d_model, d_latent)
        self.fc_decode = nn.Linear(d_latent, d_model)
        self.fc_out = nn.Sequential(nn.Tanh(), nn.Flatten(start_dim=1))

    def encode(self, input: Tensor):
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

    def decode(self, z: Tensor, **kwargs):
        if self.num_parameters is None:
            self.num_parameters = kwargs["num_parameters"]
            self.seq_length = self.num_parameters // self.d_model + 1
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

    def reparameterize(self, mu: Tensor, log_var: Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return output, input, mu, log_var

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['kld_weight']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'MSELoss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples, current_device, **kwargs):
        if self.num_parameters is None:
            self.num_parameters = kwargs["num_parameters"]
            self.seq_length = self.num_parameters // self.d_model + 1
        z = torch.randn(num_samples, self.d_latent)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        if self.num_parameters is None:
            self.num_parameters = kwargs["num_parameters"]
            self.seq_length = self.num_parameters // self.d_model + 1
        return self.forward(x)[0]

