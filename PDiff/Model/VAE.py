import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class VanillaVAE(BaseVAE):
    def __init__(self, d_latent, d_model=None, in_channels=1, **kwargs):
        super(VanillaVAE, self).__init__()
        self.num_parameters = None
        self.last_length = None
        self.latent_dim = d_latent
        self.depth = len(d_model)

        # Build Encoder
        modules = []
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                          nn.BatchNorm1d(h_dim), nn.LeakyReLU()), )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(d_model[-1], d_latent)
        self.fc_var = nn.Linear(d_model[-1], d_latent)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(d_latent, d_model[-1])
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                    nn.ConvTranspose1d(d_model[i], d_model[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm1d(d_model[i + 1]), nn.LeakyReLU()), )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose1d(d_model[-1], d_model[-1],
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm1d(d_model[-1]), nn.LeakyReLU(),
                            nn.Conv1d(d_model[-1], out_channels=in_channels, kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        # input.shape == [batch_size, num_parameters]
        input = input[:, None, :]
        self.num_parameters = input.shape[-1]
        result = self.encoder(input)
        self.last_length = result.shape[-1]
        result = result.max(dim=-1)[0]
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        if self.num_parameters is None:
            self.num_parameters = kwargs["num_parameters"]
            self.last_length = self.num_parameters // int(2 ** self.depth)
        result = self.decoder_input(z)
        result = result[:, :, None].repeat(1, 1, self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        assert self.num_parameters == result.shape[-1], f"{self.num_parameters}, {result.shape}"
        return result[:, 0, :]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return recons, input, mu, log_var

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['kld_weight']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'MSELoss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples, current_device, **kwargs) -> Tensor:
        if self.num_parameters is None:
            self.num_parameters = kwargs["num_parameters"]
            self.last_length = self.num_parameters // self.depth
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        if self.num_parameters is None:
            self.num_parameters = kwargs["num_parameters"]
            self.last_length = self.num_parameters // self.depth
        return self.forward(x)[0]


class TransformerVAE(BaseVAE):
    def __init__(self, d_model, d_latent, num_layers, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_parameters = None
        self.seq_length = None

        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers, norm=nn.LayerNorm(d_model))
        self.decoder = nn.TransformerEncoder(transformer_layer, num_layers, norm=nn.LayerNorm(d_model))

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

    def forward(self, input, **kwargs):
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

