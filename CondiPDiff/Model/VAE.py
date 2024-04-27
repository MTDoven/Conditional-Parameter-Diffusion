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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]
        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]
        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]
        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss
        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False), )
    def forward(self, input):
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self, in_channels: int, embedding_dim: int, num_embeddings: int,
                 hidden_dims=None, beta: float = 0.25, img_size: int = 64, **kwargs) -> None:
        super(VQVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256]
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(), ))
            in_channels = h_dim
        modules.append(nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(), ))
        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Sequential(
                nn.Conv1d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(), ))
        self.encoder = nn.Sequential(*modules)

        # build vq_layer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # Build Decoder
        modules = []
        modules.append(nn.Sequential(
                nn.Conv1d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(), ))
        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(), ))
        modules.append(nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1), ))
        self.decoder = nn.Sequential(*modules)

    def encode(self, input, **kwargs):
        result = self.encoder(input)
        return [result]

    def decode(self, z, **kwargs):
        result = self.decoder(z)
        return result

    def forward(self, input, **kwargs):
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        vq_loss = args[2]
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss}

    def generate(self, x, **kwargs):
        return self.forward(x, **kwargs)[0]