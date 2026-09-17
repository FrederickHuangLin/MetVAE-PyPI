import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Callable, Union

def _get_activation(name: Optional[Union[str, Callable[[], nn.Module]]]) -> nn.Module:
    if name is None or (isinstance(name, str) and name.lower() in ("none", "")):
        return nn.Identity()
    if callable(name):
        return name()
    name = name.lower()
    if name == "relu":  return nn.ReLU()
    if name == "tanh":  return nn.Tanh()
    if name == "gelu":  return nn.GELU()
    if name == "silu":  return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")

def _mlp(in_dim: int, hidden_dims: Optional[List[int]], out_dim: int,
         act: Optional[Union[str, Callable[[], nn.Module]]] = "relu",
         last_activation: Optional[Union[str, Callable[[], nn.Module]]] = None,
         dtype: Optional[torch.dtype] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    dims = [in_dim] + (hidden_dims or [])
    activation = _get_activation(act)
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b, dtype=dtype), activation]
    layers.append(nn.Linear(dims[-1], out_dim, dtype=dtype))
    if last_activation is not None:
        layers.append(_get_activation(last_activation))
    return nn.Sequential(*layers)

class VAE(nn.Module):
    """
    Variational autoencoder with a linear decoder and optional MLP encoders.

    The latent prior is standard normal and the latent posterior is sampled with the
    reparameterization trick.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    latent_dim : int
        Size of the latent representation.
    hidden_dims : list of int or None, default=None
        Hidden layer sizes of the two encoder networks. If None or empty, the encoders
        are single linear layers. The decoder is a single linear layer in all cases.
    activation : str or callable or None, default="relu"
        Nonlinearity between encoder hidden layers. One of "relu", "tanh", "gelu",
        "silu", None for identity, or a zero-argument callable returning an nn.Module.
    dtype : torch.dtype, default=torch.float64
        Data type of all parameters and layers.

    Attributes
    ----------
    encnorm : nn.LayerNorm
        Layer normalization applied to the input before encoding.
    encode_mu : nn.Sequential
        Encoder network returning the latent mean.
    encode_rho : nn.Sequential
        Encoder network returning the latent log-scale, from which the latent standard
        deviation is softplus(rho) + 1e-4.
    decode_mu : nn.Linear
        Linear decoder returning the reconstruction mean.
    decode_rho : nn.Parameter
        Scalar log-scale of the reconstruction standard deviation, shared across all
        input dimensions.
    """
    def __init__(
            self, 
            input_dim: int, 
            latent_dim: int,
            hidden_dims: Optional[List[int]] = None,
            activation: Optional[Union[str, Callable[[], nn.Module]]] = "relu",
            dtype: torch.dtype = torch.float64
            ):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encnorm = nn.LayerNorm(input_dim, dtype=dtype)
        self.encode_mu = _mlp(input_dim, hidden_dims, latent_dim, act=activation, dtype=dtype)
        self.encode_rho = _mlp(input_dim, hidden_dims, latent_dim, act=activation, dtype=dtype)

        # Decoder
        self.decode_mu = nn.Linear(latent_dim, input_dim, dtype=dtype)
        self.decode_rho = nn.Parameter(
            torch.tensor([-2.0], dtype=dtype),
            requires_grad=True
        )

    @staticmethod
    def reparameterize(
            mu: torch.Tensor, 
            std: torch.Tensor,
            *,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Draw a latent sample with the reparameterization trick.

        Parameters
        ----------
        mu : torch.Tensor
            Latent mean of shape (batch_size, latent_dim).
        std : torch.Tensor
            Latent standard deviation of shape (batch_size, latent_dim).
        generator : torch.Generator, optional
            Generator used for the normal draw. If None, the global torch RNG is used.

        Returns
        -------
        torch.Tensor
            Latent sample of shape (batch_size, latent_dim).
        """
        eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        z = mu + eps * std
        return z

    def encode(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input features into latent mean and standard deviation.

        Parameters
        ----------
        y : torch.Tensor
            Input of shape (batch_size, input_dim), typically the residuals after
            covariate adjustment.

        Returns
        -------
        mu : torch.Tensor
            Latent mean of shape (batch_size, latent_dim).
        std : torch.Tensor
            Latent standard deviation of shape (batch_size, latent_dim).
        """
        y = self.encnorm(y)
        mu = self.encode_mu(y)
        rho = self.encode_rho(y)
        std = nn.functional.softplus(rho) + 1e-4
        return mu, std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent sample back into the input space.

        Parameters
        ----------
        z : torch.Tensor
            Latent sample of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstruction of shape (batch_size, input_dim).
        """
        y = self.decode_mu(z)
        return y

    def forward(
            self, 
            y: torch.Tensor,
            *,
            generator: Optional[torch.Generator] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, encode_std = self.encode(y)
        z = self.reparameterize(mu, encode_std, generator=generator)
        recon_y = self.decode(z)
        return mu, encode_std, z, recon_y

    def training_step(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative evidence lower bound averaged over the batch.

        Parameters
        ----------
        y : torch.Tensor
            Input of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Scalar loss equal to -mean(recon_ll + neg_kl), where recon_ll is the Gaussian
            reconstruction log-likelihood summed over features and neg_kl is the negative
            Kullback-Leibler divergence from the standard normal prior to the latent
            posterior, summed over latent dimensions.
        """
        mu, encode_std, z, recon_y = self(y)
        encode_std = torch.clamp(encode_std,
                                 min=1e-3,
                                 max=10.0)
        encode_logvar = 2.0 * torch.log(encode_std)

        decode_std = nn.functional.softplus(self.decode_rho) + 1e-4
        decode_std = torch.clamp(decode_std,
                                 min=1e-3,
                                 max=10.0)

        # Gaussian log-density written in closed form; expand so the gradient w.r.t.
        # decode_rho reduces over the same axes as a broadcast Normal log_prob.
        s = decode_std.expand(recon_y.shape)
        var = s ** 2
        recon_ll = (-((y - recon_y) ** 2) / (2 * var) - s.log() - math.log(math.sqrt(2 * math.pi))).sum(dim=-1)
        neg_kl = 0.5 * (1 + encode_logvar - mu.pow(2) - encode_logvar.exp()).sum(dim=-1)
        elbo = recon_ll + neg_kl
        loss = -elbo.mean()

        return loss
