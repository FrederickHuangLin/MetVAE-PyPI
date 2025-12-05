import torch
import torch.nn as nn
import torch.distributions as dists
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
    VAE with optional MLP nonlinearities.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    latent_dim : int
        Size of the latent representation (number of latent variables).
    hidden_dims : list[int] or None, optional (default=None)
        Sequence of hidden layer sizes for the encoder and decoder.
        If None or an empty list, the encoder and decoder are simple linear mappings.
    activation : str or callable or None, optional (default="relu")
        Nonlinear activation to use between hidden layers.
        Can be a string ("relu", "tanh", "gelu", "silu") or a callable that returns an nn.Module.
        If None, no activation is applied (purely linear encoder/decoder).
    dtype : torch.dtype, optional (default=torch.float64)
        Data type for all model parameters and layers (e.g., torch.float32 or torch.float64).

    Attributes
    ----------
    encnorm : nn.LayerNorm
        Layer normalization applied to input features before encoding.
    encode_mu : nn.Module
        Encoder network mapping input features to the latent mean vector μ.
    encode_rho : nn.Module
        Encoder network mapping input features to the latent log-scale parameter ρ,
        used to compute the latent standard deviation σ = softplus(ρ).
    decode_mu : nn.Linear
        Decoder network mapping latent variables back to reconstructed inputs.
    decode_rho : nn.Parameter
        Learnable log-scale parameter controlling the global reconstruction variance
        (shared across all input dimensions).

    The model uses the reparameterization trick for the variational inference and employs a standard Gaussian
    prior over the latent variables.
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
        Reparameterization trick to sample from the latent space.

        Parameters:
        ----------
        mu : torch.Tensor, (batch_size, latent_dim)
            Mean of the latent space distribution.
        std : torch.Tensor, (batch_size, latent_dim)
            Standard deviation of the latent space distribution.

        Returns:
        -------
        z : torch.Tensor, (batch_size, latent_dim)
            Sampled tensor from the latent space distribution.
        """
        eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        z = mu + eps * std
        return z

    def encode(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input features into latent space representations.

        Parameters:
        ----------
        y : torch.Tensor, (batch_size, input_dim)
            Input tensor, usually the residuals after removing the effects of confounders.

        Returns:
        -------
        mu : torch.Tensor, (batch_size, latent_dim)
            Mean of the latent space distribution.
        std : torch.Tensor, (batch_size, latent_dim)
            Standard deviation of the latent space distribution.
        """
        y = self.encnorm(y)
        mu = self.encode_mu(y)
        rho = self.encode_rho(y)
        std = nn.functional.softplus(rho) + 1e-4
        return mu, std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent space representation back into the input space.

        Parameters:
        ----------
        z : torch.Tensor, (batch_size, latent_dim)
            Latent space representation.

        Returns:
        -------
        y : torch.Tensor, (batch_size, input_dim)
            Decoded tensor representing the original input features (reconstruction).
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
        mu, encode_std, z, recon_y = self(y)
        encode_std = torch.clamp(encode_std, 
                                 min=1e-3, 
                                 max=10.0)
        encode_logvar = 2.0 * torch.log(encode_std)
        
        decode_std = nn.functional.softplus(self.decode_rho) + 1e-4
        decode_std = torch.clamp(decode_std, 
                                 min=1e-3, 
                                 max=10.0)

        # ELBO
        recon_ll = dists.Normal(loc=recon_y, scale=decode_std).log_prob(y).sum(dim=-1)
        kl = 0.5 * (1 + encode_logvar - mu.pow(2) - encode_logvar.exp()).sum(dim=-1)
        elbo = recon_ll + kl 
        loss = -elbo.mean()
        
        return loss
