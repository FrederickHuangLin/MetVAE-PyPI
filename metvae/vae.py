from typing import Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.distributions as dists
from torch import Tensor

class VAE(nn.Module):
    """
    This class implements a Variational Autoencoder (VAE) with the capability to account for confounders 
    in the input data. 
    
    Parameters:
    - input_dim (int): The dimensionality of the input data.
    - latent_dim (int): The size of the latent space to which the input data is encoded.
    - confound_dim (int): The dimensionality of the confounder variables.

    Attributes:
    - encode_mu (nn.Linear): Linear layer to produce the mean of the latent space representation.
    - encode_logvar (nn.Linear): Linear layer to produce the log variance of the latent space representation.
    - decode_mu (nn.Linear): Linear layer to decode the latent representation back to the input space.
    - decode_logvar (nn.Parameter): Learnable parameter for the log variance used in the decoding process.
    - confound (nn.Linear): Linear layer to model the effect of confounders on the input data.

    The model uses the reparameterization trick for the variational inference and employs a standard Gaussian
    prior over the latent variables.
    """
    def __init__(self, input_dim: int, latent_dim: int, confound_dim: int):
        super(VAE, self).__init__()

        # Encoder
        self.encnorm = nn.BatchNorm1d(input_dim)
        self.encode_mu = nn.Linear(input_dim, latent_dim)
        self.encode_logvar = nn.Linear(input_dim, latent_dim)

        # Decoder
        self.decode_mu = nn.Linear(latent_dim, input_dim)
        self.decode_logvar = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent space.

        Parameters:
        ----------
        mu : torch.Tensor, (batch_size, latent_dim)
            Mean of the latent space distribution.
        logvar : torch.Tensor, (batch_size, latent_dim)
            Log variance of the latent space distribution.

        Returns:
        -------
        z : torch.Tensor, (batch_size, latent_dim)
            Sampled tensor from the latent space distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
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
        logvar : torch.Tensor, (batch_size, latent_dim)
            Log variance of the latent space distribution.
        """
        y = self.encnorm(y)
        mu = self.encode_mu(y)
        logvar = self.encode_logvar(y)
        return mu, logvar

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

    def forward(self, x: Optional[torch.Tensor], y: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)
        recon_y = self.decode(z)

        return mu, logvar, z, recon_y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # x: confounders, y: clr-transformed data
        x, y = batch 
        mu, logvar, z, recon_y = self(x, y)
        std = torch.exp(0.5 * logvar)

        # ELBO
        log_p_y_g_zx = dists.Normal(loc=recon_y, scale=torch.exp(0.5 * self.decode_logvar)).log_prob(y)
        log_p_z = dists.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(std)).log_prob(z)
        log_q_z_g_yx = dists.Normal(loc=mu, scale=std).log_prob(z)
        loss = log_p_y_g_zx.sum(dim=-1) + log_p_z.sum(dim=-1) - log_q_z_g_yx.sum(dim=-1)
        loss = loss.mean()
        return -loss

class IWAE(nn.Module):
    """
    Implements an Importance Weighted Autoencoder (IWAE), an extension of the Variational Autoencoder (VAE) that
    improves the approximation of the log-likelihood by using multiple importance samples. The IWAE aims to
    provide a more accurate estimate of the data log-likelihood compared to standard VAEs by averaging over
    multiple samples from the variational posterior. This class also incorporates adjustments for confounders
    and features presence modulation similar to the VAE class.

    Parameters:
    - input_dim (int): The dimensionality of the input data.
    - latent_dim (int): The size of the latent space.
    - confound_dim (int): The dimensionality of the confounder variables.
    - num_importance_samples (int, optional): The number of importance samples to use for estimating the
      log-likelihood. Default is 1.

    Attributes:
    - encode_mu (nn.Linear): Linear layer to produce the mean of the latent space representation.
    - encode_logvar (nn.Linear): Linear layer to produce the log variance of the latent space representation.
    - decode_mu (nn.Linear): Linear layer to decode the latent representation back to the input space.
    - decode_logvar (nn.Parameter): Learnable parameter for the log variance used in the decoding process.
    - confound (nn.Linear): Linear layer to model the effect of confounders on the input data.
    - k (int): Stores the number of importance samples specified for the log-likelihood estimation.


    The IWAE uses the reparameterization trick for variational inference, with the key difference being the
    use of multiple importance samples to calculate a tighter lower bound on the log-likelihood of the input
    data, potentially leading to improved learning of the latent space.
    """
    def __init__(self, input_dim: int, latent_dim: int, confound_dim: int, num_importance_samples: int=1):
        super(IWAE, self).__init__()

        # Encoder layers
        self.encnorm = nn.BatchNorm1d(input_dim)
        self.encode_mu = nn.Linear(input_dim, latent_dim)
        self.encode_logvar = nn.Linear(input_dim, latent_dim)

        # Decoder layer
        self.decnorm = nn.BatchNorm1d(latent_dim)
        self.decode_mu = nn.Linear(latent_dim, input_dim)
        self.decode_logvar = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        
        # The number of importance samples
        self.k = num_importance_samples

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent space.

        Parameters:
        ----------
        mu : torch.Tensor, (batch_size, latent_dim)
            Mean of the latent space distribution.
        logvar : torch.Tensor, (batch_size, latent_dim)
            Log variance of the latent space distribution.

        Returns:
        -------
        z : torch.Tensor, (batch_size, latent_dim)
            Sampled tensor from the latent space distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
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
        logvar : torch.Tensor, (batch_size, latent_dim)
            Log variance of the latent space distribution.
        """
        y = self.encnorm(y)
        mu = self.encode_mu(y)
        logvar = self.encode_logvar(y)

        return mu, logvar

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
        z = torch.permute(z, (0, 2, 1))
        # z = self.decnorm(z)
        z = torch.permute(z, (0, 2, 1))
        y = self.decode_mu(z)
        return y

    def forward(self, x: Optional[torch.Tensor], y: torch.Tensor, k=1) -> Union[
        tuple[Any, Any, Any, Tensor], tuple[Any, Tensor, Tensor, Tensor]]:
        mu, logvar = self.encode(y)
        if k > 1:
            mu_ups = mu.unsqueeze(1).repeat(1, k, 1)
            logvar_ups = logvar.unsqueeze(1).repeat(1, k, 1)
        else:
            mu_ups = mu.unsqueeze(1)
            logvar_ups = logvar.unsqueeze(1)

        z = self.reparameterize(mu_ups, logvar_ups)

        recon_y = self.decode(z)
        # fix the dimension if sample number is 1
        if k == 1:
            mu_ups = mu_ups.squeeze()
            logvar_ups = logvar_ups.squeeze()
            z = z.squeeze()
            recon_y = recon_y.squeeze()

        return mu_ups, logvar_ups, z, recon_y


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # x: confounders, y: clr-transformed data, k: number of importance samples
        k = self.k
        x, y = batch

        mu, logvar, z, recon_y = self(x, y, k)
        # Reconstruction loss
        if k > 1:
            y = y.unsqueeze(1).repeat(1, k, 1)
        std = torch.exp(0.5 * logvar)
        
        # IWAE loss
        log_p_y_g_zx = dists.Normal(loc=recon_y, scale=torch.exp(0.5 * self.decode_logvar)).log_prob(y)
        log_prior_z = dists.Normal(0, 1).log_prob(z)
        log_q_z_g_yx = dists.Normal(mu, std).log_prob(z)
        log_w = log_p_y_g_zx.sum(-1) + log_prior_z.sum(-1) - log_q_z_g_yx.sum(-1)
        
        if k > 1:
            log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w_tilde = log_w_tilde.exp().detach()
            loss = (w_tilde * log_w).sum(1).mean()
        else:
            loss = log_w.mean()
        return -loss
