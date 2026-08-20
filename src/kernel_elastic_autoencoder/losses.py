import torch
import torch.nn.functional as F
from torch import nn


class Loss(nn.Module):
    """Module for Kernel-Elastic Autoencoder loss.

    Layer storing hyperparameters, computing loss on forward pass. Losses are computed as:

    $\\mathcal{L}(\\lambda, \\delta)=\\mathcal{L}_{WCEL}(\\lambda, \\delta) + m$-$MMD(\\lambda)$
    """

    def __init__(
        self,
        *,
        hp_lambda: float,
        hp_delta: float,
        hp_sigma: float,
        kernel_dist_size: int,
        padding_idx: int,
        embedding_dim: int,
        pooling_dim: int,
    ):
        """Instantiates a Loss object.

        Args:
            hp_lambda: Hyperparameter $\\lambda$, as used in WCEL and m-MMD losses. Roughly, controls how strongly the
                shape of the latent vector distribution is penalized.
            hp_delta: Hyperparameter $\\delta$, as used in WCEL loss. Roughly, controls the relative weights of the
                vanilla-AE and VAE objectives in the reconstruction loss.
            hp_sigma: Hyperparameter $\\sigma$, as used in the Kernel function applied in m-MMD loss. Roughly,
                used as a scaling factor to control the sizes of gradients produced by the m-MMD loss.
            kernel_dist_size: Size of the sampled distribution of vectors used to penalize the shape of the latent
                vector distribution through the kernel.
            padding_idx: Index of padding token. Used internally to zero vectors corresponding to padding tokens through
                embedding layers. Should be fetched with Tokenizer.pad_token_id as specified in the Tokenizer protocol.
            embedding_dim: Embedding dimension used by nn.Embedding layers.
            pooling_dim: Sequence length dimension to which inputs are pooled after condition concatenation through the
                encoder. Proportional to the dimension of latent vectors.
        """
        super().__init__()
        self.hp_lambda = hp_lambda
        """Hyperparameter $\\lambda$, as used in WCEL and m-MMD losses. Roughly, controls how strongly the
                shape of the latent vector distribution is penalized."""
        self.hp_delta = hp_delta
        """Hyperparameter $\\delta$, as used in WCEL loss. Roughly, controls the relative weights of the
                vanilla-AE and VAE objectives in the reconstruction loss."""
        self.hp_sigma = hp_sigma
        """Hyperparameter $\\sigma$, as used in the Kernel function applied in m-MMD loss. Roughly,
                used as a scaling factor to control the sizes of gradients produced by the m-MMD loss."""
        self.kernel_dist_size = kernel_dist_size
        """Size of the sampled distribution of vectors used to penalize the shape of the latent
                vector distribution through the kernel."""
        self.padding_idx = padding_idx
        """Index of padding token. Used internally to zero vectors corresponding to padding tokens through
                embedding layers. Should be fetched with Tokenizer.pad_token_id as specified in the Tokenizer protocol."""
        self.embedding_dim = embedding_dim
        """Embedding dimension used by nn.Embedding layers."""
        self.pooling_dim = pooling_dim
        """Sequence length dimension to which inputs are pooled after condition concatenation through the
                encoder. Proportional to the dimension of latent vectors."""

        self.register_buffer("_loc", torch.zeros(self.pooling_dim * self.embedding_dim))
        self.register_buffer("_cov", torch.eye(self.pooling_dim * self.embedding_dim))

    def forward(
        self,
        prediction: torch.Tensor,
        prediction_noise: torch.Tensor,
        ground_truth: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the loss module. May also be called using Loss.__call__.

        Args:
            prediction: Tensor of dimension (B, M, L) containing teacher forcing prediction logits without noise
                before the decoder.
            prediction_noise: Tensor of dimension (B, M, L) containing teacher forcing prediction logits with added noise
                before the decoder.
            ground_truth: Tensor of dimension (B, M) containing the indices of ground truth tokens for each sequence.
            latents: Tensor of dimension (B, P * E) containing latent vectors produced by the encoder with added
                noise.

        Returns:
            torch.Tensor: Tensor of dimension (1) containing loss value.
        """
        return self._weighted_cross_entropy_loss(
            prediction, prediction_noise, ground_truth
        ) + self._modified_maximum_mean_discrepancy(latents)

    def _weighted_cross_entropy_loss(
        self,
        prediction: torch.Tensor,
        prediction_noise: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        ground_truth.to(torch.long)
        log_softmax = torch.log_softmax(prediction, dim=-1)
        log_softmax_noise = torch.log_softmax(prediction_noise, dim=-1)
        Y = F.one_hot(ground_truth, num_classes=prediction.size(-1))
        Y[..., self.padding_idx] = 0
        t1 = (Y * log_softmax_noise).sum(dim=(-1, -2))
        t2 = (Y * log_softmax).sum(dim=(-1, -2))
        coeff = -1 / (self.hp_lambda + self.hp_delta + 1)
        loss = coeff * (t1 + ((self.hp_lambda + self.hp_delta) * t2))
        return loss.mean()

    def _modified_maximum_mean_discrepancy(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        mvn = torch.distributions.MultivariateNormal(
            loc=self._loc,  # type: ignore
            covariance_matrix=self._cov,  # type: ignore
        )
        samples = mvn.rsample((self.kernel_dist_size,))
        square_difference_sum = torch.cdist(latents, samples, p=2.0).pow(2)
        kernel_pairwise_sum = torch.exp(
            ((-1 / (self.pooling_dim * self.embedding_dim)) * square_difference_sum)
            / (2 * (self.hp_sigma**2))
        ).sum()
        loss = self.hp_lambda * (
            torch.tensor(1)
            - ((1 / (latents.size(0) * self.kernel_dist_size)) * kernel_pairwise_sum)
        )
        return loss
