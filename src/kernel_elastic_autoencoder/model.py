import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from kernel_elastic_autoencoder.config import ModelConfig
from kernel_elastic_autoencoder.layers import Decoder, Encoder, TrainingNoise

CODERS = {
    ModelConfig: (
        lambda x: x.model_dump(),
        lambda data: ModelConfig.model_validate(data),
    )
}


class Model(
    nn.Module,
    PyTorchModelHubMixin,
    coders=CODERS,
    library_name="kernel_elastic_autoencoder",
    repo_url="https://github.com/cancelradius/kae",
    paper_url="https://arxiv.org/abs/2310.08685",
):
    """Base class for Kernel-Elastic Autoencoder models.

    Defines a barebones API for model calls with tensor inputs. For easy use with standard data formats, calling Model
    through the Pipeline API may be preferred.
    """

    def __init__(self, config: dict | ModelConfig):
        """Instantiates a Model object.

        Args:
            config: Dictionary or ModelConfig schema defining model parameters. Will be validated with ModelConfig
                regardless of input type.
        """
        super().__init__()

        self.config = config
        """Configuration object for Hugging Face Hub compatible serialization. Not recommended to use, as it is 
        internal-use. Use Model.config_typed instead."""
        config_typed = ModelConfig.model_validate(config)
        self.config_typed: ModelConfig = config_typed
        """Type-validated config in a Pydantic ModelConfig schema, recommended for public API use."""

        self.encoder = Encoder(
            max_len=config_typed.input.max_len,
            vocab_size=config_typed.input.vocab_size,
            condition_channels=config_typed.input.condition_channels,
            embedding_dim=config_typed.common.embedding_dim,
            pooling_dim=config_typed.common.pooling_dim,
            padding_idx=config_typed.common.padding_idx,
            padding_value=config_typed.common.padding_value,
            num_layers=config_typed.encoder.num_layers,
            num_heads=config_typed.encoder.num_heads,
            feedforward_scale=config_typed.encoder.feedforward_scale,
            dropout=config_typed.encoder.dropout,
        )
        """Model encoder module."""
        self.noise = TrainingNoise()
        """Noise-adding module for training purposes."""
        self.decoder = Decoder(
            max_len=config_typed.input.max_len,
            vocab_size=config_typed.input.vocab_size,
            condition_channels=config_typed.input.condition_channels,
            embedding_dim=config_typed.common.embedding_dim,
            pooling_dim=config_typed.common.pooling_dim,
            padding_idx=config_typed.common.padding_idx,
            padding_value=config_typed.common.padding_value,
            num_layers=config_typed.decoder.num_layers,
            num_heads=config_typed.decoder.num_heads,
            feedforward_scale=config_typed.decoder.feedforward_scale,
            dropout=config_typed.decoder.dropout,
        )
        """Model decoder module."""

    def forward(
        self,
        input_ids: torch.Tensor,
        conditions: torch.Tensor,
        token_mask: torch.Tensor,
        condition_mask: torch.Tensor,
    ):
        """Forward pass for training loop. Produces all tensors needed for computing loss with
        Loss.forward.

        Args:
            input_ids: Tensor of dimension (B, M) containing input ids for each sequence.
            conditions: Tensor of dimension (B, C) containing condition values for each sequence.
            token_mask: Tensor of dimension (B, M) containing boolean padding masks for each sequence.
            condition_mask: Tensor of dimension (B, C) containing boolean condition padding masks for each sequence.

        Returns:
            torch.Tensor: Tensor of dimension (B, M, L) containing teacher forcing prediction logits without noise
                before the decoder.
            torch.Tensor: Tensor of dimension (B, M, L) containing teacher forcing prediction logits with added noise
                before the decoder.
            torch.Tensor: Tensor of dimension (B, P * E) containing latent vectors produced by the encoder with added
                noise.
        """
        padding_mask = torch.cat((token_mask, condition_mask), 1)
        latents, condition_embeddings = self.encoder(
            input_ids, conditions, padding_mask
        )
        latents_noise = self.noise(latents)

        prediction = self.decoder(
            input_ids[:, :-1],
            latents,
            condition_embeddings,
            padding_mask[:, : self.config_typed.input.max_len - 1],
        )
        prediction_noise = self.decoder(
            input_ids[:, :-1],
            latents_noise,
            condition_embeddings,
            padding_mask[:, : self.config_typed.input.max_len - 1],
        )
        return prediction, prediction_noise, latents_noise

    def encode(
        self,
        input_ids: torch.Tensor,
        conditions: torch.Tensor,
        token_mask: torch.Tensor | None,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Basic interface for a forward pass through the Model.model.encoder module.

        Args:
            input_ids: Tensor of dimension (B, M) containing input ids for each sequence.
            conditions: Tensor of dimension (B, C) containing condition values for each sequence.
            token_mask: Tensor of dimension (B, M) containing boolean padding masks for each sequence.
            condition_mask: Tensor of dimension (B, C) containing boolean condition padding masks for each sequence.

        Returns:
            torch.Tensor: Tensor of dimension (B, P * E) containing latent vectors produced by the encoder.
        """
        padding_mask = (
            torch.cat((token_mask, condition_mask), 1)
            if (token_mask is not None)
            else torch.cat((torch.full_like(input_ids, True), condition_mask), 1)
        )
        return self.encoder(input_ids, conditions, padding_mask)[0]

    def decode(
        self,
        current_output: torch.Tensor,
        latents: torch.Tensor,
        condition_embeddings: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Basic interface for a forward pass through the Model.model.decoder module.

        Args:
            current_output: Tensor of dimension (B, S) containing previous output ids for each sequence. In
                autoregressive generation, this should initially represent a single start-of-sequence token.
            latents: Tensor of dimension (B, P * E) containing latent vectors. May be produced using Model.encode.
            condition_embeddings: Tensor of dimension (B, C, E) containing condition embeddings for each sequence. May
                be produced using Model.embed_conditions.
            token_mask: Tensor of dimension (B, S) containing boolean padding masks for each sequence. If None is
                passed, the absence of padding is assumed.

        Returns:
            torch.Tensor: Tensor of dimension (B, S, L) containing prediction logits produced by the decoder.
        """
        padding_mask = (token_mask
            if (token_mask is not None)
            else torch.full_like(current_output, False)
        ).to(torch.bool)
        return self.decoder(current_output, latents, condition_embeddings, padding_mask)

    def embed_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        """Basic interface for the separate embedding of condition vectors.

        Args:
            conditions: Tensor of dimension (B, C) containing condition values for each sequence.

        Returns:
            torch.Tensor: Tensor of dimension (B, C, E) containing condition embeddings for each sequence.
        """
        return self.encoder.embedding.conditional_embedding(conditions)
