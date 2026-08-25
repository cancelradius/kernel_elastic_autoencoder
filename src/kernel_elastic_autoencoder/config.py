from __future__ import annotations

import math
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    ImportString,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls.model_validate(d)

    def to_json(self, f: str):
        return Path(f).write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, f: FilePath):
        return cls.model_validate_json(Path(f).read_text())


class ExperimentConfig(Config):
    """Configuration schema for an experiment.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    model: ModelConfig = Field(..., description="ModelConfig object.")
    training: TrainingConfig = Field(..., description="TrainingConfig object.")


class ModelInputConfig(Config):
    """ModelConfig schema block containing input parameters.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    max_len: PositiveInt = Field(
        default=150,
        description="Maximum sequence length through the encoder and decoder.",
    )
    vocab_size: PositiveInt = Field(
        default=50,
        description="Vocabulary size of tokenizer. Should be fetched with Tokenizer.vocab_size as specified in the "
        "Tokenizer Protocol.",
    )
    condition_channels: PositiveInt = Field(
        default=2,
        description="Number of condition channels, should correspond to number of numerical columns in your "
        "dataset/inputs.",
    )


class ModelCommonConfig(Config):
    """ModelConfig schema block containing common architecture parameters.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    embedding_dim: PositiveInt = Field(
        default=128, description="Embedding dimension used by nn.Embedding layers."
    )
    pooling_dim: PositiveInt = Field(
        default=10,
        description="Sequence length dimension to which inputs are pooled after condition concatenation through the "
        "encoder. Proportional to the dimension of latent vectors.",
    )
    padding_idx: NonNegativeInt = Field(
        default=0,
        description="Index of padding token. Used internally to zero vectors corresponding to padding tokens through "
        "embedding layers. Should be fetched with Tokenizer.pad_token_id as specified in the Tokenizer "
        "protocol.",
    )
    padding_value: float = Field(
        default=-1e10,
        description='Value of "padded" condition fields. Should be set according to the value you use as a '
        "placeholder for blank numerical fields in your dataset.",
    )


class ModelEncoderConfig(Config):
    """ModelConfig schema block containing architecture parameters of the encoder.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    num_layers: PositiveInt = Field(
        default=6, description="Number of Transformer encoder layers in the encoder."
    )
    num_heads: PositiveInt = Field(
        default=4,
        description="Number of self-attention heads used in each Transformer encoder layer in the encoder.",
    )
    feedforward_scale: PositiveInt = Field(
        default=4,
        description="Factor by which the dimension of the hidden layer in the FFNs differs from the dimension of the "
        "input in the encoder. Applies to Transformer and Compression FFNs.",
    )
    dropout: PositiveFloat = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout rate applied to Transformer encoder layers in the encoder.",
    )


class ModelDecoderConfig(Config):
    """ModelConfig schema block containing architecture parameters of the decoder.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    num_layers: PositiveInt = Field(
        default=6, description="Number of Transformer decoder layers in the decoder."
    )
    num_heads: PositiveInt = Field(
        default=4,
        description="Number of self- and cross-attention heads used in each Transformer decoder layer in the decoder.",
    )
    feedforward_scale: PositiveInt = Field(
        default=4,
        description="Factor by which the dimension of the hidden layer in the FFNs differs from the dimension of the "
        "input in the decoder. Applies to Transformer and Mixing FFNs.",
    )
    dropout: PositiveFloat = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout rate applied to Transformer decoder layers in the decoder.",
    )


class ModelConfig(Config):
    """Configuration schema for a Model. Contains all parameters needed to instantiate a Model.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    input: ModelInputConfig = Field(
        default=ModelInputConfig(), description="ModelInputConfig object."
    )
    common: ModelCommonConfig = Field(
        default=ModelCommonConfig(), description="ModelCommonConfig object."
    )
    encoder: ModelEncoderConfig = Field(
        default=ModelEncoderConfig(), description="ModelEncoderConfig object."
    )
    decoder: ModelDecoderConfig = Field(
        default=ModelDecoderConfig(), description="ModelDecoderConfig object."
    )


class TrainingCommonConfig(Config):
    """TrainingConfig schema block containing common training parameters.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    max_epochs: PositiveInt = Field(
        default=200, description="Maximum number of epochs to train for."
    )
    batch_size: PositiveInt = Field(
        default=64,
        description="Batch size for training. Large values are highly prone to OOM because batches are all padded to "
        "a fixed length, as opposed to dynamically.",
    )


class TrainingHyperparameterConfig(Config):
    """TrainingConfig schema block containing hyperparameters.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    hp_lambda: float = Field(
        default=3.5,
        description=r"Hyperparameter $\lambda$, as used in WCEL and m-MMD losses. Roughly, controls how strongly the "
        r"shape of the latent vector distribution is penalized.",
    )
    hp_delta: float = Field(
        default=1.0,
        description=r"Hyperparameter $\delta$, as used in WCEL loss. Roughly, controls the relative weights of the "
        r"vanilla-AE and VAE objectives in the reconstruction loss.",
    )
    hp_sigma: float = Field(
        default=math.sqrt(32),
        description=r"Hyperparameter $\sigma$, as used in the Kernel function applied in m-MMD loss. Roughly, "
        r"used as a scaling factor to control the sizes of gradients produced by the m-MMD loss.",
    )
    kernel_dist_size: PositiveInt = Field(
        default=1000,
        description="Size of the sampled distribution of vectors used to penalize the shape of the latent vector "
        "distribution through the kernel.",
    )


class TrainingOptimizerConfig(Config):
    """TrainingConfig schema block containing configurations for the optimizer and scheduler.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    optimizer_fn: ImportString = Field(
        default="torch.optim.AdamW",
        description="Optimizer function import string. Should come from torch.optim.",
        validate_default=True,
    )
    optimizer_params: dict = Field(
        default={},
        description="Dictionary of optimizer parameters to pass to **kwargs.",
    )
    scheduler_fn: ImportString = Field(
        default="torch.optim.lr_scheduler.LinearLR",
        description="Scheduler function import string. Should come from torch.optim.lr_scheduler.",
        validate_default=True,
    )
    scheduler_params: dict = Field(
        default={},
        description="Dictionary of scheduler parameters to pass to **kwargs.",
    )


class TrainingConfig(Config):
    """Configuration schema for a Trainer. Contains all parameters needed to instantiate a Trainer. Hyperparameter defaults are set according to experimentally-determined best practice in the original paper.

    TODO: Pydantic configs will likely be deprecated or heavily refactored in the near future.
    """

    common: TrainingCommonConfig = Field(
        default=TrainingCommonConfig(), description="TrainingCommonConfig object."
    )
    hyperparameters: TrainingHyperparameterConfig = Field(
        default=TrainingHyperparameterConfig(),
        description="TrainingHyperparameterConfig object.",
    )
    optimizer: TrainingOptimizerConfig = Field(
        default=TrainingOptimizerConfig(), description="TrainingOptimizerConfig object."
    )
