import math
from types import MappingProxyType

import pandas as pd
import pytest
import torch

from kernel_elastic_autoencoder import Model, Pipeline, Top1Sampler, Trainer
from kernel_elastic_autoencoder.collate import Collator, DataframeCollator
from kernel_elastic_autoencoder.config import (
    ExperimentConfig,
    ModelCommonConfig,
    ModelConfig,
    ModelDecoderConfig,
    ModelEncoderConfig,
    ModelInputConfig,
    TrainingCommonConfig,
    TrainingConfig,
    TrainingHyperparameterConfig,
    TrainingOptimizerConfig,
)
from kernel_elastic_autoencoder.losses import Loss
from kernel_elastic_autoencoder.tokenizer import _DummySlowTokenizer


@pytest.fixture
def randn(initial_seed=0):
    def _factory(shape: torch.Size):
        rng = torch.rand(shape, generator=torch.manual_seed(_factory.seed))  # type: ignore
        _factory.seed += 1  # type: ignore
        return rng

    _factory.seed = initial_seed  # type: ignore
    return _factory


@pytest.fixture
def randlong(initial_seed=0):
    def _factory(low: int, high: int, shape: torch.Size):
        rng = torch.randint(
            low,
            high,
            shape,
            dtype=torch.long,
            generator=torch.manual_seed(_factory.seed),  # type: ignore
        )
        _factory.seed += 1  # type: ignore
        return rng

    _factory.seed = initial_seed  # type: ignore
    return _factory


@pytest.fixture
def filled():
    def _factory(value: float, shape: torch.Size):
        return (
            torch.ones(
                shape, dtype=torch.long if isinstance(value, int) else torch.float
            )
            * value
        )

    return _factory


@pytest.fixture
def up_tokenizer(sample_dataframe):
    def _factory(train=sample_dataframe()["seq"]):
        return _DummySlowTokenizer(train=list(train))

    return _factory


@pytest.fixture
def kae_pipeline(kae_model_pretrained, up_tokenizer):
    def _factory(
        model=kae_model_pretrained(),
        tokenizer=up_tokenizer(),
        collator=DataframeCollator,
        sampler=Top1Sampler,
        device=torch.device("cpu"),
    ):
        return Pipeline(
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            sampler=sampler,
            device=device,
        )

    return _factory


@pytest.fixture
def top_1_sampler(up_tokenizer):
    def _factory(tokenizer=up_tokenizer()):
        return Top1Sampler(tokenizer=tokenizer)

    return _factory


@pytest.fixture
def dataframe_collator(model_config, up_tokenizer):
    def _factory(
        collator=DataframeCollator,
        model_config=model_config(),
        tokenizer=up_tokenizer(),
    ) -> Collator:
        return collator(model_config=model_config, tokenizer=tokenizer)

    return _factory


@pytest.fixture
def sample_dataframe():
    def _factory(
        contents=MappingProxyType(
            {
                "seq": ["a", "bc", "def", "ghij", "klmno"],
                "cond1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "cond2": [0.5, 1.0, 1.5, 2.0, 2.5],
            }
        ),
    ):
        return pd.DataFrame().from_dict(dict(contents))

    return _factory


@pytest.fixture
def sample_dataframe_collated(sample_dataframe, dataframe_collator):
    def _factory(
        collator=dataframe_collator(),
        dataset=sample_dataframe(),
        seq_feature="seq",
        cond_features=(
            "cond1",
            "cond2",
        ),
        padding=True,
        add_special_tokens=True,
        device=torch.device("cpu"),
    ):
        return collator(
            dataset=dataset,
            seq_feature=seq_feature,
            cond_features=list(cond_features),
            padding=padding,
            add_special_tokens=add_special_tokens,
            device=device,
        )

    return _factory


@pytest.fixture
def kae_model(model_config):
    def _factory(config=model_config()):
        return Model(config=config)

    return _factory


@pytest.fixture
def kae_model_pretrained(kae_model, kae_trainer, sample_dataframe_collated):
    def _factory(
        trainer=kae_trainer(),
        model=kae_model(),
        ds=sample_dataframe_collated(),
        train_split=0.9,
        checkpoint="./tests/checkpoint",
    ):
        trainer.train(
            model=model,
            ds=ds,
            train_split=train_split,
            checkpoint=checkpoint,
        )
        return model

    return _factory


# TODO: Implement tests for generic trainer callbacks.
@pytest.fixture
def kae_trainer(training_config):
    def _factory(config=training_config()):
        return Trainer(config=config)

    return _factory


@pytest.fixture
def experiment_config(model_config, training_config):
    def _factory(
        model=model_config(),
        training=training_config(),
    ):
        return ExperimentConfig(model=model, training=training)

    return _factory


@pytest.fixture
def model_config(
    model_input_config, model_common_config, model_encoder_config, model_decoder_config
):
    def _factory(
        input=model_input_config(),
        common=model_common_config(),
        encoder=model_encoder_config(),
        decoder=model_decoder_config(),
    ):
        return ModelConfig(input=input, common=common, encoder=encoder, decoder=decoder)

    return _factory


@pytest.fixture
def model_input_config():
    def _factory(
        max_len=10,
        vocab_size=18,
        condition_channels=2,
    ):
        return ModelInputConfig(
            max_len=max_len,
            vocab_size=vocab_size,
            condition_channels=condition_channels,
        )

    return _factory


@pytest.fixture
def model_common_config():
    def _factory(
        embedding_dim=32,
        pooling_dim=10,
        padding_idx=0,
        padding_value=-100.0,
    ):
        return ModelCommonConfig(
            embedding_dim=embedding_dim,
            pooling_dim=pooling_dim,
            padding_idx=padding_idx,
            padding_value=padding_value,
        )

    return _factory


@pytest.fixture
def model_encoder_config():
    def _factory(
        num_layers=6,
        num_heads=4,
        feedforward_scale=4,
        dropout=0.1,
    ):
        return ModelEncoderConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_scale=feedforward_scale,
            dropout=dropout,
        )

    return _factory


@pytest.fixture
def model_decoder_config():
    def _factory(
        num_layers=6,
        num_heads=4,
        feedforward_scale=4,
        dropout=0.1,
    ):
        return ModelDecoderConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_scale=feedforward_scale,
            dropout=dropout,
        )

    return _factory


@pytest.fixture
def training_config(
    training_common_config, training_hyperparameter_config, training_optimizer_config
):
    def _factory(
        common=training_common_config(),
        hyperparameters=training_hyperparameter_config(),
        optimizer=training_optimizer_config(),
    ):
        return TrainingConfig(
            common=common, hyperparameters=hyperparameters, optimizer=optimizer
        )

    return _factory


@pytest.fixture
def training_common_config():
    def _factory(
        max_epochs=200,
        batch_size=24,
    ):
        return TrainingCommonConfig(max_epochs=max_epochs, batch_size=batch_size)

    return _factory


@pytest.fixture
def training_hyperparameter_config():
    def _factory(
        hp_lambda=3.5,
        hp_delta=1.0,
        hp_sigma=math.sqrt(32),
        kernel_dist_size=1000,
    ):
        return TrainingHyperparameterConfig(
            hp_lambda=hp_lambda,
            hp_delta=hp_delta,
            hp_sigma=hp_sigma,
            kernel_dist_size=kernel_dist_size,
        )

    return _factory


@pytest.fixture
def training_optimizer_config():
    def _factory(
        optimizer_fn="torch.optim.Adam",
        optimizer_params=MappingProxyType({"lr": 1e-4}),
        scheduler_fn="torch.optim.lr_scheduler.ReduceLROnPlateau",
        scheduler_params=MappingProxyType({"patience": 15}),
    ):
        return TrainingOptimizerConfig(
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
        )

    return _factory


@pytest.fixture
def kae_loss():
    def _factory(
        hp_lambda=3.5,
        hp_delta=1.0,
        hp_sigma=math.sqrt(32),
        kernel_dist_size=1000,
        padding_idx=0,
        embedding_dim=128,
        pooling_dim=10,
    ):
        return Loss(
            hp_lambda=hp_lambda,
            hp_delta=hp_delta,
            hp_sigma=hp_sigma,
            kernel_dist_size=kernel_dist_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            pooling_dim=pooling_dim,
        )

    return _factory
