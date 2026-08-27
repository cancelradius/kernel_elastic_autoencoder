import math

import pytest
import torch

from kernel_elastic_autoencoder.losses import Loss


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
def kae_loss():
    def _factory(
        hp_lambda=3.5,
        hp_delta=1.0,
        hp_sigma=math.sqrt(0.32),
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
