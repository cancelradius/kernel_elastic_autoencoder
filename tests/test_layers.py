import pytest
import torch

from kernel_elastic_autoencoder.layers import TransformerEmbedding


class TestTransformerEmbedding:
    @pytest.mark.parametrize(
        "batch_size,condition_channels,embedding_dim,max_len,vocab_size,padding_idx,padding_value",
        [
            (5, 3, 10, 10, 4, 0, -100.0),
            (10, 1, 20, 4, 6, 1, -30.0),
        ],
    )
    def test_shapes(
        self,
        batch_size,
        condition_channels,
        embedding_dim,
        max_len,
        vocab_size,
        padding_idx,
        padding_value,
        randlong,
        randn,
    ):
        transformer_embedding = TransformerEmbedding(
            max_len,
            vocab_size,
            embedding_dim,
            condition_channels,
            padding_idx,
            padding_value,
        )
        x = randlong(0, vocab_size, (batch_size, max_len))
        c = randn((batch_size, condition_channels))
        embedding = transformer_embedding(x, c)
        assert embedding[0].shape == torch.Size(
            [batch_size, max_len + condition_channels, embedding_dim]
        )

    @pytest.mark.parametrize(
        "padding_idx,padding_value",
        [
            (0, -10.0),
            (1, -100.0),
        ],
    )
    def test_padding(self, padding_idx, padding_value, filled):
        transformer_embedding = TransformerEmbedding(
            10,
            3,
            30,
            1,
            padding_idx,
            -100.0,
        )
        x = filled(padding_idx, (1, 10))
        c = filled(padding_value, (1, 1))
        embedding = transformer_embedding(x, c)
        assert torch.all(embedding[0] == 0.0)
        assert torch.all(embedding[1] == 0.0)
