import pytest
import torch


class TestKAELoss:
    @pytest.mark.parametrize(
        "batch_size,seq_len,vocab_size",
        [
            (4, 100, 20),
            (2, 50, 10),
        ],
    )
    def test_shapes(self, kae_loss, randn, randlong, batch_size, seq_len, vocab_size):
        loss = kae_loss()
        prediction = randn((batch_size, seq_len, vocab_size))
        prediction_noise = randn((batch_size, seq_len, vocab_size))
        ground_truth = randlong(0, vocab_size, (batch_size, seq_len))
        latents = randn((batch_size, loss.pooling_dim * loss.embedding_dim))
        c_loss = loss(prediction, prediction_noise, ground_truth, latents)
        assert c_loss.shape == torch.Size([])

    @pytest.mark.parametrize("padding_idx", [0, 1])
    def test_padding(self, kae_loss, randn, randlong, filled, padding_idx):
        loss = kae_loss(hp_lambda=0.0, padding_idx=padding_idx)
        prediction1 = randn((4, 100, 20))
        prediction1_noise = randn((4, 100, 20))
        prediction2 = randn((4, 100, 20))
        prediction2_noise = randn((4, 100, 20))
        ground_truth = filled(padding_idx, (4, 100))
        latents = randn((4, loss.pooling_dim * loss.embedding_dim))
        assert (
            loss(prediction1, prediction1_noise, ground_truth, latents)
            == loss(prediction2, prediction2_noise, ground_truth, latents)
            == 0.0
        )

    def test_hp_zero(self, kae_loss, randn, randlong, filled):
        loss = kae_loss(hp_lambda=0.0, hp_delta=0.0)
        prediction1 = randn((4, 100, 20))
        prediction2 = randn((4, 100, 20))
        prediction_noise = randn((4, 100, 20))
        ground_truth = randlong(0, 20, (4, 100))
        latents = torch.zeros((4, loss.pooling_dim * loss.embedding_dim))
        assert loss(prediction1, prediction_noise, ground_truth, latents) == loss(
            prediction2, prediction_noise, ground_truth, latents
        )
