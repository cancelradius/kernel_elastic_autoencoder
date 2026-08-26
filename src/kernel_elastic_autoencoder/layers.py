import torch
import torch.nn.functional as F
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(max_len + 1, embedding_dim, padding_idx=0)

    def forward(self, x):
        indices = torch.arange(1, x.size(1) + 1, device=x.device).repeat(x.size(0), 1)
        masked_indices = indices.masked_fill(x == self.padding_idx, 0)
        return self.embedding(masked_indices)


class ConditionEmbedding(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        embedding_dim: int,
        padding_value: float,
    ):
        super().__init__()
        self.padding_value = padding_value
        self.embedding = nn.Embedding(
            condition_channels + 1, embedding_dim, padding_idx=0
        )
        self.register_buffer("indices", torch.arange(1, condition_channels + 1))

    def forward(self, c: torch.Tensor):
        indices = self.indices.repeat(c.size(0), 1).to(c.device)  # type: ignore
        masked_indices = indices.masked_fill(c == self.padding_value, 0)
        embedding = self.embedding(masked_indices)
        return embedding * c.unsqueeze(-1).repeat(1, 1, embedding.size(-1))


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_size: int,
        embedding_dim: int,
        condition_channels: int,
        padding_idx: int,
        padding_value: float,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.positional_embedding = LearnedPositionalEmbedding(
            max_len, embedding_dim, padding_idx
        )
        self.conditional_embedding = ConditionEmbedding(
            condition_channels,
            embedding_dim,
            padding_value,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        x = self.token_embedding(x) + self.positional_embedding(x)

        if c is not None:
            c = self.conditional_embedding(c)
            x = torch.cat([x, c], dim=1)

        return x, c


class SeqLenLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        return x


class TrainingNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        dist = torch.distributions.Normal(
            torch.zeros_like(x, device=x.device),
            torch.ones_like(x, device=x.device),
        )
        sample = dist.rsample().to(x.device)
        x = x + sample
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        max_len: int,
        vocab_size: int,
        embedding_dim: int,
        pooling_dim: int,
        condition_channels: int,
        dropout: float,
        padding_idx: int,
        padding_value: float,
        num_layers: int,
        num_heads: int,
        feedforward_scale: int,
    ):
        super().__init__()

        # Layers
        ## (B, M) -> (B, (M + C), E), (C, E)
        self.embedding = TransformerEmbedding(
            max_len,
            vocab_size,
            embedding_dim,
            condition_channels,
            padding_idx,
            padding_value,
        )
        ## (B, (M + C), E) -> (B, (M + C), E)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embedding_dim,
                num_heads,
                embedding_dim * feedforward_scale,
                dropout,
                F.relu,
                batch_first=True,
                norm_first=False,
            ),
            num_layers,
        )
        ## (B, (M + C), E) -> (B, P, E)
        self.compression_feedforward = SeqLenLinear(
            max_len + condition_channels, pooling_dim
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, padding_mask: torch.Tensor):
        x, c = self.embedding(x, c)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.compression_feedforward(x)
        x = x.flatten(start_dim=1)
        return x, c


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        max_len: int,
        vocab_size: int,
        embedding_dim: int,
        pooling_dim: int,
        condition_channels: int,
        dropout: float,
        padding_idx: int,
        padding_value: float,
        num_layers: int,
        num_heads: int,
        feedforward_scale: int,
    ):
        super().__init__()
        self.pooling_dim = pooling_dim
        self.embedding_dim = embedding_dim

        # Layers
        ## (B, P, E) -> (B, (P + C), E)
        ## (B, (P + C), E) -> (B, P, E)
        self.mixing_feedforward = SeqLenLinear(
            pooling_dim + condition_channels, pooling_dim
        )
        ## (B, (M - 1)) -> (B, (M - 1), E)
        self.embedding = TransformerEmbedding(
            max_len,
            vocab_size,
            embedding_dim,
            condition_channels,
            padding_idx,
            padding_value,
        )
        ## (B, (M - 1), E), (B, P, E) -> (B, (M - 1), E)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim,
                num_heads,
                embedding_dim * feedforward_scale,
                dropout,
                F.relu,
                batch_first=True,
                norm_first=False,
            ),
            num_layers,
        )
        ## (B, (M - 1), E) -> (B, (M - 1), L)
        self.out_linear = nn.Linear(embedding_dim, vocab_size)

        self.register_buffer(
            "causal_mask", nn.Transformer.generate_square_subsequent_mask(max_len).to(torch.bool)
        )

    def forward(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
        c: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        x = x.unflatten(1, (self.pooling_dim, self.embedding_dim))
        x = torch.cat([x, c], dim=1)
        x = self.mixing_feedforward(x)
        out, _ = self.embedding(out)
        out = self.transformer_decoder(
            out,
            x,
            tgt_mask=self.causal_mask[: out.size(1), : out.size(1)],  # type: ignore
            tgt_key_padding_mask=padding_mask[:, : out.size(1)],
        )
        out = self.out_linear(out)
        return out
