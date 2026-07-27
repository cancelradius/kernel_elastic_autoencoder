from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch

from kernel_elastic_autoencoder.tokenizer import Tokenizer


class Sampler(ABC):
    """Base class for samplers.

    Provides a specification for sampling text sequences from logits.
    """

    def __init__(self, tokenizer: Tokenizer):
        """Instantiates a Sampler object.

        Args:
            tokenizer: Object implementing the Tokenizer protocol.
        """
        self.tokenizer = tokenizer
        """Object implementing the Tokenizer protocol."""

    def __call__(
        self, logits: torch.Tensor, skip_special_tokens: bool, **kwargs
    ) -> Iterable[str]:
        ids = self.sample_ids(logits, **kwargs)
        sampled = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return sampled

    @abstractmethod
    def sample_ids(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Interface method for implementing index sampling from logits.

        Args:
            logits: Tensor of dimension (B, S, L) containing logits.
            **kwargs: Keyword arguments.

        Returns:
            torch.Tensor: Tensor of dimension (B, S) containing vocabulary indices.
        """
        ...


class Top1Sampler(Sampler):
    def sample_ids(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Implementation of index sampling from logits choosing the highest-probability token.

        Args:
            logits: Tensor of dimension (B, S, L) containing logits.
            **kwargs: Keyword arguments.

        Returns:
            torch.Tensor: Tensor of dimension (B, S) containing vocabulary indices.
        """
        return (
            torch.topk(logits, k=1, dim=-1, **kwargs).indices.squeeze(-1).to(torch.long)
        )
