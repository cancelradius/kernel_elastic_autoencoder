from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch

from kernel_elastic_autoencoder.tokenizer import Tokenizer


class Sampler(ABC):
    """Base class for samplers.

    Provides a specification for sampling text sequences from logits.
    """

    def __init__(self, tokenizer: Tokenizer, **kwargs):
        """Instantiates a Sampler object.

        Args:
            tokenizer: Object implementing the Tokenizer protocol.
            **kwargs: Sampler-specific keyword args.
        """
        self.tokenizer = tokenizer
        """Object implementing the Tokenizer protocol."""

    @abstractmethod
    def __call__(
        self, logits: torch.Tensor, skip_special_tokens: bool, **kwargs
    ) -> Iterable[str]:
        """Interface method for implementing the last sampling step where outputs are decoded through the tokenizer.

        Args:
            logits: Tensor of dimension (B, S, L) containing logits at the last step of inference.
            **kwargs: Keyword arguments.

        Returns:
            Iterable[str]: Iterable of strings decoded by the tokenizer.
        """
        ...

    @abstractmethod
    def sample_ids(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Interface method for implementing index sampling from logits in intermediate steps.

        Args:
            logits: Tensor of dimension (B, S, L) containing logits.
            **kwargs: Keyword arguments.

        Returns:
            torch.Tensor: Tensor of dimension (B, S) containing vocabulary indices.
        """
        ...


class Top1Sampler(Sampler):
    def __call__(self, logits: torch.Tensor, skip_special_tokens: bool, **kwargs) -> Iterable[str]:
        return self.tokenizer.decode(logits, skip_special_tokens=skip_special_tokens)

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
