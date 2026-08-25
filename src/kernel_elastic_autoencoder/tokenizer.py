from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, Self

import torch


class Tokenizer(Protocol):
    """Protocol to be implemented for tokenizers. Supports transformers.PreTrainedTokenizerBase.

    Provides a specification for tokenizing text for model input, and recovering text from token indices.
    """

    bos_token: str
    """Token marking the beginning of a sequence."""
    bos_token_id: int
    """Index of token marking the beginning of a sequence."""
    eos_token: str
    """Token marking the end of a sequence."""
    eos_token_id: int
    """Index of token marking the end of a sequence."""
    pad_token: str
    """Token representing padding."""
    pad_token_id: int
    """Index of token representing padding."""

    def encode(
        self,
        text: Iterable[str],
        padding: str,
        max_length: int,
        add_special_tokens: bool,
        return_tensors: str,
        **kwargs,
    ) -> torch.Tensor:
        """Encodes an Iterable of sequences to a tensor of indices. Optionally, adds special tokens according to a
        template and pads the outputs to a fixed length.

        Args:
            text: Iterable of text sequences to encode.
            padding: Whether to pad the sequences to a fixed length.
            max_length: Maximum length to which sequences are padded if padding is True.
            add_special_tokens: Whether to add special tokens according to a template.
            return_tensors: Tensor return type, must be either 'pt' or 'np'.
            **kwargs: Keyword arguments.

        Returns:
            torch.Tensor: Tensor of dimension (B, S) containing vocabulary indices.
        """
        ...

    def decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool, **kwargs
    ) -> Iterable[str]:
        """Decodes a tensor of indices to an Iterable of sequences. Optionally, skips special tokens.

        Args:
            token_ids: Tensor of dimension (B, S) containing vocabulary indices to decode.
            skip_special_tokens: Whether to skip decoding special tokens when constructing outputs.
            **kwargs: Keyword arguments.

        Returns:
            Iterable[str]: Iterable of decoded sequences.
        """
        ...

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs
    ) -> Self:
        """Loads a tokenizer from a path or remote name. Supports Hugging Face from_pretrained.

        Args:
            pretrained_model_name_or_path: Local path or remote name.
            **kwargs: Keyword arguments.

        Returns:
            Tokenizer: Pretrained tokenizer.
        """
        ...

    def __len__(self) -> int: ...
