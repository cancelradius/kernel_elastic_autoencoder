from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, Self

import torch


class Tokenizer(Protocol):
    """Protocol to be implemented for tokenizers. Supports transformers.PreTrainedTokenizerBase.

    Provides a specification for tokenizing text for model input, and recovering text from token indices.
    """

    vocab_size: int
    """Number of tokens in the tokenizer's vocabulary."""
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
        seq: Iterable[str],
        padding: bool,
        max_length: int,
        add_special_tokens: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Encodes an Iterable of sequences to a tensor of indices. Optionally, adds special tokens according to a
        template and pads the outputs to a fixed length.

        Args:
            seq: Iterable of text sequences to encode.
            padding: Whether to pad the sequences to a fixed length.
            max_length: Maximum length to which sequences are padded if padding is True.
            add_special_tokens: Whether to add special tokens according to a template.
            **kwargs: Keyword arguments.

        Returns:
            torch.Tensor: Tensor of dimension (B, S) containing vocabulary indices.
        """
        ...

    def decode(
        self, ids: torch.Tensor, skip_special_tokens: bool, **kwargs
    ) -> Iterable[str]:
        """Decodes a tensor of indices to an Iterable of sequences. Optionally, skips special tokens.

        Args:
            ids: Tensor of dimension (B, S) containing vocabulary indices to decode.
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


class _DummySlowTokenizer(Tokenizer):
    def __init__(self, train: Iterable[str]):
        self.pad_token = "_"
        self.bos_token = "?"
        self.eos_token = "!"

        vocab = set[str]()
        for b in train:
            vocab = vocab.union(set(b))

        special_tokens = [self.pad_token, self.bos_token, self.eos_token]
        special_tokens.extend(list(vocab))
        self.vocab = special_tokens
        self.vocab_size = len(self.vocab)

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def encode(
        self,
        seq: Iterable[str],
        padding: bool,
        max_length: int,
        add_special_tokens: bool,
        **kwargs,
    ) -> torch.Tensor:
        ids_col = list[torch.Tensor]()
        for b in seq:
            toks = list(b)
            if add_special_tokens:
                toks.insert(0, self.bos_token)
                toks.append(self.eos_token)
            if padding:
                toks += [self.pad_token] * (max_length - len(toks))
            ids = torch.tensor([self.vocab.index(t) for t in toks], dtype=torch.long)
            ids_col.append(ids)
        return torch.stack(ids_col, dim=0)

    def decode(
        self, ids: torch.Tensor, skip_special_tokens: bool, **kwargs
    ) -> Iterable[str]:
        if skip_special_tokens:
            return [
                str(
                    [
                        (
                            self.vocab[idx]
                            if idx
                            not in [
                                self.pad_token_id,
                                self.bos_token_id,
                                self.eos_token_id,
                            ]
                            else ""
                        )
                        for idx in b
                    ]
                )
                for b in ids
            ]
        else:
            return [str([self.vocab[idx] for idx in b]) for b in ids]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs): ...
