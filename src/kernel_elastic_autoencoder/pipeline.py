from collections.abc import Iterable

import torch

from kernel_elastic_autoencoder.model import Model
from kernel_elastic_autoencoder.tokenizer import Tokenizer


class Pipeline:
    """User-facing pipeline for inference.

    Defines an easy-to-use API for decoder-only inference with a pretrained model.

    Examples:
        Wrapping a pretrained model and tokenizer:
        >>> model = Model.from_pretrained("./checkpoint")
        >>> tokenizer = MyTokenizer.from_pretrained("./checkpoint/tokenizer")
        >>> pipe = Pipeline(model, tokenizer)

        Getting a completion for sequences:
        >>> compl = pipe.completion(latents, ["abc", "def", "ghi"], [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])
        >>> print(compl.outputs)
    """

    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        device: torch.device | None = None,
    ) -> None:
        """Instantiates a Pipeline object.

        Args:
            model: Pre-trained Model object for inference. Can be obtained from Model.from_pretrained.
            tokenizer: Pre-configured Tokenizer object. The Tokenizer protocol supports tokenizers
                inheriting from transformers.PreTrainedTokenizerBase, so such tokenizers may be loaded
                from HuggingFace Hub.
            device: Torch device used for inference.
        """
        self.model = model.to(device)
        """Pre-trained Model object for inference. Moved to Pipeline.device, and placed in eval() mode."""
        self.model.eval()
        self.tokenizer = tokenizer
        """Pre-configured Tokenizer object."""
        self.device = device
        """Torch device used for inference."""

    def _ingest(
        self,
        sequences: list[str],
        conditions: list[list[float]] | torch.Tensor,
        device: torch.device | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self.tokenizer.encode(
            seq=sequences,
            padding=False,
            max_length=self.model.config_typed.input.max_len,
            add_special_tokens=False,
            **kwargs,
        )
        conditions = torch.as_tensor(conditions, dtype=torch.float, device=device)
        condition_mask = (
            conditions != self.model.config_typed.common.padding_value
        ).to(torch.bool)
        return input_ids, conditions, condition_mask

    @torch.inference_mode()
    def _completion_entry(
        self,
        latents: torch.Tensor,
        sequences: list[str],
        conditions: list[list[float]] | torch.Tensor,
        device: torch.device | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        input_ids, conds, cond_mask = self._ingest(
            sequences, conditions, device, **kwargs
        )
        conds_embed = self.model.embed_conditions(conds)
        return {
            "latents": latents,
            "input_ids": input_ids,
            "batches_completed": torch.zeros(input_ids.size(0), dtype=torch.bool),
            "condition_embeddings": conds_embed,
            "condition_mask": cond_mask,
        }

    @torch.inference_mode()
    def _completion_step(
        self, intermediate: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        intermediate["input_ids"] = torch.cat(
            [
                torch.full(
                    (intermediate["input_ids"].size(0), 1), self.tokenizer.bos_token_id
                ),
                intermediate["input_ids"],
            ],
            dim=1,
        )
        logits = self.model.decode(
            current_output=intermediate["input_ids"],
            latents=intermediate["latents"],
            condition_embeddings=intermediate["condition_embeddings"],
            token_mask=None,
            condition_mask=intermediate["condition_mask"],
        )
        new_toks = (
            torch.topk(logits[:, -1:], k=1, dim=-1).indices.squeeze(-1).to(torch.long)
        )
        intermediate["batches_completed"] |= (
            new_toks.squeeze(-1) == self.tokenizer.eos_token_id
        )
        new_toks = torch.where(
            intermediate["batches_completed"].unsqueeze(-1),
            self.tokenizer.pad_token_id,
            new_toks,
        )
        intermediate["input_ids"] = torch.cat([intermediate["input_ids"], new_toks], dim=1)
        return intermediate

    @torch.inference_mode()
    def _completion_exit(self, intermediate: dict[str, torch.Tensor]) -> Iterable[str]:
        return self.tokenizer.decode(intermediate["input_ids"], skip_special_tokens=True)

    def completion(
        self,
        latents: torch.Tensor,
        sequences: list[str],
        conditions: list[list[float]] | torch.Tensor,
        device: torch.device | None = None,
        **kwargs,
    ) -> Iterable[str]:
        """Completes each conditioned input sequence.

        The model completes each sequence in the provided list using decoder-only inference. Tokens are
        sampled greedily.

        Args:
            latents: Tensor of dimension (B, P * E) containing latent vectors for the batch.
            sequences: List of text sequences to complete.
            conditions: List of condition value lists per batch.
            device: Torch device used for inference.
            **kwargs: Additional keyword arguments passed to Tokenizer.encode.

        Returns:
            Iterable[str]: List of completed sequences, stripped of special tokens.
        """
        intermediate = self._completion_entry(
            latents=latents,
            sequences=sequences,
            conditions=conditions,
            device=device,
            **kwargs,
        )
        while (
            intermediate["input_ids"].size(1) < self.model.config_typed.input.max_len
        ) and (not intermediate["batches_completed"].all()):
            intermediate = self._completion_step(intermediate=intermediate)
        return self._completion_exit(intermediate=intermediate)
