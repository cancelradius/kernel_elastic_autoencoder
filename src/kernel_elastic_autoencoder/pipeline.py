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
        >>> print(compl)
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
        input_ids = self.tokenizer.encode(
            text=sequences,
            padding='do_not_pad',
            max_length=self.model.config_typed.input.max_len,
            add_special_tokens=False,
            return_tensors="pt",
            **kwargs,
        ).to(self.device)
        conditions = torch.as_tensor(conditions, dtype=torch.float, device=device)
        condition_mask = (
            conditions != self.model.config_typed.common.padding_value
        ).to(torch.bool)
        conds_embed = self.model.embed_conditions(conditions)
        batches_completed = torch.zeros(input_ids.size(0), dtype=torch.bool)

        while (input_ids.size(1) < self.model.config_typed.input.max_len) and (
            not batches_completed.all()
        ):
            input_ids = torch.cat(
                [
                    torch.full(
                        (input_ids.size(0), 1),
                        self.tokenizer.bos_token_id,
                    ),
                    input_ids,
                ],
                dim=1,
            )
            logits = self.model.decode(
                current_output=input_ids,
                latents=latents,
                condition_embeddings=conds_embed,
                token_mask=token_mask,
            )
            new_toks = (
                torch.topk(logits[:, -1:], k=1, dim=-1)
                .indices.squeeze(-1)
                .to(torch.long)
            )
            batches_completed |= new_toks.squeeze(-1) == self.tokenizer.eos_token_id
            new_toks = torch.where(
                batches_completed.unsqueeze(-1),
                self.tokenizer.pad_token_id,
                new_toks,
            )
            input_ids = torch.cat([input_ids, new_toks], dim=1)
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def beam_completion(
        self,
        latents: torch.Tensor,
        beam_size: int,
        sequences: list[str],
        conditions: list[list[float]] | torch.Tensor,
        device: torch.device | None = None,
        **kwargs,
    ) -> Iterable[str]:
        """Completes each conditioned input sequence, using the beam search strategy.

        The model completes each sequence in the provided list using decoder-only inference. On the
        first step, the top `beam_size` first tokens are chosen for each batch. Subsequent tokens are
        sampled greedily in parallel for every subsequence. Before returning output, the subsequence with
        the highest sum of token logits is chosen for each batch.

        Args:
            latents: Tensor of dimension (B, P * E) containing latent vectors for the batch.
            beam_size: Beam size of first step.
            sequences: List of text sequences to complete.
            conditions: List of condition value lists per batch.
            device: Torch device used for inference.
            **kwargs: Additional keyword arguments passed to Tokenizer.encode.

        Returns:
            Iterable[str]: List of completed sequences, stripped of special tokens.
        """
        input_ids = self.tokenizer.encode(
            text=sequences,
            padding='do_not_pad',
            max_length=self.model.config_typed.input.max_len,
            add_special_tokens=False,
            return_tensors="pt",
            **kwargs,
        ).to(self.device)
        input_probs = torch.zeros_like(input_ids * beam_size)
        conditions = torch.as_tensor(conditions, dtype=torch.float, device=device)
        condition_mask = (
            (conditions != self.model.config_typed.common.padding_value)
            .to(torch.bool)
            .repeat_interleave(beam_size, dim=0)
        )
        conds_embed = self.model.embed_conditions(conditions).repeat_interleave(
            beam_size, dim=0
        )
        batches_completed = torch.zeros(input_ids.size(0) * beam_size, dtype=torch.bool)

        logits = self.model.decode(
            current_output=input_ids,
            latents=latents,
            condition_embeddings=conds_embed,
            token_mask=token_mask,
        )
        new_toks = (
            torch.topk(logits[:, -1:], k=beam_size, dim=-1)
            .indices.squeeze(-1)
            .to(torch.long)
        )
        new_probs = (
            torch.topk(logits[:, -1:], k=beam_size, dim=-1)
            .values.squeeze(-1)
            .to(torch.long)
        )
        batches_completed |= new_toks.squeeze(-1) == self.tokenizer.eos_token_id
        new_toks = torch.where(
            batches_completed.unsqueeze(-1),
            self.tokenizer.pad_token_id,
            new_toks,
        )
        new_probs = torch.where(
            batches_completed.unsqueeze(-1),
            0.0,
            new_probs,
        )
        input_ids = torch.cat(
            [input_ids.repeat_interleave(beam_size, dim=0), new_toks], dim=1
        )
        input_probs = torch.cat([input_probs, new_probs], dim=1)

        while (input_ids.size(1) < self.model.config_typed.input.max_len) and (
            not batches_completed.all()
        ):
            input_ids = torch.cat(
                [
                    torch.full(
                        (input_ids.size(0), 1),
                        self.tokenizer.bos_token_id,
                    ),
                    input_ids,
                ],
                dim=1,
            )
            logits = self.model.decode(
                current_output=input_ids,
                latents=latents,
                condition_embeddings=conds_embed,
                token_mask=token_mask,
            )
            new_toks = (
                torch.topk(logits[:, -1:], k=1, dim=-1)
                .indices.squeeze(-1)
                .to(torch.long)
            )
            new_probs = (
                torch.topk(logits[:, -1:], k=1, dim=-1)
                .values.squeeze(-1)
                .to(torch.long)
            )
            batches_completed |= new_toks.squeeze(-1) == self.tokenizer.eos_token_id
            new_toks = torch.where(
                batches_completed.unsqueeze(-1),
                self.tokenizer.pad_token_id,
                new_toks,
            )
            new_probs = torch.where(
                batches_completed.unsqueeze(-1),
                0.0,
                new_probs,
            )
            input_ids = torch.cat([input_ids, new_toks], dim=1)
            input_probs = torch.cat([input_probs, new_probs], dim=1)

        top_probs = input_probs.reshape(input_probs.size(0) // beam_size, beam_size, -1)
        top_prob_inds = top_probs.sum(dim=-1).topk(k=1, dim=1).indices
        winning_ids = torch.take_along_dim(
            input_ids, top_prob_inds.unsqueeze(-1), 1
        ).squeeze(1)
        return self.tokenizer.decode(winning_ids, skip_special_tokens=True)
