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
        self.device = device
        """Torch device used for inference."""
        if self.device is None:
            self.device = torch.device("cpu")
        self.model = model.to(self.device)
        """Pre-trained Model object for inference. Moved to Pipeline.device, and placed in eval() mode."""
        self.model.eval()
        self.tokenizer = tokenizer
        """Pre-configured Tokenizer object."""

    def completion(
        self,
        latents: torch.Tensor,
        sequences: list[str],
        conditions: list[list[float]] | torch.Tensor,
        with_grad: bool = False,
        **kwargs,
    ) -> Iterable[str]:
        """Completes each conditioned input sequence.

        The model completes each sequence in the provided list using decoder-only inference. Tokens are
        sampled greedily.

        Args:
            latents: Tensor of dimension (B, P * E) containing latent vectors for the batch.
            sequences: List of text sequences to complete.
            conditions: List of condition value lists per batch.
            **kwargs: Additional keyword arguments passed to Tokenizer.encode.

        Returns:
            Iterable[str]: List of completed sequences, stripped of special tokens.
        """
        input_ids = self.tokenizer.encode(
            text=sequences,
            padding="longest",
            add_special_tokens=False,
            return_tensors="pt",
            **kwargs,
        ).to(self.device)
        input_ids = torch.cat(
            [
                torch.full(
                    (input_ids.size(0), 1),
                    self.tokenizer.bos_token_id,
                    device=self.device,
                ),
                input_ids,
            ],
            dim=1,
        )
        latents = latents.to(self.device)
        conditions = torch.as_tensor(conditions, dtype=torch.float, device=self.device)
        conds_embed = self.model.embed_conditions(conditions)
        token_mask = (input_ids == self.tokenizer.pad_token_id).to(torch.bool)
        batches_completed = torch.zeros(
            input_ids.size(0), dtype=torch.bool, device=self.device
        )

        while (input_ids.size(1) < self.model.config_typed.input.max_len) and (
            not batches_completed.all()
        ):
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
            batches_completed |= new_toks.flatten() == self.tokenizer.eos_token_id
            new_toks = torch.where(
                batches_completed.unsqueeze(-1),
                self.tokenizer.pad_token_id,
                new_toks,
            )
            input_ids = torch.cat([input_ids, new_toks], dim=1)
            token_mask = torch.cat(
                [
                    token_mask,
                    batches_completed.unsqueeze(-1),
                ],
                dim=1,
            )
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def beam_completion(
        self,
        latents: torch.Tensor,
        beam_size: int,
        sequences: list[str],
        conditions: list[list[float]] | torch.Tensor,
        **kwargs,
    ) -> Iterable[str]:
        """Completes each conditioned input sequence, using the beam search strategy.

        The model completes each sequence in the provided list using decoder-only inference. The beam search
        decoding algorithm is used. On the last step, instead of returning beam_size candidates per batch, the
        candidate with the highest length-normalized sum of log odds is selected for each batch.

        Args:
            latents: Tensor of dimension (B, P * E) containing latent vectors for the batch.
            beam_size: Beam size of first step.
            sequences: List of text sequences to complete.
            conditions: List of condition value lists per batch.
            **kwargs: Additional keyword arguments passed to Tokenizer.encode.

        Returns:
            Iterable[str]: List of completed sequences, stripped of special tokens.
        """
        input_ids = self.tokenizer.encode(
            text=sequences,
            padding="longest",
            add_special_tokens=False,
            return_tensors="pt",
            **kwargs,
        ).to(self.device)
        input_ids = torch.cat(
            [
                torch.full(
                    (input_ids.size(0), 1),
                    self.tokenizer.bos_token_id,
                    device=self.device,
                ),
                input_ids,
            ],
            dim=1,
        )
        token_mask = (input_ids == self.tokenizer.pad_token_id).to(torch.bool)
        latents = latents.to(self.device)
        input_probs = torch.zeros_like(input_ids).repeat_interleave(beam_size, dim=0)
        conditions = torch.as_tensor(conditions, dtype=torch.float, device=self.device)
        conds_embed = self.model.embed_conditions(conditions)
        batches_completed = torch.zeros(
            input_ids.size(0) * beam_size, dtype=torch.bool, device=self.device
        )

        logits = self.model.decode(
            current_output=input_ids,
            latents=latents,
            condition_embeddings=conds_embed,
            token_mask=token_mask,
        )
        odds = logits.log_softmax(dim=-1)
        new_toks = (
            torch.topk(odds[:, -1:], k=beam_size, dim=-1)
            .indices.flatten()
            .unsqueeze(-1)
            .to(torch.long)
        )
        new_probs = (
            torch.topk(odds[:, -1:], k=beam_size, dim=-1)
            .values.flatten()
            .unsqueeze(-1)
        )
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
        conds_embed = conds_embed.repeat_interleave(beam_size, dim=0)
        latents = latents.repeat_interleave(beam_size, dim=0)
        token_mask = token_mask.repeat_interleave(beam_size, dim=0)
        batches_completed |= new_toks.flatten() == self.tokenizer.eos_token_id
        token_mask = torch.cat(
            [
                token_mask,
                batches_completed.unsqueeze(-1),
            ],
            dim=1,
        )

        while (input_ids.size(1) < self.model.config_typed.input.max_len) and (
            not batches_completed.all()
        ):
            logits = self.model.decode(
                current_output=input_ids,
                latents=latents,
                condition_embeddings=conds_embed,
                token_mask=token_mask,
            )
            odds = logits.log_softmax(dim=-1)
            new_toks = (
                torch.topk(odds[:, -1:], k=beam_size, dim=-1)
                .indices.flatten()
                .unsqueeze(-1)
                .to(torch.long)
            )
            new_probs = (
                torch.topk(odds[:, -1:], k=beam_size, dim=-1)
                .values.flatten()
                .unsqueeze(-1)
            )
            new_toks = torch.where(
                batches_completed.repeat_interleave(beam_size, dim=0).unsqueeze(-1),
                self.tokenizer.pad_token_id,
                new_toks,
            )
            new_probs = torch.where(
                batches_completed.repeat_interleave(beam_size, dim=0).unsqueeze(-1),
                0.0,
                new_probs,
            )

            candidate_ids = torch.cat(
                [input_ids.repeat_interleave(beam_size, dim=0), new_toks], dim=1
            )
            candidate_probs = torch.cat(
                [input_probs.repeat_interleave(beam_size, dim=0), new_probs], dim=1
            )
            top_probs = candidate_probs.view(
                candidate_probs.size(0) // (beam_size**2), beam_size**2, -1
            )
            grouped_ids = candidate_ids.view(top_probs.size(0), top_probs.size(1), -1)
            top_prob_inds = (
                (
                    top_probs.sum(dim=-1)
                    / torch.sqrt(grouped_ids != self.tokenizer.pad_token_id)
                    .to(torch.long)
                    .sum(dim=-1)
                )
                .topk(k=beam_size, dim=1)
                .indices.squeeze(-1)
            )
            input_ids = grouped_ids[
                torch.arange(top_probs.size(0)).unsqueeze(-1).repeat(1, beam_size),
                top_prob_inds,
            ].view(input_ids.size(0), -1)
            input_probs = top_probs[
                torch.arange(top_probs.size(0)).unsqueeze(-1).repeat(1, beam_size),
                top_prob_inds,
            ].view(input_ids.size(0), -1)

            batches_completed |= (
                input_ids[:, -1:].flatten() == self.tokenizer.eos_token_id
            )
            token_mask = torch.cat(
                [
                    token_mask,
                    batches_completed.unsqueeze(-1),
                ],
                dim=1,
            )

        top_probs = input_probs.reshape(input_probs.size(0) // beam_size, beam_size, -1)
        grouped_ids = input_ids.view(top_probs.shape[0], beam_size, -1)
        top_prob_inds = (
            (
                top_probs.sum(dim=-1)
                / torch.sqrt(grouped_ids != self.tokenizer.pad_token_id)
                .to(torch.long)
                .sum(dim=-1)
            )
            .topk(k=1, dim=1)
            .indices.squeeze(-1)
        )
        winning_ids = grouped_ids[torch.arange(top_probs.size(0)), top_prob_inds]
        return self.tokenizer.decode(winning_ids, skip_special_tokens=True)

    def encoding(
        self, sequences: list[str], conditions: list[list[float]] | torch.Tensor
    ) -> torch.Tensor:
        input_ids = self.tokenizer.encode(
            text=sequences,
            padding="max_length",
            max_length=self.model.config_typed.input.max_len,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)
        conditions = torch.as_tensor(conditions, dtype=torch.float, device=self.device)
        token_mask = (input_ids == self.model.config_typed.common.padding_idx).to(
            dtype=torch.bool, device=self.device
        )
        condition_mask = (
            conditions == self.model.config_typed.common.padding_value
        ).to(dtype=torch.bool, device=self.device)

        return self.model.encode(
            input_ids=input_ids,
            conditions=conditions,
            token_mask=token_mask,
            condition_mask=condition_mask,
        )
