from collections.abc import Callable, Iterable

import torch
from pydantic import BaseModel, ConfigDict

from kernel_elastic_autoencoder.collate import Collator, DataframeCollator
from kernel_elastic_autoencoder.config import ModelConfig
from kernel_elastic_autoencoder.model import Model
from kernel_elastic_autoencoder.sample import Sampler, Top1Sampler
from kernel_elastic_autoencoder.tokenizer import Tokenizer


class CompletionIntermediate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    latents: torch.Tensor
    input_ids: torch.Tensor
    batches_completed: torch.Tensor
    condition_embeddings: torch.Tensor
    condition_mask: torch.Tensor


class Completion(BaseModel):
    """Return schema for Pipeline.completion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    outputs: Iterable[str]
    """Iterable of generated output sequences."""
    condition_embeddings: torch.Tensor
    """Tensor containing generated condition embeddings for reuse."""
    condition_mask: torch.Tensor
    """Tensor containing generated condition masks for reuse."""


class Pipeline[T]:
    """User-facing pipeline for inference.

    Defines an easy-to-use API for decoder-only inference with a pretrained model.

    Examples:
        Wrapping a pretrained model and tokenizer:
        >>> model = Model.from_pretrained("./checkpoint")
        >>> tokenizer = MyClassImplementingTokenizer.from_pretrained("./checkpoint/tokenizer")
        >>> pipe = Pipeline(model, tokenizer, MyClassImplementingCollator, MyClassImplementingSampler)
        >>> compl = pipe.completion(latents, dataset, seq_feature, cond_features)
        >>> print(compl.outputs)

        TODO: Complete examples.
    """

    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        collator: Callable[[ModelConfig, Tokenizer], Collator[T]] = DataframeCollator,
        sampler: Callable[[Tokenizer], Sampler] = Top1Sampler,
        device: torch.device | None = None,
    ) -> None:
        """Instantiates a Pipeline object.

        TODO: Pipeline setup will be streamlined such that collators and samplers are by default fetched from model
            configuration.

        Args:
            model: Pre-trained Model object for inference. Can be obtained from Model.from_pretrained.
            tokenizer: Pre-configured Tokenizer object. The Tokenizer protocol supports tokenizers inheriting from
                transformers.PreTrainedTokenizerBase, so such tokenizers may be loaded from HuggingFace Hub.
            collator: Constructor for a data collator inheriting from Collator. Determines input types to Pipeline methods.
            sampler: Constructor for a data sampler inheriting from Sampler. Determines how outputs are produced from
                Pipeline methods.
            device: Torch device used for inference.
        """
        self.model = model.to(device)
        """Pre-trained Model object for inference. Moved to Pipeline.device, and placed in eval() mode."""
        self.model.eval()
        self.tokenizer = tokenizer
        """Pre-configured Tokenizer object. The Tokenizer Protocol supports tokenizers inheriting from
            transformers.PreTrainedTokenizerBase, so such tokenizers may be loaded from HuggingFace Hub."""
        self.device = device
        """Torch device used for inference."""

        self.collator = collator(model.config_typed, tokenizer)
        """Data collator inheriting from Collator, instantiated by passing Pipeline.model.config_typed and Pipeline.tokenizer."""
        self.sampler = sampler(tokenizer)
        """Data sampler inheriting from Sampler, instantiated by passing Pipeline.tokenizer."""

    @torch.inference_mode()
    def _completion_entry(
        self,
        latents: torch.Tensor,
        dataset: T,
        seq_feature: str,
        cond_features: list[str],
        device: torch.device | None = None,
        **kwargs,
    ) -> CompletionIntermediate:
        ds = self.collator(
            dataset=dataset,
            seq_feature=seq_feature,
            cond_features=cond_features,
            padding=False,
            add_special_tokens=False,
            device=device,
            **kwargs,
        )
        conds_embed = self.model.embed_conditions(ds.conditions)
        return CompletionIntermediate(
            latents=latents,
            input_ids=ds.input_ids,
            batches_completed=torch.zeros(ds.input_ids.size(0), dtype=torch.bool),
            condition_embeddings=conds_embed,
            condition_mask=ds.condition_mask,
        )

    @torch.inference_mode()
    def _completion_step(
        self, intermediate: CompletionIntermediate, **kwargs
    ) -> CompletionIntermediate:
        intermediate.input_ids = torch.cat(
            [
                torch.full(
                    (intermediate.input_ids.size(0), 1), self.tokenizer.bos_token_id
                ),
                intermediate.input_ids,
            ],
            dim=1,
        )
        logits = self.model.decode(
            current_output=intermediate.input_ids,
            latents=intermediate.latents,
            condition_embeddings=intermediate.condition_embeddings,
            token_mask=None,
            condition_mask=intermediate.condition_mask,
        )
        new_toks = self.sampler.sample_ids(logits[:, -1:])
        intermediate.batches_completed |= (
            new_toks.squeeze(-1) == self.tokenizer.eos_token_id
        )
        torch.where(
            intermediate.batches_completed.unsqueeze(-1),
            self.tokenizer.pad_token_id,
            new_toks,
        )
        intermediate.input_ids = torch.cat([intermediate.input_ids, new_toks], dim=1)
        return intermediate

    @torch.inference_mode()
    def _completion_exit(
        self, intermediate: CompletionIntermediate, **kwargs
    ) -> Completion:
        return Completion(
            outputs=self.sampler(
                intermediate.input_ids, skip_special_tokens=True
            ),
            condition_embeddings=intermediate.condition_embeddings,
            condition_mask=intermediate.condition_mask,
        )

    def completion(
        self,
        latents: torch.Tensor,
        dataset: T,
        seq_feature: str,
        cond_features: list[str],
        device: torch.device | None = None,
        **kwargs,
    ) -> Completion:
        """Completes each batch in an input dataset with latent and condition guidance through the decoder.

        Using decoder-only inference, the model completes each batch in the provided dataset. Dataset input typing is
        uniquely defined by the used Collator. The input dataset is passed through the collator, then subject to
        autoregressive decoder-only inference through the model, where new tokens are sampled from intermediates and
        stored in a tensor format. Finally, on reaching end-of-sequence tokens or the maximum sequence length, the
        sequence is decoded by the tokenizer and returned along with the generated condition embeddings and masks in a
        Completion schema.

        TODO: Callbacks are planned to enable easy custom inference functions and streaming of text.

        Args:
            latents: Tensor of dimension (B, P * E) containing latent vectors for the batches.
            dataset: Dataset of type specified in Pipeline.collator.
            seq_feature: String to pass to collator containing the feature or column containing text sequences to
                complete.
            cond_features: List of strings to pass to collator containing features or columns containing numerical
                condition values.
            device: Torch device used for inference.
            **kwargs: Additional keyword arguments passed to Pipeline.collator.

        Returns:
            Completion: Completion object, containing outputs, as well as condition embeddings and masks for reuse.
        """
        intermediate = self._completion_entry(
            latents=latents,
            dataset=dataset,
            seq_feature=seq_feature,
            cond_features=cond_features,
            device=device,
            **kwargs,
        )
        while (
            intermediate.input_ids.size(1) < self.model.config_typed.input.max_len
        ) and (not intermediate.batches_completed.all()):
            intermediate = self._completion_step(intermediate=intermediate)
        return self._completion_exit(intermediate=intermediate)
