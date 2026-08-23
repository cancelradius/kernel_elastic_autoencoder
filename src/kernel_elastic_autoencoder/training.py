import os
from collections.abc import Iterable

import torch
from accelerate import Accelerator
from accelerate.utils import tqdm

from kernel_elastic_autoencoder.config import TrainingConfig
from kernel_elastic_autoencoder.losses import Loss
from kernel_elastic_autoencoder.model import Model
from kernel_elastic_autoencoder.tokenizer import Tokenizer


class Trainer:
    def __init__(
        self,
        config: dict | TrainingConfig,
    ):
        """Instantiates a Trainer object.

        Args:
            config: Dictionary or TrainingConfig schema defining model parameters. Will be validated with TrainingConfig
                regardless of input type.
        """
        self.config = config
        """Configuration object for Hugging Face Hub compatible serialization. Not recommended to use, as it is 
                internal-use. Use Trainer.config_typed instead."""
        config_typed = TrainingConfig.model_validate(config)
        self.config_typed: TrainingConfig = config_typed
        """Type-validated config in a Pydantic TrainingConfig schema, recommended for public API use."""

    def train(
        self,
        model: Model,
        tokenizer: Tokenizer,
        sequences: Iterable[str],
        conditions: Iterable[Iterable[float]] | torch.Tensor,
        train_split: float = 0.9,
        checkpoint: str = "./checkpoint",
    ):
        """Trains a model. Optionally, resumes from an existing checkpoint.

        Args:
            model: Freshly instantiated model.
            tokenizer: Pre-configured Tokenizer object. The Tokenizer protocol supports tokenizers
                inheriting from transformers.PreTrainedTokenizerBase, so such tokenizers may be loaded
                from HuggingFace Hub.
            sequences: Iterable of text sequences.
            conditions: Iterable of iterables of condition values per sequence.
            train_split: Fraction of dataset used for training. Must be between 0 and 1.
            checkpoint: Path of local checkpoint to be saved and/or resumed.
        """
        accelerator = Accelerator()
        device = accelerator.device

        loss_fn = Loss(
            hp_lambda=self.config_typed.hyperparameters.hp_lambda,
            hp_delta=self.config_typed.hyperparameters.hp_delta,
            hp_sigma=self.config_typed.hyperparameters.hp_sigma,
            kernel_dist_size=self.config_typed.hyperparameters.kernel_dist_size,
            padding_idx=model.config_typed.common.padding_idx,
            embedding_dim=model.config_typed.common.embedding_dim,
            pooling_dim=model.config_typed.common.pooling_dim,
        )
        optimizer = self.config_typed.optimizer.optimizer_fn(
            model.parameters(), **self.config_typed.optimizer.optimizer_params
        )
        scheduler = self.config_typed.optimizer.scheduler_fn(
            optimizer, **self.config_typed.optimizer.scheduler_params
        )

        input_ids = tokenizer.encode(
            seq=sequences,
            padding=True,
            max_length=model.config_typed.input.max_len,
            add_special_tokens=True,
        )
        conditions = torch.as_tensor(conditions, dtype=torch.float)
        token_mask = (input_ids != model.config_typed.common.padding_idx).to(torch.bool)
        condition_mask = (conditions != model.config_typed.common.padding_value).to(
            torch.bool
        )

        dataset = torch.utils.data.TensorDataset(
            input_ids, conditions, token_mask, condition_mask
        )
        dataset_train, dataset_test = torch.utils.data.random_split(
            dataset, [train_split, 1 - train_split]
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.config_typed.common.batch_size,
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.config_typed.common.batch_size,
        )

        if os.path.exists(checkpoint):
            accelerator.load_state(checkpoint)

        model, optimizer, dataloader_train, scheduler = accelerator.prepare(
            model, optimizer, dataloader_train, scheduler
        )

        for epoch in range(self.config_typed.common.max_epochs):
            model.train()
            for input_ids, conditions, token_mask, condition_mask in tqdm(
                dataloader_train, desc=f"Epoch {epoch}, Train Batch"
            ):
                optimizer.zero_grad()
                prediction, prediction_noise, latents_noise = model.forward(
                    input_ids, conditions, token_mask, condition_mask
                )
                loss = loss_fn(
                    prediction, prediction_noise, input_ids[:, 1:], latents_noise
                )
                accelerator.backward(loss)
                optimizer.step()
                train_loss = loss.detach()

            model.eval()
            for input_ids, conditions, token_mask, condition_mask in tqdm(
                dataloader_test, desc=f"Epoch {epoch}, Test Batch"
            ):
                with torch.no_grad():
                    prediction, prediction_noise, latents_noise = model.forward(
                        input_ids,
                        conditions,
                        token_mask,
                        condition_mask,
                    )
                    loss = loss_fn(
                        prediction, prediction_noise, input_ids[:, 1:], latents_noise
                    )
                test_loss = loss.detach()

            optimizer.step()
            scheduler.step(epoch)

            os.makedirs(os.path.join(checkpoint, "/dist/"), exist_ok=True)
            model.save_pretrained(os.path.join(checkpoint, "/dist/"))
            self.config_typed.to_json(
                os.path.join(checkpoint, "/dist/train_config.json")
            )
            accelerator.save_state(checkpoint)
