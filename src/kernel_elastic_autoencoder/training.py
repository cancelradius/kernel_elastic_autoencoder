import os
from typing import Protocol

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from kernel_elastic_autoencoder.collate import Collated
from kernel_elastic_autoencoder.config import TrainingConfig
from kernel_elastic_autoencoder.losses import Loss
from kernel_elastic_autoencoder.model import Model


def _is_local():
    return (
        dist.is_torchelastic_launched() and (dist.get_rank() == 0)
    ) or not dist.is_torchelastic_launched()


class TrainerCallbackCtx(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dist: bool
    device_type: str
    local_rank: int | None
    model: Model | DDP
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    dataloader_train: torch.utils.data.DataLoader
    dataloader_test: torch.utils.data.DataLoader

    epoch: int | None = None
    rel_epoch: int | None = None
    batch: int | None = None
    train_loss: torch.Tensor | None = None
    test_loss: torch.Tensor | None = None


class TrainerCallback(Protocol):
    """Protocol to be implemented for callback classes to a Trainer.

    On each hook, except before the Trainer is initialized, a TrainerCallbackCtx schema is passed with the appropriate
    information passed.
    """

    def before_init(self):
        """Called before training setup."""
        ...

    def after_init(self, ctx: TrainerCallbackCtx):
        """Called after training setup."""
        ...

    def before_epoch(self, ctx: TrainerCallbackCtx):
        """Called before each epoch."""
        ...

    def before_train_batch(self, ctx: TrainerCallbackCtx):
        """Called before each forward pass of a single training batch."""
        ...

    def after_train_batch(self, ctx: TrainerCallbackCtx):
        """Called after each forward pass of a single training batch."""
        ...

    def before_test_batch(self, ctx: TrainerCallbackCtx):
        """Called before each forward pass of a single test batch."""
        ...

    def after_test_batch(self, ctx: TrainerCallbackCtx):
        """Called after each forward pass of a single test batch."""
        ...

    def after_epoch(self, ctx: TrainerCallbackCtx):
        """Called after each epoch."""
        ...

    def after_training(self, ctx: TrainerCallbackCtx):
        """Called after training ends."""
        ...


class TrainerDefaultCallback:
    # TODO: Implement sensible default logging.
    def before_init(self):
        pass

    def after_init(self, ctx: TrainerCallbackCtx):
        pass

    def before_epoch(self, ctx: TrainerCallbackCtx):
        pass

    def before_train_batch(self, ctx: TrainerCallbackCtx):
        pass

    def after_train_batch(self, ctx: TrainerCallbackCtx):
        pass

    def before_test_batch(self, ctx: TrainerCallbackCtx):
        pass

    def after_test_batch(self, ctx: TrainerCallbackCtx):
        pass

    def after_epoch(self, ctx: TrainerCallbackCtx):
        pass

    def after_training(self, ctx: TrainerCallbackCtx):
        pass


class Trainer:
    def __init__(
        self,
        config: dict | TrainingConfig,
        callbacks: tuple[TrainerCallback] = (TrainerDefaultCallback(),),
    ):
        """Instantiates a Trainer object.

        Args:
            config: Dictionary or TrainingConfig schema defining model parameters. Will be validated with TrainingConfig
                regardless of input type.
            callbacks: Tuple of callback classes implementing TrainerCallback.
        """
        self.config = config
        """Configuration object for Hugging Face Hub compatible serialization. Not recommended to use, as it is 
                internal-use. Use Trainer.config_typed instead."""
        config_typed = TrainingConfig.model_validate(config)
        self.config_typed: TrainingConfig = config_typed
        """Type-validated config in a Pydantic TrainingConfig schema, recommended for public API use."""
        self.callbacks = callbacks
        """Tuple of callback classes implementing TrainerCallback."""

        self._ctx: TrainerCallbackCtx

        self._model: DDP | Model
        self._dist: bool
        self._loss_fn: nn.Module
        self._optimizer: torch.optim.Optimizer
        self._scheduler: torch.optim.lr_scheduler.LRScheduler
        self._dataloader_train: torch.utils.data.DataLoader
        self._dataloader_test: torch.utils.data.DataLoader
        self._device_type: str
        self._local_rank: int | None

    def train(
        self,
        model: Model,
        ds: Collated,
        train_split: float = 0.9,
        checkpoint: str = "./checkpoint",
    ):
        """Trains a model with a Collated dataset. Optionally resumes from an existing checkpoint.

        Args:
            model: Freshly instantiated model.
            ds: Tensor dataset following the Collated schema.
            train_split: Fraction of dataset used for training. Must be between 0 and 1.
            checkpoint: Path of local checkpoint to be saved and/or resumed.
        """
        next_epoch = self._setup(model, ds, train_split, checkpoint)
        for epoch in range(next_epoch, self.config_typed.common.max_epochs):
            self._ctx.rel_epoch = epoch - next_epoch
            self._epoch(epoch, checkpoint)
        self._ctx.epoch = None
        self._ctx.rel_epoch = None
        if _is_local():
            for cb in self.callbacks:
                cb.after_training(self._ctx)

    def _setup(
        self,
        model: Model,
        ds: Collated,
        train_split: float,
        checkpoint: str,
    ):
        if _is_local():
            for cb in self.callbacks:
                cb.before_init()
        self._model = model
        self._dist = dist.is_torchelastic_launched()
        if self._dist:
            self._model, self._device_type, self._local_rank = self._setup_ddp()
        else:
            self._device_type, self._local_rank = self._setup_no_ddp()
        self._loss_fn = self._setup_loss(model)
        self._optimizer, self._scheduler = self._setup_optimizer()
        self._dataloader_train, self._dataloader_test = self._setup_dataloaders(
            ds, train_split
        )
        next_epoch = 0
        if os.path.exists(checkpoint):
            next_epoch = self._resume(checkpoint)
        self._ctx = TrainerCallbackCtx(
            dist=self._dist,
            device_type=self._device_type,
            local_rank=self._local_rank,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            dataloader_train=self._dataloader_train,
            dataloader_test=self._dataloader_test,
        )
        if _is_local():
            for cb in self.callbacks:
                cb.after_init(self._ctx)
        return next_epoch

    def _setup_loss(self, model: Model):
        _loss_fn = Loss(
            hp_lambda=self.config_typed.hyperparameters.hp_lambda,
            hp_delta=self.config_typed.hyperparameters.hp_delta,
            hp_sigma=self.config_typed.hyperparameters.hp_sigma,
            kernel_dist_size=self.config_typed.hyperparameters.kernel_dist_size,
            padding_idx=model.config_typed.common.padding_idx,
            embedding_dim=model.config_typed.common.embedding_dim,
            pooling_dim=model.config_typed.common.pooling_dim,
        )
        return _loss_fn

    def _setup_optimizer(self):
        optimizer = self.config_typed.optimizer.optimizer_fn(
            self._model.parameters(), **self.config_typed.optimizer.optimizer_params
        )
        scheduler = self.config_typed.optimizer.scheduler_fn(
            optimizer, **self.config_typed.optimizer.scheduler_params
        )
        return optimizer, scheduler

    def _setup_ddp(self) -> tuple[DDP, str, int]:
        device_type, vendor_backend = self._get_backend()
        dist.init_process_group(backend=vendor_backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(self._model.to(local_rank))
        return model, device_type, local_rank

    def _setup_no_ddp(self):
        device_type, _ = self._get_backend()
        return device_type, None

    def _setup_dataloaders(self, ds: Collated, train_split: float):
        dataset = torch.utils.data.TensorDataset(
            ds.input_ids, ds.conditions, ds.token_mask, ds.condition_mask
        )
        dataset_train, dataset_test = torch.utils.data.random_split(
            dataset, [train_split, 1 - train_split]
        )
        if self._dist:
            sampler_train = DistributedSampler(dataset_train)
            sampler_test = DistributedSampler(dataset_test)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        _dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.config_typed.common.batch_size,
            sampler=sampler_train,
            pin_memory=True,
        )
        _dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.config_typed.common.batch_size,
            sampler=sampler_test,
            pin_memory=True,
        )
        return _dataloader_train, _dataloader_test

    def _resume(
        self,
        checkpoint: str,
    ):
        train_state = torch.load(os.path.join(checkpoint, "train_state.pt"))
        self._optimizer.load_state_dict(train_state["optimizer"])
        self._scheduler.load_state_dict(train_state["scheduler"])
        self._model.from_pretrained(checkpoint)  # type: ignore
        return train_state["next_epoch"]

    def _epoch(
        self,
        epoch: int,
        checkpoint: str,
    ):
        self._ctx.epoch = epoch
        if _is_local():
            for cb in self.callbacks:
                cb.before_epoch(self._ctx)

        self._model.train()
        self._ctx.batch = 0
        for input_ids, conditions, token_mask, condition_mask in self._dataloader_train:
            if _is_local():
                for cb in self.callbacks:
                    cb.before_train_batch(self._ctx)
            self._ctx.train_loss = self._batch_train(
                input_ids, conditions, token_mask, condition_mask
            )
            if _is_local():
                for cb in self.callbacks:
                    cb.after_train_batch(self._ctx)
            self._ctx.batch += 1

        self._model.eval()
        self._ctx.batch = 0
        for input_ids, conditions, token_mask, condition_mask in self._dataloader_test:
            if _is_local():
                for cb in self.callbacks:
                    cb.before_test_batch(self._ctx)
            self._ctx.test_loss = self._batch_test(
                input_ids, conditions, token_mask, condition_mask
            )
            if _is_local():
                for cb in self.callbacks:
                    cb.after_test_batch(self._ctx)
            self._ctx.batch += 1

        self._end_epoch(epoch, checkpoint)
        self._ctx.train_loss = None
        self._ctx.test_loss = None
        self._ctx.batch = None
        if _is_local():
            for cb in self.callbacks:
                cb.after_epoch(self._ctx)

    def _end_epoch(self, epoch: int, checkpoint: str):
        self._scheduler.step(epoch)
        self._save_state(checkpoint, epoch)

    def _batch_train(self, input_ids, conditions, token_mask, condition_mask):
        self._optimizer.zero_grad()
        with torch.amp.autocast(self._device_type):
            prediction, prediction_noise, latents_noise = self._model.forward(
                input_ids, conditions, token_mask, condition_mask
            )
            loss = self._loss_fn(
                prediction, prediction_noise, input_ids[:, 1:], latents_noise
            )
        loss.backward()
        self._optimizer.step()
        return loss.detach()

    def _batch_test(self, input_ids, conditions, token_mask, condition_mask):
        with torch.no_grad():
            with torch.amp.autocast(self._device_type):
                prediction, prediction_noise, latents_noise = self._model.forward(
                    input_ids,
                    conditions,
                    token_mask,
                    condition_mask,
                )
            loss = self._loss_fn(
                prediction, prediction_noise, input_ids[:, 1:], latents_noise
            )
        return loss.detach()

    def _save_state(self, checkpoint: str, epoch: int):
        if _is_local():
            os.makedirs(checkpoint, exist_ok=True)
            self._model.save_pretrained(checkpoint)  # type: ignore
            torch.save(
                {
                    "next_epoch": epoch + 1,
                    "optimizer": self._optimizer.state_dict(),
                    "scheduler": self._scheduler.state_dict(),
                },
                os.path.join(checkpoint, "train_state.pt"),
            )
            self.config_typed.to_json(os.path.join(checkpoint, "train_config.json"))

    def _get_backend(self) -> tuple[str, str]:
        if torch.accelerator.is_available():
            device_type = torch.accelerator.current_accelerator().type  # type: ignore
            vendor_backend = torch.distributed.get_default_backend_for_device(
                device_type
            )

        else:
            device_type = torch.device("cpu").type
            vendor_backend = torch.distributed.get_default_backend_for_device(
                device_type
            )

        return device_type, vendor_backend
