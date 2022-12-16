from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader

from setfit import logging
from setfit.components.body.sentence_transformer.training_args import STBodyTrainingArguments
from setfit.components.body.sentence_transformer.utils import (
    sentence_pairs_generation,
    sentence_pairs_generation_multilabel,
)
from setfit.losses import SupConLoss

from ..api import SetFitBodyI


if TYPE_CHECKING:
    import optuna
    from torch import nn

    from transformers import PreTrainedTokenizer

logger = logging.get_logger(__name__)


@dataclass
class SetFitSTBody(SetFitBodyI):
    # config: STBodyConfig = field(default_factory=STBodyConfig, repr=False)
    st: SentenceTransformer = None

    def fit(
        self,
        x_train: List[str],
        y_train: List[int],
        multi_target: Optional[bool] = None,
        loss_class: Optional[nn.Module] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        distance_metric: Optional[Callable] = None,
        margin: Optional[float] = None,
        num_iterations: Optional[int] = None,
        learning_rate: Optional[float] = None,
        warmup_proportion: Optional[float] = None,
        show_progress_bar: Optional[bool] = None,
        use_amp: Optional[bool] = None,
        samples_per_label: Optional[int] = None,
    ):
        # sentence-transformers adaptation
        if loss_class in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:
            train_examples = [InputExample(texts=[text], label=label) for text, label in zip(x_train, y_train)]
            train_data_sampler = SentenceLabelDataset(train_examples, samples_per_label=samples_per_label)

            batch_size = min(batch_size, len(train_data_sampler))
            train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True)

            if loss_class is losses.BatchHardSoftMarginTripletLoss:
                train_loss = loss_class(
                    model=self.st,
                    distance_metric=distance_metric,
                )
            elif loss_class is SupConLoss:
                train_loss = loss_class(model=self.st)
            else:
                train_loss = loss_class(
                    model=self.st,
                    distance_metric=distance_metric,
                    margin=margin,
                )

            train_steps = len(train_dataloader) * num_epochs
        else:
            train_examples = []

            for _ in range(num_iterations):
                # TODO: multi_target_strategy might not always be defined atm
                # if self.model.head.multi_target_strategy is not None:
                if multi_target:
                    train_examples = sentence_pairs_generation_multilabel(
                        np.array(x_train), np.array(y_train), train_examples
                    )
                else:
                    train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            train_loss = loss_class(self.st)
            train_steps = len(train_dataloader) * num_epochs

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Total optimization steps = {train_steps}")
        logger.info(f"  Total train batch size = {batch_size}")

        warmup_steps = math.ceil(train_steps * warmup_proportion)
        self.st.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            steps_per_epoch=train_steps,
            optimizer_params={"lr": learning_rate},
            warmup_steps=warmup_steps,
            show_progress_bar=show_progress_bar,
            use_amp=use_amp,
        )

    def forward(self, inputs: Any) -> Any:
        return self.st.forward(inputs)

    def __call__(self, inputs: Any) -> Any:
        return self.forward(inputs)

    @classmethod
    def from_sentence_transformer(cls, st: SentenceTransformer) -> SetFitSTBody:
        return cls(st=st)

    def to(self, device: Union[str, torch.device]) -> SetFitSTBody:
        self.st.to(device)
        return self

    @property
    def args_class(self) -> STBodyTrainingArguments:
        return STBodyTrainingArguments

    def encode(self, inputs: List[str]) -> torch.Tensor:
        return self.st.encode(inputs, convert_to_tensor=True)

    @property
    def device(self) -> Optional[Union[str, torch.device]]:
        return self.st.device

    def train(self, mode: bool = True):
        self.st.train(mode)
        return self

    @property
    def max_seq_length(self) -> int:
        return self.st.get_max_seq_length()
    
    @property
    def tokenizer(self) -> "PreTrainedTokenizer":
        return self.st.tokenizer

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.st.parameters()
