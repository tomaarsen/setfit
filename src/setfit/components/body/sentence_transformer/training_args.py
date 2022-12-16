from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sentence_transformers import losses
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch import nn

from ..training_args import BodyTrainingArguments


@dataclass
class STBodyTrainingArguments(BodyTrainingArguments):
    num_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5

    num_iterations: int = 20
    distance_metric: Callable = BatchHardTripletLossDistanceFunction.cosine_distance
    margin: float = 0.25
    samples_per_label: int = 2

    warmup_proportion: float = 0.1
    use_amp: bool = False

    # This one is not appropriate:
    multi_target: bool = False
    loss_class: nn.Module = losses.CosineSimilarityLoss

    show_progress_bar: bool = True

    @classmethod
    def embeddings_default(cls) -> STBodyTrainingArguments:
        return cls()

    @classmethod
    def classifier_default(cls) -> STBodyTrainingArguments:
        return cls(
            num_epochs=25,
        )
