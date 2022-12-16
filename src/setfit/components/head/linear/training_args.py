from dataclasses import dataclass
from typing import Callable
import torch
from ..training_args import HeadTrainingArguments


@dataclass
class LinearHeadTrainingArguments(HeadTrainingArguments):
    end_to_end: bool = True

    num_epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 2e-5
    # None results in the maximum acceptable length being used
    max_length: int = None
    l2_weight: float = 1e-2
    loss_class: Callable = torch.nn.CrossEntropyLoss
    normalize_embeddings: bool = False
    show_progress_bar: bool = True
