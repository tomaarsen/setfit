from abc import abstractmethod, abstractproperty
from typing import Callable, List, TYPE_CHECKING

import torch

from setfit.components.api import SetFitI
from setfit.components.body.training_args import BodyTrainingArguments

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class SetFitBodyI(SetFitI):
    @abstractproperty
    def args_class(self) -> BodyTrainingArguments:
        pass

    @abstractmethod
    def encode(self, inputs: List[str]) -> torch.Tensor:
        pass

    @abstractproperty
    def max_seq_length(self) -> int:
        pass

    @abstractproperty
    def tokenizer(self) -> "PreTrainedTokenizer":
        pass