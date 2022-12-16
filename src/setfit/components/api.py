from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import List, Optional, Union, Iterator

from torch import nn
import torch

from setfit.training_args import TrainingArguments


@dataclass
class SetFitI(ABC):
    # config: Config = field(repr=False)

    @abstractproperty
    def args_class(self) -> TrainingArguments:
        pass

    @abstractproperty
    def device(self) -> Optional[Union[str, torch.device]]:
        pass

    @abstractmethod
    def fit(self, x_train: List[str], y_train: List[int], **kwargs) -> None:
        pass

    @abstractmethod
    def to(self, device: Union[str, torch.device]) -> SetFitI:
        pass

    @abstractmethod
    def train(self, mode: bool = True) -> None:
        pass

    @abstractmethod
    def parameters(self) -> Iterator[nn.Parameter]:
        pass
