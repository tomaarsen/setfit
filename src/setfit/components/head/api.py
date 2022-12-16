from abc import abstractmethod, abstractproperty

import torch

from setfit.components.api import SetFitI
from setfit.components.head.training_args import HeadTrainingArguments


class SetFitHeadI(SetFitI):
    @abstractproperty
    def args_class(self) -> HeadTrainingArguments:
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    # def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
    #     return self.predict(inputs)
