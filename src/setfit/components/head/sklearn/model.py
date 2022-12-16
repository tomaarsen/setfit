from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Union

import numpy as np
from torch import nn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from setfit.components.head.sklearn.training_args import SklearnHeadTrainingArguments

from ..api import SetFitHeadI


@dataclass
class SetFitSklearnHead(SetFitHeadI):
    clf: LogisticRegression = None
    multi_target_strategy: str = None
    # config: SklearnHeadConfig = field(default_factory=SklearnHeadConfig, repr=False)

    def __post_init__(self):
        if self.multi_target_strategy is not None:
            multi_target_mapping = {
                "one-vs-rest": OneVsRestClassifier,
                "multi-output": MultiOutputClassifier,
                "classifier-chain": ClassifierChain,
            }
            if self.multi_target_strategy in multi_target_mapping:
                self.clf = multi_target_mapping[self.multi_target_strategy](self.clf)
            else:
                raise ValueError(
                    f"multi_target_strategy {self.multi_target_strategy!r} is not supported. "
                    f"Choose one from {list(multi_target_mapping.keys())!r}."
                )

    def to(self, device: Union[str, torch.device]) -> SetFitSklearnHead:
        return self

    @property
    def args_class(self) -> SklearnHeadTrainingArguments:
        return SklearnHeadTrainingArguments

    def fit(self, embeddings: torch.Tensor, y_train: List[int]) -> SetFitSklearnHead:
        embeddings = embeddings.detach().cpu().numpy()
        self.clf.fit(embeddings, y_train)
        return self

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        X = inputs.detach().cpu().numpy()
        y_pred: np.ndarray = self.clf.predict(X)
        return torch.Tensor(y_pred)

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        X = inputs.detach().cpu().numpy()
        y_pred: np.ndarray = self.clf.predict_proba(X)
        return torch.Tensor(y_pred)

    @property
    def device(self) -> Optional[Union[str, torch.device]]:
        return None

    def parameters(self) -> Iterator[nn.Parameter]:
        return []

    def train(self, mode: bool = True) -> None:
        pass