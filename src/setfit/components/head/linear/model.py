from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from tqdm.auto import tqdm

import torch
from sentence_transformers import models
from torch import nn

from setfit.components.head.linear.training_args import LinearHeadTrainingArguments

from ..api import SetFitHeadI


if TYPE_CHECKING:
    from numpy import ndarray


class SetFitLinearHead(SetFitHeadI, models.Dense):
    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 1,
        temperature: float = 1.0,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        super(models.Dense, self).__init__()  # init on models.Dense's parent: nn.Module

        if in_features is not None:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.LazyLinear(out_features, bias=bias)

        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.bias = bias
        target_device = device or "cuda" if torch.cuda.is_available() else "cpu"

        self.to(target_device)
        # self.apply(self._init_weight)

    def forward(
        self, features: Union[Dict[str, torch.Tensor], torch.Tensor], temperature: Optional[float] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        SetFitHead can accept embeddings in:
        1. Output format (`dict`) from Sentence-Transformers.
        2. Pure `torch.Tensor`.

        Args:
            features (`Dict[str, torch.Tensor]` or `torch.Tensor):
                The embeddings from the encoder. If using `dict` format,
                make sure to store embeddings under the key: 'sentence_embedding'
                and the outputs will be under the key: 'prediction'.
            temperature (`float`, *optional*):
                A logits' scaling factor when using multi-targets (i.e., number of targets more than 1).
                Will override the temperature given during initialization.
        Returns:
        [`Dict[str, torch.Tensor]` or `torch.Tensor`]
        """
        is_features_dict = False  # whether `features` is dict or not
        if isinstance(features, dict):
            assert "sentence_embedding" in features
            is_features_dict = True

        x = features["sentence_embedding"] if is_features_dict else features
        logits = self.linear(x)
        if self.out_features == 1:  # only has one target
            outputs = torch.sigmoid(logits)
        else:  # multiple targets
            temperature = temperature or self.temperature
            outputs = nn.functional.softmax(logits / temperature, dim=-1)

        if is_features_dict:
            features.update({"prediction": outputs})
            return features

        return outputs

    def predict_proba(self, x_test: torch.Tensor) -> torch.Tensor:
        self.eval()

        return self(x_test)

    def predict(self, x_test: Union[torch.Tensor, "ndarray"]) -> Union[torch.Tensor, "ndarray"]:
        is_tensor = isinstance(x_test, torch.Tensor)
        if not is_tensor:  # then assume it's ndarray
            x_test = torch.Tensor(x_test).to(self.device)

        probs = self.predict_proba(x_test)

        if self.out_features == 1:
            out = torch.where(probs >= 0.5, 1, 0)
        else:
            out = torch.argmax(probs, dim=-1)

        if not is_tensor:
            return out.cpu().numpy()

        return out

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the model is placed.

        Reference from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L869
        """
        return next(self.parameters()).device

    def get_config_dict(self) -> Dict[str, Optional[Union[int, float, bool]]]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "temperature": self.temperature,
            "bias": self.bias,
            "device": self.device.type,  # store the string of the device, instead of `torch.device`
        }

    @staticmethod
    def _init_weight(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 1e-2)

    @property
    def args_class(self) -> LinearHeadTrainingArguments:
        return LinearHeadTrainingArguments

    def fit(self):
        raise NotImplementedError()

    def to(self, device: Union[str, torch.device]) -> SetFitLinearHead:
        return super(models.Dense, self).to(device)

    def train(self, mode: bool = True):
        return super(models.Dense, self).train(mode)

    def parameters(self) -> Iterator[nn.Parameter]:
        return super(models.Dense, self).parameters()

    def __hash__(self):
        return super(models.Dense, self).__hash__()
