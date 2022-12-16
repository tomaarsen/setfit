import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from setfit.components.body.api import SetFitBodyI
from setfit.components.body.sentence_transformer.model import SetFitSTBody
from setfit.components.head.api import SetFitHeadI
from setfit.components.head.linear.model import SetFitLinearHead
from setfit.components.head.logistic.model import SetFitLogisticHead
from setfit.components.head.sklearn.model import SetFitSklearnHead


# Google Colab runs on Python 3.7, so we need this to be compatible
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import joblib
import numpy as np
import requests
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from sentence_transformers import InputExample, SentenceTransformer, models
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .. import logging
from ..data import SetFitDataset


if TYPE_CHECKING:
    from numpy import ndarray


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MODEL_HEAD_NAME = "model_head.pkl"


@dataclass
class SetFitModel(PyTorchModelHubMixin):
    """A SetFit model with integration to the Hugging Face Hub."""

    body: SetFitBodyI
    head: SetFitHeadI

    # def __init__(
    #     self,
    #     body: SetFitBodyI,
    #     head: SetFitHeadI,
    #     # multi_target_strategy: Optional[str] = None,
    #     # l2_weight: float = 1e-2,
    #     # normalize_embeddings: bool = False,
    # ) -> None:
    #     super(SetFitModel, self).__init__()
    #     self.body = body
    #     self.head = head

    """
        self.multi_target_strategy = multi_target_strategy
        self.l2_weight = l2_weight

        self.normalize_embeddings = normalize_embeddings

    def fit(
        self,
        x_train: List[str],
        y_train: List[int],
        num_epochs: int,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        body_learning_rate: Optional[float] = None,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
    ) -> None:
        if isinstance(self.head, nn.Module):  # train with pyTorch
            device = self.body.device
            self.body.train()
            self.head.train()

            dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
            criterion = self.head.get_loss_fn()
            optimizer = self._prepare_optimizer(learning_rate, body_learning_rate, l2_weight)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            for epoch_idx in tqdm(range(num_epochs), desc="Epoch", disable=not show_progress_bar):
                for batch in dataloader:
                    features, labels = batch
                    optimizer.zero_grad()

                    # to model's device
                    features = {k: v.to(device) for k, v in features.items()}
                    labels = labels.to(device)

                    outputs = self.body(features)
                    if self.normalize_embeddings:
                        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
                    outputs = self.head(outputs)
                    predictions = outputs["prediction"]

                    loss = criterion(predictions, labels)
                    loss.backward()
                    optimizer.step()

                scheduler.step()
        else:  # train with sklearn
            embeddings = self.body.encode(x_train, normalize_embeddings=self.normalize_embeddings)
            self.head.fit(embeddings, y_train)
    """

    def fit(
        self,
        x_train: List[str],
        y_train: List[int],
        loss_class: Optional[nn.Module] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        body_learning_rate: Optional[float] = None,
        max_length: Optional[int] = None,
        l2_weight: Optional[float] = None,
        normalize_embeddings: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
        **kwargs,
    ) -> None:
        device = self.device
        dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
        loss_func = loss_class()
        optimizer = self._prepare_optimizer(learning_rate, body_learning_rate, l2_weight)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        for _ in tqdm(range(num_epochs), desc="Epoch", disable=not show_progress_bar):
            for batch in dataloader:
                features: Dict[str, torch.Tensor]
                labels: torch.Tensor
                features, labels = batch
                optimizer.zero_grad()

                # to model's device
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)

                # TODO: Performing ["sentence_embedding"] won't work with different bodies
                embedding = self.body(features)["sentence_embedding"]
                if normalize_embeddings:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                outputs = self.head(embedding)
                predictions = outputs# ["prediction"]

                loss: torch.Tensor = loss_func(predictions, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

    def _prepare_dataloader(
        self,
        x_train: List[str],
        y_train: List[int],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        shuffle: bool = True,
    ) -> DataLoader:
        max_acceptable_length = self.body.max_seq_length
        if max_length is None:
            max_length = max_acceptable_length
            logger.warning(
                f"The `max_length` is `None`. Using the maximum acceptable length according to the current model body: {max_length}."
            )

        if max_length > max_acceptable_length:
            logger.warning(
                (
                    f"The specified `max_length`: {max_length} is greater than the maximum length of the current model body: {max_acceptable_length}. "
                    f"Using {max_acceptable_length} instead."
                )
            )
            max_length = max_acceptable_length

        dataset = SetFitDataset(
            x_train,
            y_train,
            tokenizer=self.body.tokenizer, # TODO: Maybe pass this tokenizer some other way?
            max_length=max_length,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=SetFitDataset.collate_fn, shuffle=shuffle, pin_memory=True
        )

        return dataloader

    def _prepare_optimizer(
        self,
        learning_rate: float,
        body_learning_rate: Optional[float],
        l2_weight: float,
    ) -> torch.optim.Optimizer:
        body_learning_rate = body_learning_rate or learning_rate
        l2_weight = l2_weight or self.l2_weight
        optimizer = torch.optim.AdamW(
            [
                {"params": self.body.parameters(), "lr": body_learning_rate, "weight_decay": l2_weight},
                {"params": self.head.parameters(), "lr": learning_rate, "weight_decay": l2_weight},
            ],
        )

        return optimizer

    def freeze(self, component: Optional[Literal["body", "head"]] = None) -> None:
        if component is None or component == "body":
            self._freeze_or_not(self.body, to_freeze=True)

        if component is None or component == "head":
            self._freeze_or_not(self.head, to_freeze=True)

    def unfreeze(self, component: Optional[Literal["body", "head"]] = None) -> None:
        if component is None or component == "body":
            self._freeze_or_not(self.body, to_freeze=False)

        if component is None or component == "head":
            self._freeze_or_not(self.head, to_freeze=False)

    def _freeze_or_not(self, model: torch.nn.Module, to_freeze: bool) -> None:
        for param in model.parameters():
            param.requires_grad = not to_freeze

    def encode(self, inputs: List[str]) -> torch.Tensor:
        return self.body.encode(inputs)

    def predict(self, inputs: List[str]) -> torch.Tensor:
        embeddings = self.body.encode(inputs)
        return self.head.predict(embeddings)

    def predict_proba(self, inputs: List[str]) -> torch.Tensor:
        embeddings = self.body.encode(inputs)
        return self.head.predict_proba(embeddings)

    @property
    def device(self) -> Optional[Union[str, torch.device]]:
        return self.body.device

    def __call__(self, inputs):
        return self.predict(inputs)

    def _save_pretrained(self, save_directory: str) -> None:
        self.body.save(path=save_directory)
        joblib.dump(self.head, f"{save_directory}/{MODEL_HEAD_NAME}")

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        multi_target_strategy: Optional[str] = None,
        # use_differentiable_head: bool = False,
        head: Optional[SetFitHeadI] = None,
        # normalize_embeddings: bool = False,
        **model_kwargs,
    ) -> "SetFitModel":
        st = SentenceTransformer(model_id, cache_folder=cache_dir)
        model_body = SetFitSTBody(st=st)

        if os.path.isdir(model_id):
            if MODEL_HEAD_NAME in os.listdir(model_id):
                model_head_file = os.path.join(model_id, MODEL_HEAD_NAME)
            else:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found in {Path(model_id).resolve()},"
                    " initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None
        else:
            try:
                model_head_file = hf_hub_download(
                    repo_id=model_id,
                    filename=MODEL_HEAD_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found on HuggingFace Hub, initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None

        if model_head_file is not None:
            model_head = joblib.load(model_head_file)
        elif head:
            model_head = head
        else:
            model_head = SetFitLogisticHead(multi_target_strategy=multi_target_strategy)

        return SetFitModel(
            body=model_body,
            head=model_head,
            # multi_target_strategy=multi_target_strategy,
            # normalize_embeddings=normalize_embeddings,
        )


class SKLearnWrapper:
    def __init__(self, st_model=None, clf=None):
        self.st_model = st_model
        self.clf = clf

    def fit(self, x_train, y_train):
        embeddings = self.st_model.encode(x_train)
        self.clf.fit(embeddings, y_train)

    def predict(self, x_test):
        embeddings = self.st_model.encode(x_test)
        return self.clf.predict(embeddings)

    def predict_proba(self, x_test):
        embeddings = self.st_model.encode(x_test)
        return self.clf.predict_proba(embeddings)

    def save(self, path):
        self.st_model.save(path=path)
        joblib.dump(self.clf, f"{path}/setfit_head.pkl")

    def load(self, path):
        self.st_model = SentenceTransformer(model_name_or_path=path)
        self.clf = joblib.load(f"{path}/setfit_head.pkl")
