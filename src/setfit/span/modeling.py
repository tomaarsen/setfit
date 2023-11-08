import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import requests
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import SoftTemporaryDirectory, validate_hf_hub_args

from .. import logging
from ..modeling import SetFitModel
from .aspect_extractor import AspectExtractor


if TYPE_CHECKING:
    from spacy.tokens import Doc

logger = logging.get_logger(__name__)

CONFIG_NAME = "config_span_setfit.json"


@dataclass
class SpanSetFitModel(SetFitModel):
    span_context: int = 0

    def prepend_aspects(self, docs: List["Doc"], aspects_list: List[List[slice]]) -> List[str]:
        for doc, aspects in zip(docs, aspects_list):
            for aspect_slice in aspects:
                aspect = doc[max(aspect_slice.start - self.span_context, 0) : aspect_slice.stop + self.span_context]
                # TODO: Investigate performance difference of different formats
                yield aspect.text + ":" + doc.text

    def __call__(self, docs: List["Doc"], aspects_list: List[List[slice]]) -> List[bool]:
        # TODO: Use configured value for context instead
        inputs_list = list(self.prepend_aspects(docs, aspects_list))
        preds = self.predict(inputs_list, as_numpy=True)
        iter_preds = iter(preds)
        return [[next(iter_preds) for _ in aspects] for aspects in aspects_list]

    @classmethod
    @validate_hf_hub_args
    def _from_pretrained(
        cls,
        model_id: str,
        span_context: Optional[int] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        **model_kwargs,
    ) -> "SpanSetFitModel":
        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                pass

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update(config)

        if span_context is not None:
            model_kwargs["span_context"] = span_context

        return super(SpanSetFitModel, cls)._from_pretrained(
            model_id,
            revision,
            cache_dir,
            force_download,
            proxies,
            resume_download,
            local_files_only,
            token,
            **model_kwargs,
        )

    def _save_pretrained(self, save_directory: Union[Path, str]) -> None:
        path = os.path.join(save_directory, CONFIG_NAME)
        with open(path, "w") as f:
            json.dump({"span_context": self.span_context}, f, indent=2)

        super()._save_pretrained(save_directory)


class AspectModel(SpanSetFitModel):
    # TODO: Assumes binary SetFitModel with 0 == no aspect, 1 == aspect
    def __call__(self, docs: List["Doc"], aspects_list: List[List[slice]]) -> List[bool]:
        sentence_preds = super().__call__(docs, aspects_list)
        return [
            [aspect for aspect, pred in zip(aspects, preds) if pred == 1]
            for aspects, preds in zip(aspects_list, sentence_preds)
        ]


@dataclass
class PolarityModel(SpanSetFitModel):
    span_context: int = 3


@dataclass
class AbsaModel:
    aspect_extractor: AspectExtractor
    aspect_model: AspectModel
    polarity_model: PolarityModel

    def predict(self, inputs: Union[str, List[str]]) -> List[Dict[str, Any]]:
        is_str = isinstance(inputs, str)
        inputs_list = [inputs] if is_str else inputs
        docs, aspects_list = self.aspect_extractor(inputs_list)
        aspects_list = self.aspect_model(docs, aspects_list)
        polarity_list = self.polarity_model(docs, aspects_list)
        outputs = []
        for docs, aspects, polarities in zip(docs, aspects_list, polarity_list):
            outputs.append(
                [
                    {"span": docs[aspect_slice].text, "polarity": polarity}
                    for aspect_slice, polarity in zip(aspects, polarities)
                ]
            )
        return outputs if not is_str else outputs[0]

    @property
    def device(self) -> torch.device:
        return self.aspect_model.device

    def to(self, device: Union[str, torch.device]) -> "AbsaModel":
        self.aspect_model.to(device)
        self.polarity_model.to(device)

    def __call__(self, inputs: Union[str, List[str]]) -> List[Dict[str, Any]]:
        return self.predict(inputs)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        polarity_save_directory: Optional[Union[str, Path]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        if polarity_save_directory is None:
            base_save_directory = Path(save_directory)
            save_directory = base_save_directory.parent / (base_save_directory.name + "-aspect")
            polarity_save_directory = base_save_directory.parent / (base_save_directory.name + "-polarity")
        self.aspect_model.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        self.polarity_model.save_pretrained(save_directory=polarity_save_directory, push_to_hub=push_to_hub, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        polarity_model_id: Optional[str] = None,
        spacy_model: Optional[str] = "en_core_web_lg",
        span_contexts: Tuple[Optional[int], Optional[int]] = (None, None),
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        use_differentiable_head: bool = False,
        normalize_embeddings: bool = False,
        **model_kwargs,
    ) -> "AbsaModel":
        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")
        aspect_model = AspectModel.from_pretrained(
            model_id,
            span_context=span_contexts[0],
            revision=revision,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_differentiable_head=use_differentiable_head,
            normalize_embeddings=normalize_embeddings,
            **model_kwargs,
        )
        if polarity_model_id:
            model_id = polarity_model_id
            revision = None
            if len(model_id.split("@")) == 2:
                model_id, revision = model_id.split("@")
        polarity_model = PolarityModel.from_pretrained(
            model_id,
            span_context=span_contexts[1],
            revision=revision,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_differentiable_head=use_differentiable_head,
            normalize_embeddings=normalize_embeddings,
            **model_kwargs,
        )

        aspect_extractor = AspectExtractor(spacy_model=spacy_model)

        return cls(aspect_extractor, aspect_model, polarity_model)

    def push_to_hub(self, repo_id: str, polarity_repo_id: Optional[str] = None, **kwargs) -> None:
        if "/" not in repo_id:
            raise ValueError(
                '`repo_id` must be a full repository ID, including organisation, e.g. "tomaarsen/setfit-absa-restaurant".'
            )
        if polarity_repo_id is not None and "/" not in polarity_repo_id:
            raise ValueError(
                '`polarity_repo_id` must be a full repository ID, including organisation, e.g. "tomaarsen/setfit-absa-restaurant".'
            )
        commit_message = kwargs.pop("commit_message", "Add SetFit ABSA model")

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp_dir:
            save_directory = Path(tmp_dir) / repo_id
            polarity_save_directory = None if polarity_repo_id is None else Path(tmp_dir) / polarity_repo_id
            self.save_pretrained(
                save_directory=save_directory,
                polarity_save_directory=polarity_save_directory,
                push_to_hub=True,
                commit_message=commit_message,
                **kwargs,
            )
