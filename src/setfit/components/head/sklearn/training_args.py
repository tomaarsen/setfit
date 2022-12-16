from dataclasses import dataclass

from ..training_args import HeadTrainingArguments


@dataclass
class SklearnHeadTrainingArguments(HeadTrainingArguments):
    end_to_end: bool = False
