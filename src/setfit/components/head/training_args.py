from __future__ import annotations
from dataclasses import dataclass

from setfit.training_args import TrainingArguments


@dataclass
class HeadTrainingArguments(TrainingArguments):
    end_to_end: bool

    @classmethod
    def classifier_default(cls) -> HeadTrainingArguments:
        return cls()
