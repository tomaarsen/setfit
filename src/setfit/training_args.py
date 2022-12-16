from __future__ import annotations

from abc import ABC, abstractclassmethod
from dataclasses import dataclass, fields


@dataclass
class TrainingArguments(ABC):
    @abstractclassmethod
    def classifier_default(cls) -> TrainingArguments:
        raise NotImplementedError()

    def to_dict(self):
        # filter out fields that are defined as field(init=False)
        return {field.name: getattr(self, field.name) for field in fields(self) if field.init}
