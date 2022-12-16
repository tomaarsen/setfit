from abc import abstractclassmethod

from setfit.training_args import TrainingArguments


class BodyTrainingArguments(TrainingArguments):
    @abstractclassmethod
    def embeddings_default(cls) -> TrainingArguments:
        raise NotImplementedError()
