__version__ = "0.8.0.dev0"

import warnings

from .data import get_templated_dataset, sample_dataset
from .modeling import SetFitHead, SetFitModel
from .trainer import SetFitTrainer, Trainer
from .trainer_distillation import DistillationSetFitTrainer, DistillationTrainer
from .training_args import TrainingArguments
from .span import AspectExtractor, AbsaModel, AspectModel, PolarityModel, AbsaTrainer

# Ensure that DeprecationWarnings are shown by default, as recommended by
# https://docs.python.org/3/library/warnings.html#overriding-the-default-filter
warnings.filterwarnings("default", category=DeprecationWarning)
