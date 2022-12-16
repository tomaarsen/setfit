__version__ = "0.5.0.dev0"

from .components.modeling import SetFitModel
from .data import add_templated_examples, sample_dataset
from .trainer import Trainer
from .trainer_distillation import DistillationSetFitTrainer
