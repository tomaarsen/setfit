from dataclasses import dataclass, field
from typing import Optional

from sklearn.linear_model import LogisticRegression

from setfit.components.head.sklearn.model import SetFitSklearnHead


@dataclass
class SetFitLogisticHead(SetFitSklearnHead):
    clf: LogisticRegression = field(default_factory=LogisticRegression)
