from abc import ABC, abstractmethod

import daft
from fastertext import load_model


class BaseClassifier(ABC):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.id2label = self.model.get_labels()

    @abstractmethod
    def predict(self, texts: daft.Series) -> list[str] | list[float]: ...
