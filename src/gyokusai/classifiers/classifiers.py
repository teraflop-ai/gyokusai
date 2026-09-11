import daft

from .schemas import BaseClassifier


@daft.cls
class LanguagePredictor(BaseClassifier):
    @daft.method.batch(return_dtype=daft.DataType.string())
    def predict(self, texts: daft.Series) -> list[str]:
        labels, probs = self.model.batch(texts.to_pylist(), k=1)
        return [
            self.id2label[int(l)] if p > 0 else ""
            for l, p in zip(labels[:, 0], probs[:, 0])
        ]


@daft.cls
class NSFWPredictor(BaseClassifier):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.nsfw_id = next(i for i, l in self.id2label.items() if "non" not in l)

    @daft.method.batch(return_dtype=daft.DataType.float64())
    def predict(self, texts: daft.Series) -> list[float]:
        labels, probs = self.model.batch(texts.to_pylist(), k=1)
        return [
            (float(p) if l == self.nsfw_id else 1 - float(p)) if p > 0 else 0.0
            for l, p in zip(labels[:, 0], probs[:, 0])
        ]
