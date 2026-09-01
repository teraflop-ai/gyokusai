import daft
from daft import DataFrame, col
from daft import functions as F
from fastertext import load_model
from huggingface_hub import hf_hub_download


@daft.cls
class LanguagePredictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.id2label = self.model.get_labels()

    @daft.method.batch(return_dtype=daft.DataType.string())
    def predict(self, texts: daft.Series) -> list[str]:
        labels, probs = self.model.batch(texts.to_pylist(), k=1)
        return [
            self.id2label[int(l)] if p > 0 else ""
            for l, p in zip(labels[:, 0], probs[:, 0])
        ]


class ExtractLanguage:
    def __init__(self, model_repo_id="cis-lmu/glotlid", model_filename="model_v3.bin"):
        model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
        self.predictor = LanguagePredictor(model_path)

    def __call__(self, df: DataFrame) -> DataFrame:
        cleaned = F.strip(F.regexp_replace(col("text"), r"[\r\n]+", " "))
        return df.with_column("language", self.predictor.predict(cleaned))
