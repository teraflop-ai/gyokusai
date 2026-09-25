from typing import Optional

from daft import DataFrame, col
from daft import functions as F
from huggingface_hub import hf_hub_download

from .classifiers import LanguagePredictor, NSFWPredictor, WhichlangPredictor
from .factories import attribute_model_factory
from .regexes import FASTTEXT_REGEX


class Whichlang:
    def __init__(self, input_column: str = "text", output_column: str = "language"):
        self.input_column = input_column
        self.output_column = output_column
        self.predictor = WhichlangPredictor()

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.predictor.predict(col(self.input_column))
        )


class NSFW:
    def __init__(
        self,
        model_repo_id="allenai/dolma-jigsaw-fasttext-bigrams-nsfw",
        model_filename="model.bin",
        input_column: str = "text",
        output_column: str = "nsfw_score",
    ):
        model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
        self.input_column = input_column
        self.output_column = output_column
        self.predictor = NSFWPredictor(model_path)

    def __call__(self, df: DataFrame) -> DataFrame:
        cleaned = F.strip(F.regexp_replace(col(self.input_column), FASTTEXT_REGEX, " "))
        return df.with_column(self.output_column, self.predictor.predict(cleaned))


class LanguageID:
    def __init__(
        self,
        model_repo_id="cis-lmu/glotlid",
        model_filename="model_v3.bin",
        input_column: str = "text",
        output_column: str = "language",
    ):
        model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
        self.input_column = input_column
        self.output_column = output_column
        self.predictor = LanguagePredictor(model_path)

    def __call__(self, df: DataFrame) -> DataFrame:
        cleaned = F.strip(F.regexp_replace(col(self.input_column), FASTTEXT_REGEX, " "))
        return df.with_column(self.output_column, self.predictor.predict(cleaned))


class AttributeModel:
    def __init__(
        self,
        input_column: str = "text_embedding",
        output_column: str = "attribute_model_score",
        *,
        checkpoint_path: str,
        label: str,
        task: str,
        batch_size: int = 32,
        gpus: int | float = 0,
        cpus: Optional[float] = None,
        max_concurrency: Optional[int] = None,
        name: str = "AttributeModel",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.name = name
        self.score_batch = attribute_model_factory(
            checkpoint_path=checkpoint_path,
            label=label,
            task=task,
            batch_size=batch_size,
            gpus=gpus,
            cpus=cpus,
            max_concurrency=max_concurrency,
        )

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column,
            self.score_batch(col(self.input_column)),
        )
