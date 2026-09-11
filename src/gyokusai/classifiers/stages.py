from daft import DataFrame, col
from daft import functions as F
from huggingface_hub import hf_hub_download

from .classifiers import LanguagePredictor, NSFWPredictor
from .regexes import FASTTEXT_REGEX


class NSFW:
    def __init__(
        self,
        model_repo_id="allenai/dolma-jigsaw-fasttext-bigrams-nsfw",
        model_filename="model.bin",
    ):
        model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
        self.predictor = NSFWPredictor(model_path)

    def __call__(self, df: DataFrame) -> DataFrame:
        cleaned = F.strip(F.regexp_replace(col("text"), FASTTEXT_REGEX, " "))
        return df.with_column("nsfw_score", self.predictor.predict(cleaned))


class LanguageID:
    def __init__(self, model_repo_id="cis-lmu/glotlid", model_filename="model_v3.bin"):
        model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
        self.predictor = LanguagePredictor(model_path)

    def __call__(self, df: DataFrame) -> DataFrame:
        cleaned = F.strip(F.regexp_replace(col("text"), FASTTEXT_REGEX, " "))
        return df.with_column("language", self.predictor.predict(cleaned))
