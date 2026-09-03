from typing import Optional

from daft import DataFrame, col


def ai_content_factory(
    *,
    model_name: str,
    ai_label: str,
    revision: str | None = None,
    batch_size: int = 16,
    max_length: int | None = None,
    gpus: int | float = 0,
    cpus: Optional[float] = None,
):
    """Build a batched Daft scorer for one sequence-classifier label."""
    import daft
    from daft import DataType, Series

    @daft.cls(gpus=gpus, cpus=cpus)
    class AIContentClassifier:
        def __init__(self):
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_kwargs = {"revision": revision} if revision else {}
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **model_kwargs
            )
            self.device = "cuda" if gpus else "cpu"
            self.model.eval().to(self.device)
            self.inference_mode = torch.inference_mode
            labels = {
                label.casefold(): int(index)
                for index, label in self.model.config.id2label.items()
            }
            self.label_index = labels[ai_label.casefold()]

        @daft.method.batch(return_dtype=DataType.float64(), batch_size=batch_size)
        def score(self, text: Series) -> list[float]:
            values = text.to_pylist()
            valid = [
                (index, value)
                for index, value in enumerate(values)
                if value is not None and value.strip()
            ]
            scores = [0.0] * len(values)
            if not valid:
                return scores

            tokenizer_kwargs = {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }
            if max_length is not None:
                tokenizer_kwargs["max_length"] = max_length
            inputs = self.tokenizer([value for _, value in valid], **tokenizer_kwargs)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.inference_mode():
                probabilities = self.model(**inputs).logits.softmax(dim=-1)

            for (index, _), score in zip(
                valid, probabilities[:, self.label_index].cpu().tolist()
            ):
                scores[index] = score
            return scores

    return AIContentClassifier().score


class AIContentScorer:
    """Add a model-specific softmax score without filtering rows."""

    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "ai_content_score",
        name: str = "AIContentScorer",
        **factory_kwargs,
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.name = name
        self.score = ai_content_factory(**factory_kwargs)

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(self.output_column, self.score(col(self.input_column)))
