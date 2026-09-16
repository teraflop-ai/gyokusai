import daft
from daft import DataFrame, col
from daft.functions import normalize, regexp_replace
from ftfy import fix_text

from .schemas import BaseNormalizer
from .tokenize import FastTextTokenize


class FixEncoding(BaseNormalizer):
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "text",
        name: str = "FixEncoding",
    ):
        super().__init__(input_column, output_column, name)

    @staticmethod
    @daft.func
    def ftfy_text(text: str) -> str:
        return fix_text(text)

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.ftfy_text(col(self.input_column))
        )


class FastTextPreprocess(BaseNormalizer):
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "text",
        name: str = "FastTextPreprocess",
        model: str = "Qwen/Qwen3.5-9B",
    ):
        super().__init__(input_column, output_column, name)
        self.tokenizer = FastTextTokenize(model)

    def __call__(self, df: DataFrame) -> DataFrame:
        text = regexp_replace(col(self.input_column), r"\n{3,}", "\n\n")
        text = normalize(text, lowercase=True, nfd_unicode=True)
        text = regexp_replace(text, r"\p{Mn}", "")
        return df.with_column(self.output_column, self.tokenizer.tokenize(text))
