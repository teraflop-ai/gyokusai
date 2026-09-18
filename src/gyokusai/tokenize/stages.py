from daft import DataFrame, col
from daft.functions import normalize, regexp_replace

from .tokenize import FastTextTokenizeText, TokenizeText


class Tokenize:
    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-9B",
        input_column: str = "text",
        output_column: str = "input_ids",
        name: str = "Tokenize",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.name = name
        self.tokenizer = TokenizeText(model)

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.tokenizer.tokenize(col(self.input_column))
        )


class FastTextTokenize:
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "text",
        name: str = "FastTextTokenize",
        model: str = "Qwen/Qwen3.5-9B",
    ):
        super().__init__()
        self.tokenizer = FastTextTokenizeText(model)
        self.input_column = input_column
        self.output_column = output_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        text = regexp_replace(col(self.input_column), r"\n{3,}", "\n\n")
        text = normalize(text, lowercase=True, nfd_unicode=True)
        text = regexp_replace(text, r"\p{Mn}", "")
        return df.with_column(self.output_column, self.tokenizer.tokenize(text))
