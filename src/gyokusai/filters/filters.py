from daft import DataFrame, col
from daft.functions import length


class LengthFilter:
    def __init__(
        self,
        input_column: str = "text",
        max_len: int = 10000,
        min_len: int = 0,
        name: str = "LengthFilter",
    ):
        self.input_column = input_column
        self.max_len = max_len
        self.min_len = min_len
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        text_len = length(col(self.input_column))
        return df.where((text_len < self.max_len) & (text_len > self.min_len))
