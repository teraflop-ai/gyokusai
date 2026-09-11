import daft
from daft import DataFrame, col
from ftfy import fix_text

from .schemas import BaseNormalizer


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
