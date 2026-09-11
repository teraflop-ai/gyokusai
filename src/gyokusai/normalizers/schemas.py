from abc import ABC, abstractmethod

from daft import DataFrame


class BaseNormalizer(ABC):
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "text",
        name: str | None = None,
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.name = name

    @abstractmethod
    def __call__(self, df: DataFrame) -> DataFrame: ...
