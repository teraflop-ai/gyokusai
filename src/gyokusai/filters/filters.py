from daft import DataFrame, col, element
from daft.functions import (
    contains,
    length,
    list_count,
    list_filter,
    list_join,
    lower,
    regexp,
    regexp_count,
    try_divide,
)

from gyokusai.regexes import TERMINAL_PUNCTUATION
from gyokusai.utils import sentences


class LoremIpsumFilter:
    def __init__(self, input_column: str = "text", name: str = "LoremIpsumFilter"):
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.where(~contains(lower(col(self.input_column)), "lorem ipsum"))


class CurlyBraceFilter:
    def __init__(self, input_column: str = "text", name: str = "CurlyBraceFilter"):
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.where(~contains(col(self.input_column), "{"))


class MinLinesFilter:
    def __init__(
        self,
        min_lines: int = 3,
        input_column: str = "text",
        name: str = "MinLinesFilter",
    ):
        self.min_lines = min_lines
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.where(list_count(sentences(self.input_column)) >= self.min_lines)


class TerminalPunctuationLineFilter:
    def __init__(
        self, input_column: str = "text", name: str = "TerminalPunctuationLineFilter"
    ):
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = list_filter(
            sentences(self.input_column), regexp(element(), TERMINAL_PUNCTUATION)
        )
        return df.with_column(self.input_column, list_join(lines, "\n"))


class MinWordsLineFilter:
    def __init__(
        self,
        min_words: int = 3,
        input_column: str = "text",
        name: str = "MinWordsLineFilter",
    ):
        self.min_words = min_words
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = list_filter(
            sentences(self.input_column),
            regexp_count(element(), r"\S+") >= self.min_words,
        )
        return df.with_column(self.input_column, list_join(lines, "\n"))


class JavascriptLineFilter:
    def __init__(self, input_column: str = "text", name: str = "JavascriptLineFilter"):
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = list_filter(
            sentences(self.input_column), ~contains(lower(element()), "javascript")
        )
        return df.with_column(self.input_column, list_join(lines, "\n"))


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


class PunctuationFilter:
    def __init__(
        self,
        max_ratio: float = 0.85,
        input_column: str = "text",
        name: str = "PunctuationFilter",
    ):
        self.max_ratio = max_ratio
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        without = list_count(
            list_filter(lines, ~regexp(element(), TERMINAL_PUNCTUATION))
        )
        return df.where(try_divide(without, list_count(lines)) <= self.max_ratio)
