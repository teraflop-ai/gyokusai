import gzip

import daft
from daft import DataFrame, col, element
from daft.functions import (
    contains,
    count_matches,
    length,
    list_count,
    list_distinct,
    list_filter,
    list_join,
    lower,
    regexp,
    regexp_count,
    try_divide,
)

from gyokusai.filters.regexes import ALPHA_NUMERIC, ELLIPSIS, TERMINAL_PUNCTUATION
from gyokusai.filters.utils import load_badwords, sentences


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
        min_words: int = 5,
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


class BadWordsFilter:
    def __init__(
        self,
        input_column: str = "text",
        badwords: list[str] | None = None,
        name: str = "BadWordsFilter",
    ):
        self.input_column = input_column
        self.badwords = list(badwords) if badwords is not None else load_badwords()
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        matches = count_matches(
            col(self.input_column),
            self.badwords,
            whole_words=True,
            case_sensitive=False,
        )
        return df.where(matches == 0)


class GzipCompressionFilter:
    def __init__(
        self,
        min_ratio: float = 0.3,
        max_ratio: float = 0.9,
        input_column: str = "text",
        name: str = "GzipCompressionFilter",
    ):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.input_column = input_column
        self.name = name

    @staticmethod
    @daft.func
    def gzip_ratio(text: str) -> float:
        if not text:
            return 0.0
        raw = text.encode()
        return len(gzip.compress(raw)) / len(raw)

    def __call__(self, df: DataFrame) -> DataFrame:
        ratio = self.gzip_ratio(col(self.input_column))
        return df.where((ratio >= self.min_ratio) & (ratio <= self.max_ratio))


class DigitRatioFilter:
    def __init__(
        self,
        max_ratio: float = 0.15,
        input_column: str = "text",
        name: str = "DigitRatioFilter",
    ):
        self.max_ratio = max_ratio
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        text = col(self.input_column)
        ratio = try_divide(regexp_count(text, r"\d"), length(text))
        return df.where(ratio <= self.max_ratio)


class NonAlphaNumericFilter:
    def __init__(
        self,
        max_ratio: float = 0.25,
        input_column: str = "text",
        name: str = "NonAlphaNumericFilter",
    ):
        self.max_ratio = max_ratio
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        text = col(self.input_column)
        ratio = try_divide(regexp_count(text, ALPHA_NUMERIC), length(text))
        return df.where(ratio <= self.max_ratio)


class EllipsisFilter:
    def __init__(
        self,
        max_ratio: float = 0.3,
        input_column: str = "text",
        name: str = "EllipsisFilter",
    ):
        self.max_ratio = max_ratio
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        ending = list_count(list_filter(lines, regexp(element(), ELLIPSIS)))
        return df.where(try_divide(ending, list_count(lines)) <= self.max_ratio)


class WordCountFilter:
    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100000,
        input_column: str = "text",
        name: str = "WordCountFilter",
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        words = regexp_count(col(self.input_column), r"\S+")
        return df.where((words >= self.min_words) & (words <= self.max_words))


from gyokusai.filters.utils import paragraphs


class RepeatedParagraphsFilter:
    def __init__(
        self,
        min_unique_ratio: float = 0.7,
        input_column: str = "text",
        name: str = "RepeatedParagraphsFilter",
    ):
        self.min_unique_ratio = min_unique_ratio
        self.input_column = input_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        paras = paragraphs(self.input_column)
        ratio = try_divide(list_count(list_distinct(paras)), list_count(paras))
        return df.where(ratio >= self.min_unique_ratio)
