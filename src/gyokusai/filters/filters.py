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
    list_map,
    lower,
    min,
    regexp,
    regexp_count,
    regexp_extract_all,
    regexp_replace,
    strip,
    to_list,
    try_divide,
    when,
)

from gyokusai.filters.regexes import (
    ALPHA_NUMERIC,
    ELLIPSIS,
    REPEATEDSENTENCES,
    TERMINAL_PUNCTUATION,
)
from gyokusai.filters.utils import load_badwords, paragraphs, sentences

from .schemas import BaseFilter


class LoremIpsumFilter(BaseFilter):
    def __init__(self, input_column: str = "text", name: str = "LoremIpsumFilter"):
        super().__init__(input_column, name)

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.where(~contains(lower(col(self.input_column)), "lorem ipsum"))


class CurlyBraceFilter(BaseFilter):
    def __init__(self, input_column: str = "text", name: str = "CurlyBraceFilter"):
        super().__init__(input_column, name)

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.where(~contains(col(self.input_column), "{"))


class MinLinesFilter(BaseFilter):
    def __init__(
        self,
        min_lines: int = 3,
        input_column: str = "text",
        name: str = "MinLinesFilter",
    ):
        super().__init__(input_column, name)
        self.min_lines = min_lines

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.where(list_count(sentences(self.input_column)) >= self.min_lines)


class TerminalPunctuationLineFilter(BaseFilter):
    def __init__(
        self, input_column: str = "text", name: str = "TerminalPunctuationLineFilter"
    ):
        super().__init__(input_column, name)

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = list_filter(
            sentences(self.input_column), regexp(element(), TERMINAL_PUNCTUATION)
        )
        return df.with_column(self.input_column, list_join(lines, "\n"))


class MinWordsLineFilter(BaseFilter):
    def __init__(
        self,
        min_words: int = 5,
        input_column: str = "text",
        name: str = "MinWordsLineFilter",
    ):
        super().__init__(input_column, name)
        self.min_words = min_words

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = list_filter(
            sentences(self.input_column),
            regexp_count(element(), r"\S+") >= self.min_words,
        )
        return df.with_column(self.input_column, list_join(lines, "\n"))


class JavascriptLineFilter(BaseFilter):
    def __init__(self, input_column: str = "text", name: str = "JavascriptLineFilter"):
        super().__init__(input_column, name)

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = list_filter(
            sentences(self.input_column), ~contains(lower(element()), "javascript")
        )
        return df.with_column(self.input_column, list_join(lines, "\n"))


class LengthFilter(BaseFilter):
    def __init__(
        self,
        input_column: str = "text",
        max_len: int = 10000,
        min_len: int = 0,
        name: str = "LengthFilter",
    ):
        super().__init__(input_column, name)
        self.max_len = max_len
        self.min_len = min_len

    def __call__(self, df: DataFrame) -> DataFrame:
        text_len = length(col(self.input_column))
        return df.where((text_len < self.max_len) & (text_len > self.min_len))


class PunctuationFilter(BaseFilter):
    def __init__(
        self,
        max_ratio: float = 0.85,
        input_column: str = "text",
        name: str = "PunctuationFilter",
    ):
        super().__init__(input_column, name)
        self.max_ratio = max_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        without = list_count(
            list_filter(lines, ~regexp(element(), TERMINAL_PUNCTUATION))
        )
        return df.where(try_divide(without, list_count(lines)) <= self.max_ratio)


class BadWordsFilter(BaseFilter):
    def __init__(
        self,
        input_column: str = "text",
        badwords: list[str] | None = None,
        name: str = "BadWordsFilter",
    ):
        super().__init__(input_column, name)
        self.badwords = list(badwords) if badwords is not None else load_badwords()

    def __call__(self, df: DataFrame) -> DataFrame:
        matches = count_matches(
            col(self.input_column),
            self.badwords,
            whole_words=True,
            case_sensitive=False,
        )
        return df.where(matches == 0)


class GzipCompressionFilter(BaseFilter):
    def __init__(
        self,
        min_ratio: float = 0.3,
        max_ratio: float = 0.9,
        input_column: str = "text",
        name: str = "GzipCompressionFilter",
    ):
        super().__init__(input_column, name)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

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


class DigitRatioFilter(BaseFilter):
    def __init__(
        self,
        max_ratio: float = 0.15,
        input_column: str = "text",
        name: str = "DigitRatioFilter",
    ):
        super().__init__(input_column, name)
        self.max_ratio = max_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        text = col(self.input_column)
        ratio = try_divide(regexp_count(text, r"\d"), length(text))
        return df.where(ratio <= self.max_ratio)


class NonAlphaNumericFilter(BaseFilter):
    def __init__(
        self,
        max_ratio: float = 0.25,
        input_column: str = "text",
        name: str = "NonAlphaNumericFilter",
    ):
        super().__init__(input_column, name)
        self.max_ratio = max_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        text = col(self.input_column)
        ratio = try_divide(regexp_count(text, ALPHA_NUMERIC), length(text))
        return df.where(ratio <= self.max_ratio)


class EllipsisFilter(BaseFilter):
    def __init__(
        self,
        max_ratio: float = 0.3,
        input_column: str = "text",
        name: str = "EllipsisFilter",
    ):
        super().__init__(input_column, name)
        self.max_ratio = max_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        ending = list_count(list_filter(lines, regexp(element(), ELLIPSIS)))
        return df.where(try_divide(ending, list_count(lines)) <= self.max_ratio)


class WordCountFilter(BaseFilter):
    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100000,
        input_column: str = "text",
        name: str = "WordCountFilter",
    ):
        super().__init__(input_column, name)
        self.min_words = min_words
        self.max_words = max_words

    def __call__(self, df: DataFrame) -> DataFrame:
        words = regexp_count(col(self.input_column), r"\S+")
        return df.where((words >= self.min_words) & (words <= self.max_words))


class RepeatedParagraphsFilter(BaseFilter):
    def __init__(
        self,
        ratio: float = 0.7,
        input_column: str = "text",
        name: str = "RepeatedParagraphsFilter",
    ):
        super().__init__(input_column, name)
        self.ratio = ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        paras = paragraphs(self.input_column)
        ratio = try_divide(list_count(list_distinct(paras)), list_count(paras))
        return df.where(ratio >= self.ratio)


class RepeatedLinesFilter(BaseFilter):
    def __init__(
        self,
        ratio: float = 0.7,
        input_column: str = "text",
        name: str = "RepeatedLinesFilter",
    ):
        super().__init__(input_column, name)
        self.ratio = ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        ratio = try_divide(list_count(list_distinct(lines)), list_count(lines))
        return df.where(ratio >= self.ratio)


class ThreeSentenceDedupFilter(BaseFilter):
    def __init__(
        self,
        doc_id: str = "record_id",
        input_column: str = "text",
        name: str = "ThreeSentenceDedupFilter",
    ):
        super().__init__(input_column, name)
        self.doc_id = doc_id

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = list_join(sentences(self.input_column), "\n")
        spans = regexp_extract_all(lines, REPEATEDSENTENCES)
        spans = list_map(spans, regexp_replace(strip(element()), "\n", " "))
        spans = when(
            list_count(spans) == 0,
            then=to_list(col(self.input_column)),
        ).otherwise(spans)

        novel_docs = (
            df.select(col(self.doc_id).alias("_doc_id"), spans.alias("_span"))
            .explode("_span")
            .groupby("_span")
            .agg(min(col("_doc_id")).alias(self.doc_id))
            .select(self.doc_id)
            .distinct()
        )
        return df.join(novel_docs, on=self.doc_id, how="inner")
