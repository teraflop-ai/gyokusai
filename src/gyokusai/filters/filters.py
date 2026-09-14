import gzip
from collections import Counter

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
    list_mean,
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
    ALPHABETIC_WORD,
    BULLET_LINE,
    ELLIPSIS,
    ELLIPSIS_SYMBOL,
    HASH,
    NON_ALPHA_NUMERIC,
    REPEATED_SENTENCES,
    TABLE_LINE,
    TERMINAL_PUNCTUATION,
    WORD,
)
from gyokusai.filters.utils import load_badwords, ngrams, paragraphs, sentences

from .config import ENGLISH_STOPWORDS
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
            regexp_count(element(), WORD) >= self.min_words,
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
        ratio = try_divide(regexp_count(text, NON_ALPHA_NUMERIC), length(text))
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
        words = regexp_count(col(self.input_column), WORD)
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
        spans = regexp_extract_all(lines, REPEATED_SENTENCES)
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


class StopWordDensityFilter(BaseFilter):
    def __init__(
        self,
        min_ratio: float = 0.06,
        stopwords: list[str] | None = None,
        input_column: str = "text",
        name: str = "StopWordDensityFilter",
    ):
        super().__init__(input_column, name)
        self.min_ratio = min_ratio
        self.stopwords = list(stopwords) if stopwords is not None else ENGLISH_STOPWORDS

    def __call__(self, df: DataFrame) -> DataFrame:
        text = col(self.input_column)
        hits = count_matches(
            text, self.stopwords, whole_words=True, case_sensitive=False
        )
        words = regexp_count(text, WORD)
        return df.where(try_divide(hits, words) >= self.min_ratio)


class StopWordFilter(BaseFilter):
    def __init__(
        self,
        min_stop_words: int = 2,
        stopwords: list[str] | None = None,
        input_column: str = "text",
        name: str = "StopWordFilter",
    ):
        super().__init__(input_column, name)
        self.min_stop_words = min_stop_words
        self.stopwords = list(stopwords) if stopwords is not None else ENGLISH_STOPWORDS

    def __call__(self, df: DataFrame) -> DataFrame:
        hits = count_matches(
            col(self.input_column),
            self.stopwords,
            whole_words=True,
            case_sensitive=False,
        )
        return df.where(hits >= self.min_stop_words)


class TableRatioFilter(BaseFilter):
    def __init__(
        self,
        max_ratio: float = 0.5,
        input_column: str = "text",
        name: str = "TableRatioFilter",
    ):
        super().__init__(input_column, name)
        self.max_ratio = max_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        table = list_count(list_filter(lines, regexp(element(), TABLE_LINE)))
        return df.where(try_divide(table, list_count(lines)) <= self.max_ratio)


class AlphabeticWordsFilter(BaseFilter):
    def __init__(
        self,
        min_ratio: float = 0.8,
        input_column: str = "text",
        name: str = "AlphabeticWordsFilter",
    ):
        super().__init__(input_column, name)
        self.min_ratio = min_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        text = col(self.input_column)
        alpha = regexp_count(text, ALPHABETIC_WORD)
        words = regexp_count(text, WORD)
        return df.where(try_divide(alpha, words) >= self.min_ratio)


class BulletsFilter(BaseFilter):
    def __init__(
        self,
        max_ratio: float = 0.9,
        input_column: str = "text",
        name: str = "BulletsFilter",
    ):
        super().__init__(input_column, name)
        self.max_ratio = max_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        bullets = list_count(list_filter(lines, regexp(element(), BULLET_LINE)))
        return df.where(try_divide(bullets, list_count(lines)) <= self.max_ratio)


class SymbolsToWordsFilter(BaseFilter):
    def __init__(
        self,
        max_ratio: float = 0.1,
        input_column: str = "text",
        name: str = "SymbolsToWordsFilter",
    ):
        super().__init__(input_column, name)
        self.max_ratio = max_ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        text = col(self.input_column)
        words = regexp_count(text, WORD)
        hashes = try_divide(regexp_count(text, HASH), words)
        ellipses = try_divide(regexp_count(text, ELLIPSIS_SYMBOL), words)
        return df.where((hashes <= self.max_ratio) & (ellipses <= self.max_ratio))


class RepeatingTopNGramsFilter(BaseFilter):
    def __init__(
        self,
        n: int = 2,
        max_ratio: float = 0.2,
        input_column: str = "text",
        name: str = "RepeatingTopNGramsFilter",
    ):
        super().__init__(input_column, name)
        self.n = n
        self.max_ratio = max_ratio

    @staticmethod
    @daft.func
    def top_ngram_ratio(text: str, n: int) -> float:
        grams = ngrams(text, n)
        if not grams:
            return 1.0
        top = Counter(grams).most_common(1)[0][0]
        i = c = 0
        while i < len(grams):
            hit = grams[i] == top
            c += hit
            i += n if hit else 1
        return c * (sum(map(len, top)) + n - 1) / len(text)

    def __call__(self, df: DataFrame) -> DataFrame:
        ratio = self.top_ngram_ratio(col(self.input_column), self.n)
        return df.where(ratio <= self.max_ratio)


class RepeatingDuplicateNGramsFilter(BaseFilter):
    def __init__(
        self,
        n: int = 2,
        max_ratio: float = 0.2,
        input_column: str = "text",
        name: str = "RepeatingDuplicateNGramsFilter",
    ):
        super().__init__(input_column, name)
        self.n = n
        self.max_ratio = max_ratio

    @staticmethod
    @daft.func
    def duplicate_ngram_ratio(text: str, n: int) -> float:
        grams = ngrams(text, n)
        if not grams:
            return 1.0
        seen, chars, overlap = set(), 0, 0
        for g in grams:
            if g in seen:
                chars += sum(map(len, g[overlap:])) + min(n - overlap, n - 1)
                overlap = n
            seen.add(g)
            overlap = max(overlap - 1, 0)
        return chars / len(text)

    def __call__(self, df: DataFrame) -> DataFrame:
        ratio = self.duplicate_ngram_ratio(col(self.input_column), self.n)
        return df.where(ratio <= self.max_ratio)


class RepeatedLinesByCharFilter(BaseFilter):
    def __init__(
        self,
        ratio: float = 0.8,
        input_column: str = "text",
        name: str = "RepeatedLinesByCharFilter",
    ):
        super().__init__(input_column, name)
        self.ratio = ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        lines = sentences(self.input_column)
        ratio = try_divide(
            length(list_join(list_distinct(lines), "")),
            length(list_join(lines, "")),
        )
        return df.where(ratio >= self.ratio)


class RepeatedParagraphsByCharFilter(BaseFilter):
    def __init__(
        self,
        ratio: float = 0.8,
        input_column: str = "text",
        name: str = "RepeatedParagraphsByCharFilter",
    ):
        super().__init__(input_column, name)
        self.ratio = ratio

    def __call__(self, df: DataFrame) -> DataFrame:
        paras = paragraphs(self.input_column)
        ratio = try_divide(
            length(list_join(list_distinct(paras), "")),
            length(list_join(paras, "")),
        )
        return df.where(ratio >= self.ratio)


class MeanWordLengthFilter(BaseFilter):
    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 10,
        input_column: str = "text",
        name: str = "MeanWordLengthFilter",
    ):
        super().__init__(input_column, name)
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, df: DataFrame) -> DataFrame:
        words = regexp_extract_all(col(self.input_column), WORD)
        mean_length = list_mean(list_map(words, length(element())))
        return df.where(
            (mean_length >= self.min_length) & (mean_length <= self.max_length)
        )
