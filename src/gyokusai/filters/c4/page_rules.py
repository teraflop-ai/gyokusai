"""Page-level C4 heuristics, exposed as Daft row-wise UDFs."""

import daft
from daft import DataFrame, col

from ...config import C4_MIN_SENTENCES
from .line_rules import (
    _ends_with_terminal_punctuation,
    _has_min_words,
    _mentions_javascript,
)


@daft.func
def clean_page(
    text: str,
    min_word_flag: bool,
    terminal_punctuation_flag: bool,
    javascript_flag: bool,
    min_line_flag: bool,
) -> str | None:
    """Keep only lines passing the line-level rules; drop the page if too short."""
    kept = []
    for line in text.splitlines():
        keep_flag = True
        if min_word_flag:
            keep_flag = keep_flag and _has_min_words(line)
        if terminal_punctuation_flag:
            keep_flag = keep_flag and _ends_with_terminal_punctuation(line)
        if javascript_flag:
            keep_flag = keep_flag and (not _mentions_javascript(line))
        if keep_flag:
            kept.append(line)

    min_line_threshold = C4_MIN_SENTENCES if min_line_flag else 0

    return "\n".join(kept) if len(kept) >= min_line_threshold else None


@daft.func
def has_lorem_ipsum(text: str) -> bool:
    return "lorem ipsum" in text.lower()


@daft.func
def has_curly_brace(text: str) -> bool:
    return "{" in text


class C4PageFilter:
    def __init__(
        self,
        input_column: str = "text",
        lorem_ipsum_filter: bool = True,
        curly_brace_filter: bool = False,
        line_terminal_punctuation_filter: bool = True,
        line_javascript_filter: bool = False,
        line_min_word_filter: bool = True,
        page_min_line_filter: bool = True,
        name: str = "C4PageFilter",
    ):
        self.input_column = input_column
        self.lorem_ipsum_filter = lorem_ipsum_filter
        self.curly_brace_filter = curly_brace_filter
        self.line_terminal_punctuation_filter = line_terminal_punctuation_filter
        self.line_javascript_filter = line_javascript_filter
        self.line_min_word_filter = line_min_word_filter
        self.page_min_line_filter = page_min_line_filter
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        if self.lorem_ipsum_filter:
            df = df.where(~has_lorem_ipsum(col(self.input_column)))
        if self.curly_brace_filter:
            df = df.where(~has_curly_brace(col(self.input_column)))
        df = df.with_column(
            self.input_column,
            clean_page(
                col(self.input_column),
                self.line_min_word_filter,
                self.line_terminal_punctuation_filter,
                self.line_terminal_punctuation_filter,
                self.page_min_line_filter,
            ),
        ).drop_null(self.input_column)
        return df
