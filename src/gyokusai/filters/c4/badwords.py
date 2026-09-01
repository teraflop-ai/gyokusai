import daft
import requests
from daft import DataFrame, col

from ...config import BAD_WORDS_URL


@daft.func
def has_badword(text: str, badwords: set) -> bool:
    return not set(text.lower().split()).isdisjoint(badwords)


def load_badwords(url: str = BAD_WORDS_URL) -> frozenset[str]:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return frozenset(w.strip().lower() for w in resp.text.splitlines() if w.strip())


class BadWordsFilter:
    def __init__(
        self,
        input_column: str = "text",
        badwords: set[str] | None = None,
        badwords_url: str = BAD_WORDS_URL,
        name: str = "BadWordsFilter",
    ):
        self.input_column = input_column
        self.name = name

        if badwords is not None:
            self.badwords = badwords
        else:
            self.badwords = load_badwords(badwords_url)

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.where(~has_badword(col(self.input_column), self.badwords))
        return df
