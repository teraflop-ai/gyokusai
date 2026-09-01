from pathlib import Path

from daft import col, element
from daft.functions import length, list_filter, list_map, split, strip

from gyokusai.filters.config import BAD_WORDS_PATH


def sentences(column: str):
    lines = list_map(split(col(column), "\n"), strip(element()))
    return list_filter(lines, length(element()) > 0)


def paragraphs(column: str):
    paras = list_map(split(col(column), "\n\n"), strip(element()))
    return list_filter(paras, length(element()) > 0)


def load_badwords(path: Path = BAD_WORDS_PATH) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [w.strip().lower() for w in lines if w.strip()]
