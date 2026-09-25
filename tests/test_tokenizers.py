import daft
from daft import DataType
from transformers import AutoTokenizer

from gyokusai.tokenize import FastTextTokenize, Tokenize

MODEL = "Qwen/Qwen3.5-9B"


def test_tokenize():
    texts = ["", "hello world", "日本語のテキスト"]
    result = Tokenize(MODEL)(daft.from_pydict({"text": texts}))
    expected = AutoTokenizer.from_pretrained(MODEL)(texts, return_attention_mask=False)[
        "input_ids"
    ]

    assert result.schema()["input_ids"].dtype == DataType.list(DataType.int32())
    assert result.to_pydict()["input_ids"] == expected


def test_fasttext_preprocessor():
    texts = ["Café\n\n\n\nFrançais", "Hello   World\tbye"]
    df = daft.from_pydict({"text": texts})

    result = FastTextTokenize(model=MODEL)(df).to_pydict()["text"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    expected = [
        " ".join(tok.tokenize(t)) for t in ["cafe\n\nfrancais", "hello   world\tbye"]
    ]
    assert result == expected
    assert all(r == " ".join(r.split()) for r in result)
