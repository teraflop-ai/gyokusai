import daft
from transformers import AutoTokenizer

from gyokusai.normalizers import FastTextPreprocess, FixEncoding

MODEL = "Qwen/Qwen3.5-9B"


def test_fix_encoding():
    texts = ["cafÃ©", "FranÃ§ais", "Already correct."]
    df = daft.from_pydict({"text": texts})

    result = FixEncoding()(df).to_pydict()

    assert result["text"] == ["café", "Français", "Already correct."]


def test_fasttext_preprocessor():
    texts = ["Café\n\n\n\nFrançais", "Hello   World\tbye"]
    df = daft.from_pydict({"text": texts})

    result = FastTextPreprocess(model=MODEL)(df).to_pydict()["text"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    expected = [
        " ".join(tok.tokenize(t)) for t in ["cafe\n\nfrancais", "hello   world\tbye"]
    ]
    assert result == expected
    assert all(r == " ".join(r.split()) for r in result)
