import daft

from gyokusai.normalizers import FixEncoding


def test_fix_encoding():
    texts = ["cafÃ©", "FranÃ§ais", "Already correct."]
    df = daft.from_pydict({"text": texts})

    result = FixEncoding()(df).to_pydict()

    assert result["text"] == ["café", "Français", "Already correct."]
