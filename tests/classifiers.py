import daft

from gyokusai.classifiers import NSFW, LanguageID


def test_extract_language():
    texts = [
        "The library is open every day. You can borrow books and read newspapers.",
        "Die Bibliothek ist jeden Tag geöffnet. Dort kann man Bücher ausleihen.",
        "La bibliothèque est ouverte tous les jours. Vous pouvez emprunter des livres.",
    ]
    df = daft.from_pydict({"text": texts})

    result = LanguageID()(df).to_pydict()

    assert result["text"] == texts
    assert result["language"] == [
        "__label__eng_Latn",
        "__label__deu_Latn",
        "__label__fra_Latn",
    ]


def test_extract_language_with_newlines():
    text = (
        "The library is open every day.\r\nYou can borrow books.\nEveryone is welcome."
    )
    df = daft.from_pydict({"text": [text]})

    result = LanguageID()(df).to_pydict()

    assert result["text"] == [text]
    assert result["language"] == ["__label__eng_Latn"]


def test_extract_nsfw():
    texts = [
        "The library is open every day. You can borrow books and read newspapers.",
        "Shut the fuck up you stupid piece of shit, nobody gives a damn.",
    ]
    df = daft.from_pydict({"text": texts})

    result = NSFW()(df).to_pydict()

    assert result["text"] == texts
    clean, nsfw = result["nsfw_score"]
    assert clean < 0.5 < nsfw


def test_extract_nsfw_with_newlines():
    text = (
        "The library is open every day.\r\nYou can borrow books.\nEveryone is welcome."
    )
    df = daft.from_pydict({"text": [text]})

    result = NSFW()(df).to_pydict()

    assert result["text"] == [text]
    assert result["nsfw_score"][0] < 0.5


def test_extract_nsfw_empty():
    df = daft.from_pydict({"text": ["", None]})

    result = NSFW()(df).to_pydict()

    assert result["nsfw_score"] == [0.0, 0.0]
