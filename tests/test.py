def test_resume():
    from gyokusai.loader import DataLoader

    dataloader = DataLoader()

    df = dataloader.read_data(input_path="/home/henry/EDGAR/10-Q")

    written = df.write_parquet(
        "/home/henry/llm-data/processed",
        write_mode="append",
        write_success_file=True,
    )
    return written


def test_embed_text():
    import daft

    from gyokusai.factories.embedding import EmbedText

    df = daft.read_parquet("/home/henry/EDGAR/10-Q")
    df = EmbedText(
        input_column="text",
        output_column="text_embedding",
        model_name="lightonai/DenseOn",
        batch_size=32,
        embedding_dim=128,
        precision="float32",
        gpus=1,
    )(df)
    df.show()


def test_fix_text_encoding():
    import daft

    from gyokusai.encoding import FixEncoding

    df = daft.from_pydict(
        {
            "text": [
                "âœ” No problems",
                "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.",
            ]
        }
    )

    fix_unicode = FixEncoding()

    df = fix_unicode(df)

    df.show()


def test_parse_html():
    import daft

    from gyokusai.parsers.html import ParseHtml

    df = daft.from_pydict(
        {
            "html": [
                """

<!doctype html>
<html>
<head>
    <title>Example Domain</title>

    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />  
</head>

<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>
"""
            ]
        }
    )

    parser = ParseHtml()

    df = parser(df)
    df.show()


def test_data_engine():
    import daft

    from gyokusai.encoding import FixEncoding
    from gyokusai.engine import DataEngine

    df = daft.from_pydict(
        {
            "text": [
                "âœ” No problems",
                "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.",
            ]
        }
    )

    engine = DataEngine()
    engine.add(FixEncoding())
    print(engine)
    result = engine.run(df)
    result.show()


def test_parse_math():
    import daft

    from gyokusai.parsers.math import ParseMath

    MATH_HTML = """
<!DOCTYPE html>
<html>
<head><title>Math Page</title></head>
<body>
<article>
  <h1>Math Article</h1>
  <p>Here is some text before the equation that provides context for the math content below.</p>
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow>
  </math>
  <p>More text after the equation to provide enough content density for the extraction algorithm to work properly.</p>
  <p>Additional paragraph with even more text to ensure the article passes the minimum length requirements for extraction.</p>
</article>
</body>
</html>
"""
    df = daft.from_pydict({"html": [MATH_HTML]})

    parser = ParseMath()

    df = parser(df)
    df.show()
    result = df.collect()
    text = result.to_pydict()["text"][0]
    print(text)


def test_badwords_filter():
    import daft

    from gyokusai.filters.filters import BadWordsFilter

    df = daft.from_pydict(
        {
            "text": [
                "perfectly clean text",
                "a blue waffle appears",
                "badwords is a different token",
                None,
            ]
        }
    )

    badwords_filter = BadWordsFilter()

    df = badwords_filter(df)

    df.show()
    assert df.to_pydict()["text"] == [
        "perfectly clean text",
        "badwords is a different token",
    ]


if __name__ == "__main__":
    # test_resume()
    # test_embed_text()
    # test_fix_text_encoding()
    # test_parse_html()
    # test_data_engine()
    # test_parse_math()
    test_badwords_filter()
    pass
