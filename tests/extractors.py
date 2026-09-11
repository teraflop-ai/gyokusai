import shutil

import daft
import pytest

from gyokusai.extractors import ExtractHTML, MagicHTMLExtractor


@pytest.fixture
def html():
    return """
    <!doctype html>
    <html lang="en">
      <head><title>Community garden opens</title></head>
      <body>
        <nav>Home | News | Contact</nav>
        <main>
          <article>
            <h1>Community garden opens</h1>
            <p>Residents planted apple trees in the community garden on Saturday.
            Volunteers prepared the soil, carried water, and helped children
            plant their first seedlings. The garden provides a shared space
            where neighbours can grow fresh food and learn together.</p>
            <p>The garden will host free workshops throughout the summer.
            Experienced gardeners will explain composting, watering, and
            seasonal planting. Families can visit every weekend to help
            maintain the beds and collect vegetables when they are ready.</p>
          </article>
        </main>
        <footer>Copyright 2026</footer>
        <script>window.hiddenMarker = "SCRIPT_SHOULD_NOT_APPEAR";</script>
      </body>
    </html>
    """


def test_trafilatura_extractor(html):
    df = daft.from_pydict({"html": [html]})

    result = ExtractHTML(extractor_type="trafilatura")(df).to_pydict()

    assert result["html"] == [html]
    assert "Residents planted apple trees" in result["text"][0]
    assert "free workshops" in result["text"][0]
    assert "SCRIPT_SHOULD_NOT_APPEAR" not in result["text"][0]


def test_resiliparse_extractor(html):
    df = daft.from_pydict({"html": [html]})

    result = ExtractHTML(extractor_type="resiliparse")(df).to_pydict()

    assert result["html"] == [html]
    assert "Residents planted apple trees" in result["text"][0]
    assert "free workshops" in result["text"][0]
    assert "SCRIPT_SHOULD_NOT_APPEAR" not in result["text"][0]


@pytest.mark.skipif(shutil.which("lynx") is None, reason="lynx is not installed")
def test_lynx_extractor(html):
    df = daft.from_pydict({"html": [html]})

    result = ExtractHTML(extractor_type="lynx", width=200, timeout=10)(df).to_pydict()
    text = " ".join(result["text"][0].split())

    assert "Residents planted apple trees" in text
    assert "free workshops" in text
    assert "SCRIPT_SHOULD_NOT_APPEAR" not in text


@pytest.mark.skipif(shutil.which("w3m") is None, reason="w3m is not installed")
def test_w3m_extractor(html):
    df = daft.from_pydict({"html": [html]})

    result = ExtractHTML(extractor_type="w3m", width=200, timeout=10)(df).to_pydict()
    text = " ".join(result["text"][0].split())

    assert "Residents planted apple trees" in text
    assert "free workshops" in text
    assert "SCRIPT_SHOULD_NOT_APPEAR" not in text


@pytest.mark.skipif(shutil.which("elinks") is None, reason="elinks is not installed")
def test_elinks_extractor(html):
    df = daft.from_pydict({"html": [html]})

    result = ExtractHTML(extractor_type="elinks", width=200, timeout=10)(df).to_pydict()
    text = " ".join(result["text"][0].split())

    assert "Residents planted apple trees" in text
    assert "free workshops" in text
    assert "SCRIPT_SHOULD_NOT_APPEAR" not in text


def test_magichtml_extractor(html):
    df = daft.from_pydict({"html": [html]})
    extractor = MagicHTMLExtractor()

    result = df.with_column(
        "cleaned_html", extractor.text_extraction(daft.col("html"))
    ).to_pydict()
    cleaned = result["cleaned_html"][0]

    assert "Residents planted apple trees" in cleaned
    assert "free workshops" in cleaned
    assert "SCRIPT_SHOULD_NOT_APPEAR" not in cleaned


def test_extract_html_with_cleaning(html):
    df = daft.from_pydict({"html": [html]})

    result = ExtractHTML(extractor_type="resiliparse", clean=True)(df).to_pydict()

    assert "Residents planted apple trees" in result["text"][0]
    assert "free workshops" in result["text"][0]
    assert "SCRIPT_SHOULD_NOT_APPEAR" not in result["text"][0]


def test_extract_html_invalid_extractor():
    with pytest.raises(ValueError, match="Extractor not available: unknown"):
        ExtractHTML(extractor_type="unknown")
