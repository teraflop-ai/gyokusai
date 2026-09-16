import daft
import pytest
from daft import col

from gyokusai.detectors import CodeDetector, MathDetector


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (None, False),
        ("", False),
        ("<pre><code>print(1)</code></pre>", True),
        ("<pre><code>&#112;rint(1)</code></pre>", True),
        ("<pre><code>hello world</code></pre>", False),
        ("<code>print(1)</code>", False),
        ("<!-- <pre><code>print(1)</code></pre> -->", False),
    ],
)
def test_code_detector(html, expected):
    df = daft.from_pydict({"html": [html]})
    detector = CodeDetector()
    result = df.with_column("has_code", detector.contains(col("html"))).to_pydict()

    assert result["has_code"] == [expected]


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (None, False),
        ("", False),
        ("<p>x + y = z</p>", False),
        ("<math></math>", False),
        ("<math><mi>x</mi></math>", True),
        ('<span class="math-formula">x^2</span>', True),
        ('<annotation encoding="application/x-tex">x^2</annotation>', True),
        (r'<script type="math/tex">\frac{1}{2}</script>', True),
        (
            '<script type="math/mml"><math><mi>x</mi></math></script>',
            True,
        ),
        (r'<script type="math/tex">\frac{1}{2</script>', False),
        (
            '<script type="math/tex">x^2</script>'
            r'<script type="math/tex">\frac{1}{2</script>',
            False,
        ),
        ("<math><foo>x</foo></math>", False),
        ("<!-- <math><mi>x</mi></math> -->", False),
    ],
)
def test_math_detector(html, expected):
    df = daft.from_pydict({"html": [html]})
    detector = MathDetector()
    result = df.with_column("has_math", detector.contains(col("html"))).to_pydict()

    assert result["has_math"] == [expected]
