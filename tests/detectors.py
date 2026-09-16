import daft
import pytest
from daft import col

from gyokusai.detectors import CodeDetector


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
