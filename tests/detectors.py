import daft
import pytest
from daft import DataType, col

from gyokusai.detectors import AIPhrase, AIStyle, AITrace, Code, Math


def frame(column, value):
    return daft.from_pydict({column: [value]}).with_column(
        column, col(column).cast(DataType.string())
    )


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
def test_code(html, expected):
    result = Code()(frame("html", html)).to_pydict()

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
def test_math(html, expected):
    result = Math()(frame("html", html)).to_pydict()

    assert result["has_math"] == [expected]


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (None, False),
        ("", False),
        ('<a href="https://x.com/?utm_source=chatgpt.com">x</a>', True),
        (
            '<a href="https://r.com/?u=https%3A%2F%2Fx.com%2F%3Futm_source%3Dchatgpt.com">x</a>',
            True,
        ),
        ('<a href="https://x.com/?utm_source=newsletter">x</a>', False),
        ('<div data-message-model-slug="gpt-4o-mini"></div>', True),
        ('<div data-testid="conversation-turn-3"></div>', True),
        ('<h3 data-start="1181" data-end="1230">x</h3>', True),
        ('<pre data-start="5"><code>x</code></pre>', False),
        ('<p data-is-last-node="">x</p>', True),
        ("<p>hello world</p>", False),
    ],
)
def test_ai_trace(html, expected):
    result = AITrace()(frame("html", html)).to_pydict()

    assert result["has_ai_trace"] == [expected]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, False),
        ("", False),
        ("As an AI language model, I cannot browse the internet.", True),
        ("As an AI, I can't help with that.", True),
        ("As an AI researcher, I disagree.", False),
        ("I'm an AI.", True),
        ("Intro\nI am ChatGPT, a model.", True),
        ("I was trained by OpenAI.", True),
        ("Some text. As of my last knowledge update, X was true.", True),
        ('He wrote "As an AI language model" in the essay.', False),
        ("I'm sorry, but I cannot provide medical advice.", True),
        ("I’m sorry, but I can’t generate that.", True),
        ("I'm sorry, but I can't make it on Friday.", False),
        ("I don't have access to real-time data.", True),
        ("I don't have real time to spend on this.", False),
        ("I can't access the internet at home.", False),
        ("Certainly! Here's a summary of the article.", True),
        ("Sure, here's my config file.", False),
        ("Here's an updated version of your essay.", True),
        ("Here's an updated version of the changelog.", False),
        ("Regenerate response", True),
        ("Click Regenerate response to retry.", False),
    ],
)
def test_ai_phrase(text, expected):
    result = AIPhrase()(frame("text", text)).to_pydict()

    assert result["has_ai_phrase"] == [expected]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, False),
        ("", False),
        (
            "It's important to note that in today's fast-paced world, "
            "unlock your full potential is a testament to hard work.",
            True,
        ),
        ("This is a testament to the team. Let's dive into the results.", False),
        (
            "lorem ipsum " * 600
            + "A testament to it. Without further ado. Only time will tell.",
            False,
        ),
    ],
)
def test_ai_style(text, expected):
    result = AIStyle()(frame("text", text)).to_pydict()

    assert result["has_ai_style"] == [expected]
