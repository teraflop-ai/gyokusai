import json

import daft
import jsonschema
import pytest

from gyokusai.factories.generation import GenerateText

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 400},
        "score": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}


@pytest.mark.gpu
def test_generate_text():
    df = daft.from_pydict(
        {
            "text": [
                "The derivative of x^2 is 2x by the power rule: d/dx x^n = n x^(n-1)."
            ]
        }
    )
    step = GenerateText(
        model_path="Qwen/Qwen3.5-0.8B",
        instruction="Rate how useful the text is for learning math, from 0 to 5. Give brief reasoning, then the score.",
        max_new_tokens=512,
        context_length=2048,
        mem_fraction_static=0.5,
        json_schema=SCHEMA,
    )
    out = step(df).to_pydict()["extract"]
    jsonschema.validate(json.loads(out[0]), SCHEMA)
