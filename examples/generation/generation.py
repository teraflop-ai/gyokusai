import argparse
import json
import os

from gyokusai.engine import DataEngine
from gyokusai.factories.generation import GenerateText
from gyokusai.prompts import WEBPAGE_EDU_SCORE_INSTRUCTION
from gyokusai.runner import SlurmRunner


def json_arg(v):
    if not v:
        return None
    if os.path.isfile(v):
        with open(v) as f:
            return json.load(f)
    return json.loads(v)


if __name__ == "__main__":
    p = SlurmRunner.add_args(argparse.ArgumentParser())
    p.add_argument("--input-column", default="text")
    p.add_argument("--output-column", default="extract")
    p.add_argument(
        "--model-path",
        default="/e/data1/datasets/products/mmlaion/shared/models/Qwen/Qwen3.8-27B-FP8",
    )
    p.add_argument("--instruction", default=WEBPAGE_EDU_SCORE_INSTRUCTION)
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("SLURM_GPUS_ON_NODE", 4)),
    )
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--mem-fraction-static", type=float, default=0.9)
    p.add_argument("--chunked-prefill-size", type=int, default=16384)
    p.add_argument("--truncate-to", type=int, default=32000)
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument(
        "--json-schema",
        type=json_arg,
        default="/e/project1/reformo/enrico/qwen-sglang/quality.json",
    )

    job, kwargs = SlurmRunner.from_args(p.parse_args())
    job.run(DataEngine([GenerateText(**kwargs)], name="edu-score"))
