import daft
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gyokusai.deduplication.bloom import (
    DocumentDeduplicator,
    ParagraphDeduplicator,
    URLDeduplicator,
)
from gyokusai.deduplication.embedding import Dedup
from gyokusai.deduplication.stages import BloomDeduplicationStage

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
D, K, N = 64, 4, 500


def make(seed=0):
    torch.manual_seed(seed)
    centers = torch.nn.functional.normalize(torch.randn(K, D), dim=1)
    x = torch.nn.functional.normalize(
        centers.repeat_interleave(N, 0) + 0.1 * torch.randn(K * N, D), dim=1
    )
    dups = torch.arange(0, K * N, 100)
    x = torch.cat(
        [
            x,
            torch.nn.functional.normalize(
                x[dups] + 1e-3 * torch.randn(len(dups), D), dim=1
            ),
        ]
    )
    return x, dups


def test_soar_and_pairs(tmp_path):
    dd = Dedup(str(tmp_path), str(tmp_path), d=D, k=K)
    x, dups = make()
    x = x.cuda()
    C = torch.nn.functional.normalize(torch.randn(K, D, device="cuda"), dim=1)
    parts = dd.soar(x, C)
    assert (parts[:, 0] == (x @ C.T).argmax(1)).all()
    assert (parts[:, 0] != parts[:, 1]).all()
    pairs = dd.dup_pairs(x)
    assert {tuple(p) for p in pairs.tolist()} == {
        (int(i), K * N + j) for j, i in enumerate(dups)
    }


def test_pipeline(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for d in (src, dst / "parts", dst / "drop"):
        d.mkdir(parents=True)
    x, dups = make()
    uuid = np.array([f"u{i:06d}" for i in range(len(x))])
    for f, idx in enumerate(np.array_split(np.arange(len(x)), 4)):
        pq.write_table(
            pa.table({"uuid": uuid[idx], "embedding": list(x[idx].numpy())}),
            src / f"{f}.parquet",
        )
    dd = Dedup(str(src), str(dst), d=D, k=K)
    dd.train()
    dd.assign()
    dd.dedup()
    dropped = set(
        pq.read_table(list((dst / "drop").glob("*.parquet")))["uuid"].to_pylist()
    )
    assert dropped == set(uuid[K * N :])


def test_url_deduplicator():
    rows = [
        {"url": "a", "text": "first"},
        {"url": "a", "text": "second"},
        {"url": "b", "text": "third"},
    ]

    result = list(URLDeduplicator(capacity=100).run(rows))

    assert result == [rows[0], rows[2]]


def test_document_deduplicator():
    rows = [
        {"url": "a", "text": "same"},
        {"url": "b", "text": "same"},
        {"url": "c", "text": "different"},
    ]

    result = list(DocumentDeduplicator(capacity=100).run(rows))

    assert result == [rows[0], rows[2]]


def test_paragraph_deduplicator():
    rows = [
        {"url": "a", "text": "A\nB\nA"},
        {"url": "b", "text": "B\nC"},
        {"url": "c", "text": "A\nB"},
    ]

    result = list(ParagraphDeduplicator(capacity=100).run(rows))

    assert result == [
        {"url": "a", "text": "A\nB"},
        {"url": "b", "text": "C"},
    ]


def test_deduplication_stage():
    df = daft.from_pydict(
        {
            "url": ["a", "a", "b", "c", "d"],
            "text": ["A\nB", "ignored", "A\nB", "B\nC", "B"],
        }
    )
    stage = BloomDeduplicationStage(
        URLDeduplicator(capacity=100),
        DocumentDeduplicator(capacity=100),
        ParagraphDeduplicator(capacity=100),
    )

    result = list(stage.run(df))

    assert result == [
        {"url": "a", "text": "A\nB"},
        {"url": "c", "text": "C"},
    ]
