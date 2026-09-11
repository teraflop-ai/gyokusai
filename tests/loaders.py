import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gyokusai.loaders import (
    CsvLoader,
    JsonLoader,
    LanceLoader,
    ParquetLoader,
    WarcLoader,
)


def test_parquet_loader(tmp_path):
    expected = {"id": [1, 2], "name": ["Alice", "Bob"]}
    path = tmp_path / "test.parquet"
    pq.write_table(pa.table(expected), path)

    df = ParquetLoader().read_data(str(path))

    assert df.sort("id").to_pydict() == expected


def test_json_loader(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(
        '{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n',
        encoding="utf-8",
    )

    df = JsonLoader().read_data(str(path))

    assert df.sort("id").to_pydict() == {
        "id": [1, 2],
        "name": ["Alice", "Bob"],
    }


def test_csv_loader(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")

    df = CsvLoader().read_data(str(path))

    assert df.sort("id").to_pydict() == {
        "id": [1, 2],
        "name": ["Alice", "Bob"],
    }


def test_warc_loader(tmp_path):
    path = tmp_path / "test.warc"
    payload = b"Hello, world!"
    headers = (
        "WARC/1.0\r\n"
        "WARC-Type: resource\r\n"
        "WARC-Record-ID: <urn:uuid:12345678-1234-1234-1234-123456789abc>\r\n"
        "WARC-Date: 2026-01-01T00:00:00Z\r\n"
        "WARC-Target-URI: https://example.com/\r\n"
        "Content-Type: text/plain\r\n"
        f"Content-Length: {len(payload)}\r\n\r\n"
    )
    path.write_bytes(headers.encode("ascii") + payload + b"\r\n\r\n")

    result = WarcLoader().read_data(str(path)).to_pydict()

    assert result["WARC-Type"] == ["resource"]
    assert result["warc_content"] == [payload]
    assert "source_path" not in result


def test_lance_loader(tmp_path):
    expected = {"id": [1, 2], "name": ["Alice", "Bob"]}
    path = tmp_path / "test.lance"
    lance.write_dataset(pa.table(expected), str(path))

    df = LanceLoader().read_data(str(path))

    assert df.sort("id").to_pydict() == expected


@pytest.mark.skip(reason="HuggingFaceLoader requires a Hub repository, not local files")
def test_huggingface_loader(tmp_path):
    pass
