from gyokusai.loaders import WarcLoader
from gyokusai.warc import WarcProcessor

URL = (
    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-33/"
    "segments/1754151279521.11/warc/"
    "CC-MAIN-20250802220907-20250803010907-00000.warc.gz"
)


def test_warc_processor():
    source = WarcLoader().read_data(URL).limit(100).collect()
    raw = source.to_pydict()
    result = WarcProcessor()(source).to_pydict()

    expected_ids = [
        record_id
        for record_id, kind in zip(raw["WARC-Record-ID"], raw["WARC-Type"])
        if kind == "response"
    ]

    assert expected_ids
    assert set(result) == {"WARC-Record-ID", "html", "WARC-Target-URI", "WARC-Date"}
    assert sorted(result["WARC-Record-ID"]) == sorted(expected_ids)
    assert any(result["html"])
    assert all(html is None or isinstance(html, str) for html in result["html"])
