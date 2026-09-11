from gyokusai.loaders import WarcLoader
from gyokusai.warc import WarcProcessor

URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-33/segments/1754151279521.11/warc/CC-MAIN-20250802220907-20250803010907-00000.warc.gz"


def test_warc_processor():
    df = WarcLoader().read_data(URL)
    result = WarcProcessor()(df).to_pydict()
