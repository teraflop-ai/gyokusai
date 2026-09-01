from gyokusai.loader import DataLoader
from gyokusai.parsers.warc import ExtractWarc

url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-33/segments/1754151279521.11/warc/CC-MAIN-20250802220907-20250803010907-00000.warc.gz"

df = DataLoader(loader_type="warc").read_data(url)
warc = ExtractWarc()
df = warc(df)
df.show()
