from daft import DataFrame, col

from gyokusai.utils import decode_html


class WarcProcessor:
    def __init__(self, name: str = "ExtractWarc"):
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.where(col("WARC-Type") == "response")
        df = df.with_column("html", decode_html(col("warc_content")))
        return df.select(
            col("WARC-Record-ID"),
            "html",
            col("WARC-Target-URI"),
            col("WARC-Date"),
        )
