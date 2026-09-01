import shutil
import subprocess

import daft
from daft import DataFrame, col
from magic_html import GeneralExtractor

from gyokusai.utils import decode_html


@daft.func
def magic_parse(html: str) -> str:
    return GeneralExtractor().extract(html)["html"]


@daft.func
def lynx_extract(html: str, width: int, timeout: int) -> str:
    """
    Nemotron-CC-Math: https://arxiv.org/abs/2508.15096
    """
    return decode_html(
        subprocess.run(
            [
                "lynx",
                "-dump",
                "-stdin",
                "-nolist",
                f"-width={width}",
                "-assume_charset=utf-8",
                "-display_charset=utf-8",
                "-localhost",
                "-force_html",
            ],
            input=html.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        ).stdout
    )


@daft.func
def w3m_extract(html: str, width: int, timeout: int) -> str:
    """
    UltraData-Math: https://arxiv.org/abs/2602.09003
    """
    return decode_html(
        subprocess.run(
            [
                "w3m",
                "-dump",
                "-T",
                "text/html",
                "-cols",
                str(width),
                "-I",
                "UTF-8",
                "-O",
                "UTF-8",
            ],
            input=html.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        ).stdout
    )


class ParseMath:
    def __init__(
        self,
        input_column: str = "html",
        output_column: str = "text",
        timeout: int = 30,
        extractor_type: str = "w3m",
        width: int = 10000,
        name: str = "ParseMath",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.timeout = timeout
        self.extractor_type = extractor_type
        self.width = width
        self.name = name

        if self.extractor_type == "lynx" and not shutil.which("lynx"):
            raise RuntimeError("Lynx browser not found.")

        if self.extractor_type == "w3m" and not shutil.which("w3m"):
            raise RuntimeError("w3m browser not found.")

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.with_column("cleaned_html", magic_parse(col(self.input_column)))
        if self.extractor_type == "lynx":
            df = df.with_column(
                "text",
                lynx_extract(
                    col("cleaned_html"), width=self.width, timeout=self.timeout
                ),
            )
        elif self.extractor_type == "w3m":
            df = df.with_column(
                "text",
                w3m_extract(
                    col("cleaned_html"), width=self.width, timeout=self.timeout
                ),
            )
        else:
            raise ValueError(f"Extractor not available: {self.extractor_type}")
        return df.exclude("cleaned_html")
