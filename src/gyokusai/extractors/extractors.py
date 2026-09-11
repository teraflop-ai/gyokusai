import shutil

import daft
from magic_html import GeneralExtractor
from resiliparse.extract.html2text import extract_plain_text
from trafilatura import extract

from .schemas import BaseExtractor
from .utils import run_process


@daft.cls(use_process=True)
class TrafilaturaExtractor(BaseExtractor):
    def __init__(
        self,
        favor_precision: bool = False,
        favor_recall: bool = False,
    ):
        self.favor_precision = favor_precision
        self.favor_recall = favor_recall

    def text_extraction(self, html: str) -> str:
        text = extract(
            filecontent=html,
            favor_precision=self.favor_precision,
            favor_recall=self.favor_recall,
        )
        return text


@daft.cls(use_process=True)
class MagicHTMLExtractor(BaseExtractor):
    def __init__(self):
        self.magic = GeneralExtractor()

    def text_extraction(self, html: str) -> str:
        text = self.magic.extract(html)["html"]
        return text


@daft.cls(use_process=True)
class ResiliparseExtractor(BaseExtractor):
    def __init__(
        self,
        preserve_formatting: bool = True,
        main_content: bool = True,
    ):
        self.preserve_formatting = preserve_formatting
        self.main_content = main_content

    def text_extraction(self, html: str) -> str:
        text = extract_plain_text(
            html,
            preserve_formatting=self.preserve_formatting,
            main_content=self.main_content,
        )
        return text


@daft.cls
class LynxExtractor(BaseExtractor):
    """
    Nemotron-CC-Math: https://arxiv.org/abs/2508.15096
    """

    def __init__(
        self,
        width,
        timeout,
    ):
        if not shutil.which("lynx"):
            raise RuntimeError("Lynx browser not found.")

        self.width = width
        self.timeout = timeout

    def text_extraction(self, html: str) -> str:
        text = run_process(
            [
                "lynx",
                "-dump",
                "-stdin",
                "-nolist",
                f"-width={self.width}",
                "-assume_charset=utf-8",
                "-display_charset=utf-8",
                "-localhost",
                "-force_html",
            ],
            html,
            self.timeout,
        )
        return text


@daft.cls
class W3MExtractor(BaseExtractor):
    """
    UltraData-Math: https://arxiv.org/abs/2602.09003
    """

    def __init__(
        self,
        width,
        timeout,
    ):
        self.width = width
        self.timeout = timeout

        if not shutil.which("w3m"):
            raise RuntimeError("w3m browser not found.")

    def text_extraction(self, html: str) -> str:
        text = run_process(
            [
                "w3m",
                "-dump",
                "-T",
                "text/html",
                "-cols",
                str(self.width),
                "-I",
                "UTF-8",
                "-O",
                "UTF-8",
            ],
            html,
            self.timeout,
        )
        return text


@daft.cls
class ElinksExtractor(BaseExtractor):
    def __init__(self, width: int, timeout: float):
        if not shutil.which("elinks"):
            raise RuntimeError("elinks browser not found.")
        self.width = width
        self.timeout = timeout

    def text_extraction(self, html: str) -> str:
        text = run_process(
            [
                "elinks",
                "-dump",
                "1",
                "-dump-width",
                str(self.width),
                "-dump-charset",
                "utf-8",
                "-no-references",
                "1",
                "-no-numbering",
                "1",
                "-force-html",
                "1",
                "/dev/stdin",
            ],
            html,
            self.timeout,
        )
        return text


EXTRACTORS = {
    "trafilatura": TrafilaturaExtractor,
    "resiliparse": ResiliparseExtractor,
    "lynx": LynxExtractor,
    "w3m": W3MExtractor,
    "elinks": ElinksExtractor,
}
