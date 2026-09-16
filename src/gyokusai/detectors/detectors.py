import daft
from lxml import etree
from lxml import html as lhtml

from .regexes import CODE_PATTERN


@daft.cls(use_process=True)
class CodeDetector:
    def __init__(self):
        self.parser = lhtml.HTMLParser(encoding="utf-8")

    def contains(self, html: str | None) -> bool:
        if not html:
            return False

        try:
            root = lhtml.document_fromstring(html.encode("utf-8"), parser=self.parser)
        except (etree.ParserError, etree.XMLSyntaxError):
            return False

        seen = set()
        for node in root.xpath("//code[parent::pre or parent::tbody]"):
            block = node.getparent()
            if block.tag == "pre":
                block = next(block.iterancestors("tbody"), block)
            if block in seen:
                continue
            seen.add(block)

            if CODE_PATTERN.search(block.text_content()):
                return True

        return False
