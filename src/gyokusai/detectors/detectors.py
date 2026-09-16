import daft
from lxml import etree
from lxml import html as lhtml
from py_asciimath.translator.translator import MathML2Tex
from pylatexenc.latexwalker import LatexWalker, LatexWalkerParseError

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


@daft.cls(use_process=True)
class MathDetector:
    def __init__(self):
        self.parser = lhtml.HTMLParser(encoding="utf-8")
        self.converter = MathML2Tex()
        self.candidates = etree.XPath(
            "(//math | //annotation[@encoding='application/x-tex'] | "
            "//span[contains(concat(' ', normalize-space(@class), ' '), "
            "' math-formula ')] | "
            "//script[@type='math/tex' or @type='math/latex' or "
            "@type='math/asciimath' or @type='math/mml'])"
            "[not(ancestor::math)]"
        )

    def contains(self, html: str | None) -> bool:
        if not html:
            return False

        try:
            root = lhtml.document_fromstring(html.encode("utf-8"), parser=self.parser)
        except (etree.ParserError, etree.XMLSyntaxError):
            return False

        found = False
        for node in self.candidates(root):
            if node.tag == "script" and node.get("type") == "math/mml":
                try:
                    node = lhtml.fromstring(node.text_content(), parser=self.parser)
                except (etree.ParserError, etree.XMLSyntaxError):
                    continue
                if node.tag != "math":
                    continue

            if node.tag == "math":
                annotations = node.xpath(".//annotation[@encoding='application/x-tex']")
                if annotations:
                    formula = annotations[0].text_content()
                else:
                    node.set("xmlns", "http://www.w3.org/1998/Math/MathML")
                    markup = etree.tostring(node, encoding="unicode", with_tail=False)
                    try:
                        formula = self.converter.translate(
                            markup, network=False, from_file=False
                        )
                    except Exception:
                        continue
                    formula = formula.replace("\ufeff", "").strip().strip("$")
            else:
                formula = node.text_content()

            formula = formula.strip()
            if not formula:
                continue

            try:
                LatexWalker(formula, tolerant_parsing=False).get_latex_nodes(pos=0)
            except LatexWalkerParseError:
                return False

            found = True

        return found
