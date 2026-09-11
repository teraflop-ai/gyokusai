from daft import DataFrame, col

from .extractors import EXTRACTORS, MagicHTMLExtractor


class ExtractHTML:
    def __init__(
        self,
        input_column: str = "html",
        output_column: str = "text",
        extractor_type: str = "resiliparse",
        clean: bool = False,
        name: str = "ExtractHTML",
        **extractor_kwargs,
    ):
        if extractor_type not in EXTRACTORS:
            raise ValueError(f"Extractor not available: {extractor_type}")
        self.input_column = input_column
        self.output_column = output_column
        self.extractor = EXTRACTORS[extractor_type](**extractor_kwargs)
        self.cleaner = MagicHTMLExtractor()
        self.clean = clean
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        html = col(self.input_column)
        if self.clean:
            html = self.cleaner.text_extraction(html)
        return df.with_column(self.output_column, self.extractor.text_extraction(html))
