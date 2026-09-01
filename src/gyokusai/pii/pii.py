import daft
from daft import DataFrame, col
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


@daft.cls
class PresidioMasker:
    def __init__(self, language: str = "en"):
        # Initialized once per worker process
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        self.language = language

    @daft.method
    def mask_text(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language=self.language)
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text


class PIIAnonymizer:
    def __init__(
        self,
        input_column: str = "text",
        language: str = "en",
        name: str = "PIIAnonymizer",
    ):
        self.input_column = input_column
        self.language = language
        self.name = name

        self.masker = PresidioMasker(language=language)

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.with_column(
            self.input_column, self.masker.mask_text(col(self.input_column))
        )
        return df
