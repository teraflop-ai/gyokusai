from daft import DataFrame, col

from gyokusai.detectors import (
    AIPhraseDetector,
    AIStyleDetector,
    AITraceDetector,
    CodeDetector,
    MathDetector,
)


class Code:
    def __init__(self, input_column: str = "html", output_column: str = "has_code"):
        self.input_column = input_column
        self.output_column = output_column
        self.detector = CodeDetector()

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.detector.contains(col(self.input_column))
        )


class Math:
    def __init__(self, input_column: str = "html", output_column: str = "has_math"):
        self.input_column = input_column
        self.output_column = output_column
        self.detector = MathDetector()

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.detector.contains(col(self.input_column))
        )


class AIPhrase:
    def __init__(
        self, input_column: str = "text", output_column: str = "has_ai_phrase"
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.detector = AIPhraseDetector()

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.detector.contains(col(self.input_column))
        )


class AIStyle:
    def __init__(
        self,
        min_hits: int = 3,
        min_per_kchar: float = 0.5,
        input_column: str = "text",
        output_column: str = "has_ai_style",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.detector = AIStyleDetector(min_hits, min_per_kchar)

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.detector.contains(col(self.input_column))
        )


class AITrace:
    def __init__(self, input_column: str = "html", output_column: str = "has_ai_trace"):
        self.input_column = input_column
        self.output_column = output_column
        self.detector = AITraceDetector()

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column, self.detector.contains(col(self.input_column))
        )
