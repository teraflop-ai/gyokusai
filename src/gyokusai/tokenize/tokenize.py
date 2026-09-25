import daft
from daft import DataType, Series

from .schemas import BaseTokenizer


@daft.cls(max_concurrency=1, use_process=True)
class TokenizeText(BaseTokenizer):
    @daft.method.batch(return_dtype=DataType.list(DataType.int32()))
    def tokenize(self, text: Series) -> Series:
        import pyarrow as pa

        return Series.from_arrow(self._encode(text).cast(pa.large_list(pa.int32())))


@daft.cls(max_concurrency=1, use_process=True)
class FastTextTokenizeText(BaseTokenizer):
    def __init__(self, model: str = "Qwen/Qwen3.5-9B"):
        import pyarrow as pa

        super().__init__(model)
        self.vocab = pa.array(
            self.hf.convert_ids_to_tokens(list(range(len(self.hf)))), pa.large_string()
        )

    @daft.method.batch(return_dtype=DataType.string())
    def tokenize(self, text: Series) -> Series:
        import pyarrow as pa
        import pyarrow.compute as pc

        ids = self._encode(text)
        words = pa.LargeListArray.from_arrays(ids.offsets, self.vocab.take(ids.values))
        return Series.from_arrow(
            pc.binary_join(words, pa.scalar(" ", pa.large_string()))
        )
