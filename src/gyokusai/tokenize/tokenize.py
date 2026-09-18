from abc import ABC, abstractmethod

import daft
from daft import DataType, Series


class BaseTokenizer(ABC):
    def __init__(self, model: str = "Qwen/Qwen3.5-9B"):
        import gigatoken as gt
        from transformers import AutoTokenizer

        self.hf = AutoTokenizer.from_pretrained(model)
        self.tok = gt.Tokenizer(self.hf).as_hf()

    @abstractmethod
    def tokenize(self, text: Series) -> Series: ...


@daft.cls(max_concurrency=1, use_process=True)
class TokenizeText(BaseTokenizer):
    @daft.method.batch(return_dtype=DataType.list(DataType.int64()))
    def tokenize(self, text: Series) -> Series:
        ids = self.tok(text.to_pylist(), return_attention_mask=False)["input_ids"]
        return Series.from_pylist(ids)


@daft.cls(max_concurrency=1, use_process=True)
class FastTextTokenizeText(BaseTokenizer):
    def __init__(self, model: str = "Qwen/Qwen3.5-9B"):
        super().__init__(model)
        self.id2tok = self.hf.convert_ids_to_tokens(list(range(len(self.hf))))

    @daft.method.batch(return_dtype=DataType.string())
    def tokenize(self, text: Series) -> Series:
        ids = self.tok(
            text.to_pylist(), return_attention_mask=False, add_special_tokens=False
        )["input_ids"]
        return Series.from_pylist(
            [" ".join(self.id2tok[i] for i in row) for row in ids]
        )
