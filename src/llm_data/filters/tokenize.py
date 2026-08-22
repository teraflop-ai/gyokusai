import daft
from daft import col, DataType, Series

@daft.cls(max_concurrency=1, use_process=True)
class TokenCounter:
    def __init__(self, model: str = "Qwen/Qwen3.5-9B"):
        import gigatoken as gt
        from transformers import AutoTokenizer
        self.tok = gt.Tokenizer(AutoTokenizer.from_pretrained(model)).as_hf()

    @daft.method.batch(return_dtype=DataType.int64())
    def num_tokens(self, text: Series) -> Series:
        enc = self.tok(text.to_pylist())
        return Series.from_pylist([len(ids) for ids in enc["input_ids"]])


@daft.cls(max_concurrency=1, use_process=True)
class TokenizeText:
    def __init__(self, model: str = "Qwen/Qwen3.5-9B"):
        import gigatoken as gt
        from transformers import AutoTokenizer
        self.tok = gt.Tokenizer(AutoTokenizer.from_pretrained(model)).as_hf()

    @daft.method.batch(return_dtype=DataType.list(DataType.int64()))
    def tokenize(self, text: Series) -> Series:
        ids = self.tok(text.to_pylist(), return_attention_mask=False)["input_ids"]
        return Series.from_pylist(ids)