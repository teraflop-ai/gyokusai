from abc import ABC, abstractmethod

from daft import Series


class BaseTokenizer(ABC):
    def __init__(self, model: str = "Qwen/Qwen3.5-9B"):
        import gigatoken as gt
        from transformers import AutoTokenizer

        self.hf = AutoTokenizer.from_pretrained(model)
        self.tok = gt.Tokenizer(self.hf).as_hf()

    @abstractmethod
    def tokenize(self, text: Series) -> Series: ...
