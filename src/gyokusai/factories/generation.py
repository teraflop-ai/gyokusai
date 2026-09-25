from typing import Optional

import daft
from daft import DataFrame, col

from .config import GENERATION_KWARGS


def generation_factory(
    *,
    model_path: str,
    instruction: str,
    enable_thinking: bool = False,
    batch_size: int = 64,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    context_length: int = 32768,
    mem_fraction_static: float = 0.85,
    chunked_prefill_size: int = 8192,
    disable_cuda_graph: bool = False,
    gpus: int | float = 1,
    cpus: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    json_schema: Optional[dict] = None,
    truncate_to: Optional[int] = None,
    engine_kwargs: Optional[dict] = None,
    sampling_kwargs: Optional[dict] = None,
):
    from daft import DataType, Series

    engine_kwargs = {
        **GENERATION_KWARGS,
        "max_model_len": context_length,
        "gpu_memory_utilization": mem_fraction_static,
        "max_num_batched_tokens": chunked_prefill_size,
        "enforce_eager": disable_cuda_graph,
        **(engine_kwargs or {}),
    }
    sampling = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        **(sampling_kwargs or {}),
    }

    def messages(doc: str) -> list[dict]:
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": doc},
        ]

    @daft.cls(gpus=gpus, cpus=cpus, max_concurrency=max_concurrency)
    class TextGeneration:
        def __init__(self):
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            self.llm = LLM(model=model_path, **engine_kwargs)
            self.tok = self.llm.get_tokenizer()
            template = self.tok.apply_chat_template(
                messages(""),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            self.max_doc_tokens = (
                engine_kwargs["max_model_len"]
                - sampling["max_tokens"]
                - len(self.tok.encode(template, add_special_tokens=False))
            )
            self.params = SamplingParams(
                structured_outputs=StructuredOutputsParams(json=json_schema)
                if json_schema
                else None,
                **sampling,
            )

        @daft.method.batch(return_dtype=DataType.string(), batch_size=batch_size)
        def generate(self, docs: Series):
            ids = self.tok(
                [(d or "")[:truncate_to] for d in docs.to_pylist()],
                add_special_tokens=False,
            )["input_ids"]
            out = self.llm.chat(
                [messages(self.tok.decode(i[: self.max_doc_tokens])) for i in ids],
                self.params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
            return [o.outputs[0].text for o in out]

    return TextGeneration().generate


class GenerateText:
    def __init__(
        self,
        input_column: str = "text",
        output_column: str = "extract",
        name: str = "GenerateText",
        **factory_kwargs,
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.name = name
        self.generate = generation_factory(**factory_kwargs)

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(self.output_column, self.generate(col(self.input_column)))
