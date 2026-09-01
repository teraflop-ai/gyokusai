from typing import Optional

import daft
from daft import DataFrame, col

DEFAULT_INSTRUCTION = (
    "Extract the high-quality informational content from the following raw web page text. "
    "Remove boilerplate, navigation, ads, and duplicated fragments. "
    "Output only the cleaned extract, preserving the original wording. "
    "If there is no substantive content, output nothing."
)


def generation_factory(
    *,
    model_path: str,
    instruction: str = DEFAULT_INSTRUCTION,
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
):
    from daft import DataType, Series

    @daft.cls(gpus=gpus, cpus=cpus, max_concurrency=max_concurrency)
    class TextGeneration:
        def __init__(self):
            import json

            import sglang as sgl
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.engine = sgl.Engine(
                model_path=model_path,
                trust_remote_code=True,
                kv_cache_dtype="fp8_e4m3",
                mem_fraction_static=mem_fraction_static,
                context_length=context_length,
                attention_backend="flashinfer",
                chunked_prefill_size=chunked_prefill_size,
                mamba_full_memory_ratio=1.87,
                mamba_radix_cache_strategy="extra_buffer_lazy",
                mamba_ssm_dtype="bfloat16",
                disable_cuda_graph=disable_cuda_graph,
                allow_auto_truncate=True,
            )
            self.sampling = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
            if json_schema is not None:
                self.sampling["json_schema"] = json.dumps(json_schema)

        @daft.method.batch(return_dtype=DataType.string(), batch_size=batch_size)
        def generate(self, docs: Series):
            prompts = [
                self.tok.apply_chat_template(
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": doc[:truncate_to]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for doc in docs.to_pylist()
            ]
            out = self.engine.generate(prompts, sampling_params=self.sampling)
            return [o["text"] for o in out]

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
