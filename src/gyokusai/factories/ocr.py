from daft import DataFrame, col


def ocr_factory(
    *,
    model_name_or_path: str,
    batch_size: int = 4,
    gpus: int | float = 1,
):
    import daft
    from PIL import Image
    from vllm import LLM, SamplingParams

    return_dtype = daft.DataType.string()

    @daft.cls(gpus=gpus)
    class OvisOCR2Parser:
        def __init__(self):
            self.model = LLM(
                model=model_name_or_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                gdn_prefill_backend="triton",
            )

            prompt = (
                "\nExtract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <"
                + 'img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing.'
            )
            self.prompt = self.model.get_tokenizer().apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            self.sampling_params = SamplingParams(max_tokens=16384, temperature=0.0)

        def _clean_truncated_repeats(
            self,
            text: str,
            min_text_len: int = 8000,
            max_period: int = 200,
            min_period: int = 1,
            min_repeat_chars: int = 100,
            min_repeat_times: int = 5,
        ) -> str:
            n = len(text)
            if n < min_text_len:
                return text

            max_period = min(max_period, n - 1)
            for unit_len in range(min_period, max_period + 1):
                if text[n - 1] != text[n - 1 - unit_len]:
                    continue

                match_len = 1
                idx = n - 2
                while idx >= unit_len and text[idx] == text[idx - unit_len]:
                    match_len += 1
                    idx -= 1

                total_len = match_len + unit_len
                repeat_times = total_len // unit_len
                tail_len = total_len % unit_len

                if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
                    return text[: n - total_len + unit_len] + text[n - tail_len :]

            return text

        @daft.method.batch(return_dtype=return_dtype, batch_size=batch_size)
        def parse(
            self, images: list[Image.Image], filter_imgtags: bool = True
        ) -> list[str]:
            vllm_inputs = [
                {
                    "prompt": self.prompt,
                    "multi_modal_data": {"image": image},
                    "mm_processor_kwargs": {
                        "images_kwargs": {
                            "min_pixels": 448 * 448,
                            "max_pixels": 2880 * 2880,
                        }
                    },
                }
                for image in images
            ]

            outputs = self.model.generate(vllm_inputs, self.sampling_params)

            markdowns = []
            for output in outputs:
                text = output.outputs[0].text.strip()
                if filter_imgtags:
                    text = "\n\n".join(
                        block
                        for block in text.split("\n\n")
                        if not block.strip().startswith('<img src="images/bbox_')
                    )
                markdowns.append(self._clean_truncated_repeats(text))

            return markdowns

    return OvisOCR2Parser()


class OCRText:
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 4,
        gpus: int | float = 1,
        image_col: str = "image",
        output_col: str = "markdown",
        name: str = "OCRText",
    ):
        self.ocr_batch = ocr_factory(
            model_name_or_path=model_name_or_path, batch_size=batch_size, gpus=gpus
        )
        self.image_col = image_col
        self.output_col = output_col
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_col, self.ocr_batch.parse(col(self.image_col))
        )
