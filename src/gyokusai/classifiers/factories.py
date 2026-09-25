from typing import Optional


def attribute_model_factory(
    *,
    checkpoint_path: str,
    label: str,
    task: str,
    batch_size: int = 32,
    gpus: int | float = 0,
    cpus: Optional[float] = None,
    max_concurrency: Optional[int] = None,
):
    import daft
    from daft import DataType, Series

    if task not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"Unknown task: {task}")

    @daft.cls(gpus=gpus, cpus=cpus, max_concurrency=max_concurrency)
    class AttributeModelClassifier:
        def __init__(self):
            import torch

            self.device = "cuda" if gpus else "cpu"
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.weight = ckpt["state_dict"]["probe.weight"]
            self.bias = ckpt["state_dict"]["probe.bias"]
            self.index = ckpt["label2id"][label]

        @daft.method.batch(return_dtype=DataType.float64(), batch_size=batch_size)
        def score_batch(self, embeddings: Series) -> list[float]:
            import numpy as np
            import torch

            with torch.inference_mode():
                inputs = torch.from_numpy(np.stack(embeddings.to_pylist())).to(
                    self.device, torch.float32
                )
                logits = torch.nn.functional.linear(inputs, self.weight, self.bias)
                p = logits.softmax(-1) if task == "multiclass" else logits.sigmoid()
                if task == "binary":
                    p = torch.cat((1 - p, p), dim=-1)
                return p[:, self.index].cpu().tolist()

    return AttributeModelClassifier().score_batch
