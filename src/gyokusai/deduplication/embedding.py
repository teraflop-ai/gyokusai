import glob
import os

import cudf
import cugraph
import daft
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from daft import col
from flash_kmeans import FlashKMeans


class Dedup:
    def __init__(
        self,
        src,
        dst,
        d: int = 512,
        k: int = 100,
        thr: float = 0.95,
        tile: int = 8192,
        lam: float = 1.0,
    ):
        self.src = src
        self.dst = dst
        self.d = d
        self.k = k
        self.thr = thr
        self.tile = tile
        self.lam = lam
        self.tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        self.n = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        self.files = sorted(glob.glob(f"{src}/*.parquet"))

    def to_x(self, col_):
        ch = col_.chunks if isinstance(col_, pa.ChunkedArray) else [col_]
        x = torch.cat(
            [torch.from_numpy(c.flatten().to_numpy()).view(-1, self.d) for c in ch]
        )
        return torch.nn.functional.normalize(x.float(), dim=1).half()

    def dup_pairs(self, x):
        x = torch.nn.functional.normalize(x.cuda().half(), dim=1)
        N, T, out = x.shape[0], self.tile, []
        for a in range(0, N, T):
            for b in range(a, N, T):
                s = x[a : a + T] @ x[b : b + T].T
                i, j = (s >= self.thr).nonzero(as_tuple=True)
                keep = (j + b) > (i + a)
                out.append(torch.stack([i[keep] + a, j[keep] + b], 1).cpu())
        return torch.cat(out) if out else torch.empty(0, 2, dtype=torch.long)

    def soar(self, x, C):
        d = 2 - 2 * (x @ C.T)
        c1 = d.argmin(1)
        r = torch.nn.functional.normalize(x - C[c1], dim=1)
        proj = (x * r).sum(1, keepdim=True) - r @ C.T
        s = d + self.lam * proj**2
        s[torch.arange(len(x)), c1] = float("inf")
        return torch.stack([c1, s.argmin(1)], 1)

    def train(self):
        sample = self.files[:: max(1, len(self.files) // 64)]
        x = self.to_x(
            pq.read_table(sample, columns=["embedding"])["embedding"]
        ).pin_memory()
        km = FlashKMeans(d=self.d, k=self.k, niter=25, dtype=torch.float16, device=None)
        km.train(x)
        torch.save(km.centroids_b[0].cpu(), f"{self.dst}/centroids.pt")

    def assign(self):
        C = torch.nn.functional.normalize(
            torch.load(f"{self.dst}/centroids.pt").cuda().float(), dim=1
        )

        @daft.func.batch(
            return_dtype=daft.DataType.list(daft.DataType.int32()),
            batch_size=1_000_000,
            gpus=1,
            use_process=False,
        )
        def part(v: daft.Series) -> list:
            return (
                self.soar(self.to_x(v.to_arrow()).cuda().float(), C)
                .int()
                .cpu()
                .tolist()
            )

        (
            daft.read_parquet(self.files[self.tid :: self.n])
            .with_column("part", part(col("embedding")))
            .explode(col("part"))
            .write_parquet(f"{self.dst}/parts/task={self.tid}", partition_cols=["part"])
        )

    def dedup(self):
        for p in range(self.k)[self.tid :: self.n]:
            t = pq.read_table(
                glob.glob(f"{self.dst}/parts/task=*/part={p}/*.parquet"),
                columns=["uuid", "embedding"],
            )
            pairs = self.dup_pairs(self.to_x(t["embedding"]))
            if len(pairs) == 0:
                continue
            uuid = cudf.Series(t["uuid"].to_numpy())
            g = cugraph.Graph()
            g.from_cudf_edgelist(
                cudf.DataFrame(
                    {"src": pairs[:, 0].numpy(), "dst": pairs[:, 1].numpy()}
                ),
                "src",
                "dst",
            )
            cc = cugraph.weakly_connected_components(g)
            cc["uuid"] = uuid.iloc[cc["vertex"]].reset_index(drop=True)
            drop = cc[cc["uuid"] != cc.groupby("labels")["uuid"].transform("min")][
                "uuid"
            ]
            pq.write_table(
                pa.table({"uuid": drop.to_arrow()}), f"{self.dst}/drop/{p}.parquet"
            )
