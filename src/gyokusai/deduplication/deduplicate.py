import argparse
import os
import shutil
import struct

import daft
from daft import col
from daft.functions import when


@daft.func()
def band_keys(sig: list[int], b: int, r: int) -> list[bytes]:
    return [struct.pack(f"<B{r}Q", i, *sig[i * r : (i + 1) * r]) for i in range(b)]


def _publish(tmp: str, final: str):
    os.rename(tmp, final)


def _tmp(final: str) -> str:
    t = final + ".tmp"
    if os.path.isdir(t):
        shutil.rmtree(t)
    return t


def _canon(e):
    return (
        e.select(
            when(col("u") < col("v"), col("u")).otherwise(col("v")).alias("a"),
            when(col("u") < col("v"), col("v")).otherwise(col("u")).alias("b"),
        )
        .select(col("a").alias("u"), col("b").alias("v"))
        .where(col("u") != col("v"))
        .distinct()
    )


def _large_star(e):
    und = e.select("u", "v").union_all(
        e.select(col("v").alias("u"), col("u").alias("v"))
    )
    m = und.groupby("u").agg(col("v").min().alias("mn"))
    m = m.select(
        "u", when(col("u") < col("mn"), col("u")).otherwise(col("mn")).alias("m")
    )
    return (
        und.join(m, on="u")
        .where(col("v") > col("u"))
        .select(col("v").alias("u"), col("m").alias("v"))
        .where(col("u") != col("v"))
        .distinct()
        .collect()
    )


def _small_star(e):
    d = _canon(e).select(
        col("v").alias("u"), col("u").alias("v")
    )  # u = larger endpoint
    j = d.join(d.groupby("u").agg(col("v").min().alias("m")), on="u").collect()
    return (
        j.select(col("v").alias("u"), col("m").alias("v"))
        .union_all(
            j.select("u", col("m").alias("v"))
        )  # keep u attached to m (paper's N(u) ∪ {u})
        .where(col("u") != col("v"))
        .distinct()
        .collect()
    )


def _same(a, b) -> bool:
    return (
        a.join(b, on=["u", "v"], how="anti").count_rows() == 0
        and b.join(a, on=["u", "v"], how="anti").count_rows() == 0
    )


def fuzzy_dedupe(
    input_path,
    work,
    output_dir,
    key="doc_id",
    text="paragraph_text",
    num_hashes=112,
    ngram=5,
    seed=42,
    b=14,
    r=8,
):
    assert b * r == num_hashes
    daft.set_runner_ray()
    sigs, edges_dir, removal_dir = f"{work}/sigs", f"{work}/edges", f"{work}/removal"

    (
        daft.read_parquet(
            input_path,
            file_path_column="source_path",
            checkpoint=daft.CheckpointConfig(
                daft.CheckpointStore(f"{work}/ckpt_sig"), on="source_path"
            ),
        )
        .select(
            col(key),
            col(text)
            .normalize(
                remove_punct=True, lowercase=True, nfd_unicode=True, white_space=True
            )
            .minhash(
                num_hashes=num_hashes,
                ngram_size=ngram,
                seed=seed,
                hash_function="xxhash",
            )
            .alias("sig"),
        )
        .where(col("sig").not_null())
        .write_parquet(sigs)
    )

    if not os.path.isdir(edges_dir):
        bands = (
            daft.read_parquet(sigs)
            .select(col(key), band_keys(col("sig"), b, r).alias("bk"))
            .explode("bk")
        )
        grouped = (
            bands.groupby("bk")
            .agg(col(key).min().alias("dst"), col(key).count().alias("n"))
            .where(col("n") > 1)
        )
        t = _tmp(edges_dir)
        (
            bands.join(grouped, on="bk")
            .where(col(key) != col("dst"))
            .select(col(key).alias("u"), col("dst").alias("v"))
            .distinct()
            .write_parquet(t)
        )
        _publish(t, edges_dir)

    if not os.path.isdir(removal_dir):
        e = daft.read_parquet(edges_dir).collect()
        for _ in range(100):
            e2 = _small_star(_large_star(e))
            if _same(_canon(e), _canon(e2)):
                e = e2
                break
            e = e2
        und = e.union_all(e.select(col("v").alias("u"), col("u").alias("v")))
        rep = und.groupby("u").agg(col("v").min().alias("rep"))
        t = _tmp(removal_dir)
        rep.where(col("rep") < col("u")).select(col("u").alias(key)).write_parquet(t)
        _publish(t, removal_dir)

    t = _tmp(output_dir)
    (
        daft.read_parquet(input_path)
        .join(daft.read_parquet(removal_dir), on=key, how="anti")
        .write_parquet(t)
    )
    _publish(t, output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_path")
    p.add_argument("work_dir")
    p.add_argument("output_dir")
    p.add_argument("--key", default="doc_id")
    p.add_argument("--text", default="paragraph_text")
    a = p.parse_args()
    fuzzy_dedupe(a.input_path, a.work_dir, a.output_dir, key=a.key, text=a.text)
