import argparse
import os

import daft
from daft import DataType, col
from daft.functions import coalesce, list_agg, to_list
from deltalake import DeltaTable


def _uri(p):
    return p if "://" in p else "file://" + os.path.abspath(p)


def _done(d):
    return DeltaTable.is_deltatable(d)


def _write(df, d):
    df.write_deltalake(d, mode="overwrite")


def _mins(e):
    return e.groupby("u").agg(col("v").min().alias("m"))


def _large_star(e, p):
    return (
        e.join(_mins(e).select(col("u").alias("v"), "m"), on="v", how="left")
        .select("u", coalesce(col("m"), col("v")).alias("v"))
        .repartition(p, "u", "v")
        .distinct()
        .collect()
    )


def _small_star(e, p):
    return (
        e.join(_mins(e), on="u")
        .select(to_list(col("u"), col("v")).alias("u"), col("m").alias("v"))
        .explode("u")
        .where(col("u") != col("v"))
        .repartition(p, "u", "v")
        .distinct()
        .collect()
    )


def _components(e):
    p = max(1, min(512, e.count_rows() // 2_000_000))
    while True:
        e2 = _small_star(_large_star(e, p), p)
        if (
            e2.count_rows() == e.count_rows()
            and e2.join(e, on=["u", "v"], how="anti").count_rows() == 0
        ):
            return e2
        e = e2


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
    partitions=1024,
):
    assert b * r == num_hashes
    daft.set_runner_ray(noop_if_initialized=True)
    daft.set_execution_config(
        shuffle_algorithm="flight_shuffle",
        shuffle_aggregation_default_partitions=partitions,
    )
    work, output_dir = _uri(work), _uri(output_dir)
    sigs, uniq, edges, removal = (
        f"{work}/{d}" for d in ("sigs", "uniq", "edges", "removal")
    )

    if not _done(sigs):
        store = daft.CheckpointStore(f"{work}/ckpt")
        (
            daft.read_parquet(
                input_path, checkpoint=daft.CheckpointConfig(store, on=key)
            )
            .select(
                col(key),
                col(text)
                .normalize(
                    remove_punct=True,
                    lowercase=True,
                    nfd_unicode=True,
                    white_space=True,
                )
                .minhash(
                    num_hashes=num_hashes,
                    ngram_size=ngram,
                    seed=seed,
                    hash_function="xxhash",
                )
                .cast(DataType.list(DataType.int64()))
                .alias("sig"),
            )
            .where(col("sig").not_null())
            .write_deltalake(sigs, checkpoint=daft.IdempotentCommit(store, "sigs"))
        )

    if not _done(uniq):
        _write(
            daft.read_deltalake(sigs).groupby("sig").agg(col(key).min().alias(key)),
            uniq,
        )

    if not _done(edges):
        _write(
            daft.read_deltalake(uniq)
            .select(
                col(key),
                to_list(
                    *[col("sig").slice(i * r, (i + 1) * r) for i in range(b)]
                ).alias("bk"),
            )
            .explode("bk", index_column="band")
            .groupby("band", "bk")
            .agg(col(key).min().alias("v"), list_agg(col(key)).alias("u"))
            .select("v", "u")
            .explode("u")
            .where(col("u") != col("v"))
            .distinct(),
            edges,
        )

    if not _done(removal):
        fuzzy = _components(daft.read_deltalake(edges).collect()).select(
            col("u").alias(key)
        )
        exact = (
            daft.read_deltalake(sigs)
            .select(key)
            .join(daft.read_deltalake(uniq).select(key), on=key, how="anti")
        )
        _write(exact.union_all(fuzzy), removal)

    if not _done(output_dir):
        _write(
            daft.read_parquet(input_path).join(
                daft.read_deltalake(removal), on=key, how="anti"
            ),
            output_dir,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_path")
    p.add_argument("work_dir")
    p.add_argument("output_dir")
    p.add_argument("--key", default="doc_id")
    p.add_argument("--text", default="paragraph_text")
    p.add_argument("--partitions", type=int, default=1024)
    a = p.parse_args()
    fuzzy_dedupe(
        a.input_path,
        a.work_dir,
        a.output_dir,
        key=a.key,
        text=a.text,
        partitions=a.partitions,
    )
