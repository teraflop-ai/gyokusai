from typing import TypedDict

import daft
import numpy as np

from daft import col, DataType, DataFrame
from sklearn.cluster import MiniBatchKMeans

from llm_data.factories.embedding import EmbedText
from ..config import EMBEDDING_DEDUPE_BATCH_SIZE, EMBEDDING_DEDUPE_DIM, EMBEDDING_DEDUPE_MODEL_NAME


class PruneResult(TypedDict):
    doc_id: str
    keep: bool


@daft.cls(max_concurrency=1, use_process=True)
class ClusterAssignerUDF:
 
    def __init__(self, centroids: np.ndarray) -> None:
        self.centroids = centroids  # (K, dim), L2-normalized
 
    @daft.method.batch(
        return_dtype=DataType.struct(
            {"cluster_id": DataType.int32(), "sim_to_centroid": DataType.float32()}
        ),
        batch_size=2048,
    )
    def assign(self, embeddings):
        vecs = np.asarray(embeddings, dtype=np.float32)  # (n, dim)
        sims = vecs @ self.centroids.T  # (n, K) cosine sim, both unit-norm
        cluster_id = np.argmax(sims, axis=1).astype(np.int32)
        sim_to_centroid = sims[np.arange(len(vecs)), cluster_id].astype(np.float32)
        return [
            {"cluster_id": int(c), "sim_to_centroid": float(s)}
            for c, s in zip(cluster_id, sim_to_centroid)
        ]


def fit_kmeans_centroids(
    df: DataFrame,
    n_clusters: int,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """ We follow SemDeDup here and fit means on a representative sample,
    followed by re-using centroids on the full pass (rather than fitting
    on the full sample).)
    """
    sample_rows = (
        df.select("embedding")
        .limit(sample_size)
        .to_pydict()["embedding"]
    )
    sample = np.asarray(sample_rows, dtype=np.float32)
 
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        batch_size=4096,
    )
    kmeans.fit(sample)
 
    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
    return centroids


def assign_clusters(df: DataFrame, centroids: np.ndarray) -> DataFrame:
    assigner = ClusterAssignerUDF(centroids)
    return (
        df.with_column("assignment", assigner.assign(col("embedding")))
        .with_column("cluster_id", col("assignment").get("cluster_id"))
        .with_column("sim_to_centroid", col("assignment").get("sim_to_centroid"))
        .exclude("assignment")
    )

 
def _semdedup_keep_mask(embeddings: np.ndarray, sim_to_centroid: np.ndarray, threshold: float,
                        max_block_size: int=4096) -> np.ndarray:
    n = embeddings.shape[0]
    order = np.argsort(-sim_to_centroid, kind="stable")  # descending
    sorted_embeddings = embeddings[order]
 
    keep_sorted = np.ones(n, dtype=bool)
 
    for start in range(0, n, max_block_size):
        end = min(start + max_block_size, n)
        block = sorted_embeddings[start:end]              # (b, dim)
        prior = sorted_embeddings[:end]                    # (end, dim), grows each block
        sims = block @ prior.T                              # (b, end)
        for local_i, global_i in enumerate(range(start, end)):
            # Compare only against strictly-earlier points (exclude self
            # and anything at/after this index).
            row = sims[local_i, :global_i]
            if row.size and row.max() > threshold:
                keep_sorted[global_i] = False
 
    keep = np.empty(n, dtype=bool)
    keep[order] = keep_sorted
    return keep
 
 
@daft.func
def prune_cluster(
    doc_id: list,
    embedding: list,
    sim_to_centroid: list,
    threshold: float = 0.93
) -> list[PruneResult]:
    ids = np.asarray(doc_id)
    sims = np.asarray(sim_to_centroid, dtype=np.float32)
    embs = np.asarray(embedding, dtype=np.float32)
 
    keep_mask = _semdedup_keep_mask(embs, sims, threshold)
 
    return [
        {"doc_id": str(i), "keep": bool(k)}
        for i, k in zip(ids.tolist(), keep_mask.tolist())
    ]
 
 
def semdedup_prune(df: DataFrame) -> DataFrame:
    grouped = (
        df.groupby("cluster_id")
        .list_agg("doc_id", "embedding", "sim_to_centroid")
    )
 
    pruned = (
        grouped.with_column(
            "results",
            prune_cluster(col("doc_id"), col("embedding"), col("sim_to_centroid")),
        )
        .select("cluster_id", "results")
        .explode("results")
        .with_column("doc_id", col("results").get("doc_id"))
        .with_column("keep", col("results").get("keep"))
        .select("doc_id", "keep")
    )
 
    deduped = (
        df.join(pruned, on="doc_id", how="inner")
        .where(col("keep"))
        .exclude("keep")
    )
    return deduped


class EmbeddingDeduper:
    def __init__(
        self,
        input_column: str='text',
        n_clusters: int=2000,
        kmeans_sample_size: int=200_000
    ):
        self.input_column = input_column
        self.embed_udf = EmbedText(input_column, 'embedding')
        self.n_clusters = n_clusters
        self.kmeans_sample_size = kmeans_sample_size

    def __call__(self, df: DataFrame) -> DataFrame:
        df = self.embed_udf(df)

        # Materializing so we don't recompute embeddings later
        df = df.collect()
        centroids = fit_kmeans_centroids(df, self.n_clusters, self.kmeans_sample_size, seed=0)
        df = assign_clusters(df, centroids).collect()
        deduped = semdedup_prune(df)
        return deduped