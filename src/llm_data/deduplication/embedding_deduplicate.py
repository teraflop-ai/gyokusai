import numpy as np
import usearch.index as usearch_index

from daft import col, DataFrame


class UnionFind:

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)
 
    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root: 
            self.parent[x], x = root, self.parent[x]
        return root
 
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
 
 
def find_duplicate_groups(
    embeddings: np.ndarray,
    sim_threshold: float,
    k_neighbors: int,
) -> np.ndarray:
    n = embeddings.shape[0]
    index = usearch_index.Index(ndim=embeddings.shape[1], metric="cos", dtype="f32")
    index.add(np.arange(n), embeddings, threads=0)
 
    # +1 because the nearest neighbor of any point already in the index
    # is itself (distance 0); filtered below
    matches = index.search(embeddings, count=k_neighbors + 1, threads=0)
 
    uf = UnionFind(n)
    for i, row in enumerate(matches):
        for m in row:
            j = int(m.key)
            if j == i:
                continue
            sim = 1.0 - float(m.distance) 
            if sim > sim_threshold:
                uf.union(i, j)
 
    return np.array([uf.find(i) for i in range(n)])
 
 
def pick_representatives(group_ids: np.ndarray, text_lengths: np.ndarray) -> np.ndarray:
    """For each duplicate group, keep the row with the longest text.
    Returns a boolean keep-mask aligned with the original row order."""
    n = len(group_ids)
    keep = np.zeros(n, dtype=bool)
    # argsort by (group, -length) so the first row of each group in the
    # sorted order is the longest; avoids a Python-level loop over groups.
    order = np.lexsort((-text_lengths, group_ids))
    sorted_groups = group_ids[order]
    is_first_in_group = np.empty(n, dtype=bool)
    is_first_in_group[0] = True
    is_first_in_group[1:] = sorted_groups[1:] != sorted_groups[:-1]
    keep[order[is_first_in_group]] = True
    return keep


class EmbeddingDeduper:
    def __init__(
        self,
        embedding_column: str='text_embedding',
        text_column: str='text',
        id_column: str='record_id',
        sim_threshold: float=0.93,
        k_neighbors: int=10
    ):
        self.embedding_column = embedding_column
        self.text_column = text_column
        self.id_column = id_column
        self.sim_threshold = sim_threshold
        self.k_neighors = k_neighbors

    def __call__(self, df: DataFrame) -> DataFrame:
        cols = df.select(self.id_column, self.text_column, self.embedding_colum).to_pydict()
        embeddings = np.asarray(self.embedding_colum, dtype=np.float32)
        text_lengths = np.array([len(t) for t in cols[self.text_column]])
    
        group_ids = find_duplicate_groups(embeddings, self.sim_threshold, self.k_neighbors)
        keep_mask = pick_representatives(group_ids, text_lengths)
        keep_ids = set(np.asarray(cols[self.id_column])[keep_mask].tolist())
        deduped = df.where(col(self.id_column).is_in(list(keep_ids)))
        return deduped
        