from collections.abc import Iterator

import daft

from .schemas import BloomDeduplicator


class BloomDeduplicationStage:
    def __init__(self, *deduplicators: BloomDeduplicator):
        self.deduplicators = deduplicators

    def run(self, df: daft.DataFrame) -> Iterator[dict]:
        rows = df.iter_rows(results_buffer_size=1)
        for deduplicator in self.deduplicators:
            rows = deduplicator.run(rows)
        return rows
