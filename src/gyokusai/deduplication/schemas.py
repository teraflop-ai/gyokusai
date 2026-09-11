from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from pyfastbloom import BloomFilter


class BloomDeduplicator(ABC):
    def __init__(self, capacity: int, fpr: float = 1e-6):
        self.seen = BloomFilter(expected_items=capacity, false_positive_rate=fpr)

    def _duplicate(self, value: str) -> bool:
        if not isinstance(value, str):
            raise ValueError("Deduplication keys must be non-null strings")
        if value in self.seen:
            return True
        self.seen.insert(value)
        return False

    @abstractmethod
    def process(self, row: dict) -> dict | None:
        pass

    def run(self, rows: Iterable[dict]) -> Iterator[dict]:
        for row in rows:
            result = self.process(row)
            if result is not None:
                yield result
