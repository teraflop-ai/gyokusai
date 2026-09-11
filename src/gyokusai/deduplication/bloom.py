from .schemas import BloomDeduplicator


class URLDeduplicator(BloomDeduplicator):
    def process(self, row: dict) -> dict | None:
        return None if self._duplicate(row["url"]) else row


class DocumentDeduplicator(BloomDeduplicator):
    def process(self, row: dict) -> dict | None:
        return None if self._duplicate(row["text"]) else row


class ParagraphDeduplicator(BloomDeduplicator):
    def process(self, row: dict) -> dict | None:
        text = row["text"]
        if not isinstance(text, str):
            raise ValueError("text must be a non-null string")
        kept = [p for p in text.split("\n") if not self._duplicate(p)]
        return {**row, "text": "\n".join(kept)} if kept else None
