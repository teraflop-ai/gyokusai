from abc import ABC, abstractmethod

from daft import Expression


class BaseDetector(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or type(self).__name__

    @abstractmethod
    def contains(self, html: Expression) -> Expression: ...
