from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """
    A base class for HTML Text Extractors.
    """

    @abstractmethod
    def text_extraction(self, html: str) -> str: ...
