from abc import ABC, abstractmethod

import daft


class BaseDataLoader(ABC):
    @abstractmethod
    def read_data(self, input_path: str) -> daft.DataFrame: ...
