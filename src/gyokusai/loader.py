from abc import ABC, abstractmethod
from typing import Optional

import daft
from daft import CheckpointConfig, CheckpointStore, KeyFilteringSettings

from gyokusai.utils import checkpoint_uri


class DataLoader(ABC):
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        file_path_column_name: Optional[str] = "source_path",
        checkpoint_on: Optional[str] = "source_path",
        num_workers: Optional[int] = None,
        cpus_per_worker: Optional[float] = None,
        config: Optional[CheckpointConfig] = None,
    ):
        if config is not None and checkpoint_path is not None:
            raise ValueError("Pass either `config` or `checkpoint_path`, not both")

        if checkpoint_path is not None and checkpoint_on is None:
            raise ValueError(
                "`checkpoint_on` is required when `checkpoint_path` is set"
            )

        self.file_path_column_name = file_path_column_name
        self.checkpoint_path = (
            checkpoint_uri(checkpoint_path) if checkpoint_path else None
        )

        if self.checkpoint_path:
            config = CheckpointConfig(
                store=CheckpointStore(self.checkpoint_path),
                on=checkpoint_on,
                settings=KeyFilteringSettings(
                    num_workers=num_workers,
                    cpus_per_worker=cpus_per_worker,
                ),
            )
        self.config = config

    @property
    def _read_kwargs(self) -> dict:
        return {
            "checkpoint": self.config,
            "file_path_column": self.file_path_column_name,
        }

    @abstractmethod
    def read_data(self, input_path: str) -> daft.DataFrame: ...


class ParquetLoader(DataLoader):
    def read_data(self, input_path: str) -> daft.DataFrame:
        return daft.read_parquet(input_path, **self._read_kwargs)


class JsonLoader(DataLoader):
    def read_data(self, input_path: str) -> daft.DataFrame:
        return daft.read_json(input_path, **self._read_kwargs)


class CsvLoader(DataLoader):
    def read_data(self, input_path: str) -> daft.DataFrame:
        return daft.read_csv(input_path, **self._read_kwargs)


class WarcLoader(DataLoader):
    def read_data(self, input_path: str) -> daft.DataFrame:
        return daft.read_warc(input_path, **self._read_kwargs)


class LanceLoader(DataLoader):
    def read_data(self, input_path: str) -> daft.DataFrame:
        return daft.read_lance(input_path, checkpoint=self.config)


class HuggingFaceLoader(DataLoader):
    def read_data(self, input_path: str) -> daft.DataFrame:
        return daft.read_huggingface(input_path)


LOADERS = {
    "parquet": ParquetLoader,
    "json": JsonLoader,
    "csv": CsvLoader,
    "warc": WarcLoader,
    "lance": LanceLoader,
    "huggingface": HuggingFaceLoader,
}
