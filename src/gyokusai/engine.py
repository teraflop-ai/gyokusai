from dataclasses import field

from daft import DataFrame


class DataEngine:
    def __init__(
        self,
        components: list = field(default_factory=list),
        name: str = "DataEngine",
    ):
        self.components = components
        self.name = name

    def add(self, name):
        self.components.append(name)
        return self

    def run(self, df: DataFrame = None) -> DataFrame:
        for component in self.components:
            df = component(df)
        return df

    def __repr__(self) -> str:
        components = ", ".join(getattr(c, "name") for c in self.components)
        return f"DataEngine(name={self.name!r}, components=[{components}])"
