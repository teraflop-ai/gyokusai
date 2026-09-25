from daft import DataFrame


class DataEngine:
    def __init__(self, components: list | None = None, name: str = "DataEngine"):
        self.components = list(components or [])
        self.name = name

    def add(self, component):
        self.components.append(component)
        return self

    def run(self, df: DataFrame | None = None) -> DataFrame:
        for component in self.components:
            df = component(df)
        return df

    def __call__(self, df: DataFrame | None = None) -> DataFrame:
        return self.run(df)

    def __repr__(self) -> str:
        components = ", ".join(
            getattr(c, "name", type(c).__name__) for c in self.components
        )
        return f"DataEngine(name={self.name!r}, components=[{components}])"
