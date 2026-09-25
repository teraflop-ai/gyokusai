import os

import daft


class SlurmRunner:
    def __init__(
        self,
        input: str,
        output: str,
        num_tasks: int | None = None,
        extension: str = ".parquet",
    ):
        env = os.environ.get
        self.task_id = int(env("SLURM_ARRAY_TASK_ID", 0)) - int(
            env("SLURM_ARRAY_TASK_MIN", 0)
        )
        self.num_tasks = num_tasks or int(env("SLURM_ARRAY_TASK_COUNT", 1))
        self.input = input
        self.extension = extension
        self.output = os.path.join(output, f"task_{self.task_id:05d}")

    @staticmethod
    def add_args(parser):
        parser.add_argument("input")
        parser.add_argument("output")
        parser.add_argument("num_tasks", type=int, nargs="?")
        return parser

    @classmethod
    def from_args(cls, args):
        kwargs = vars(args)
        return cls(
            kwargs.pop("input"), kwargs.pop("output"), kwargs.pop("num_tasks")
        ), kwargs

    def files(self) -> list[str]:
        files = sorted(
            os.path.join(root, f)
            for root, _, names in os.walk(self.input)
            for f in names
            if f.endswith(self.extension)
        )
        return sorted(files, key=os.path.getsize)[self.task_id :: self.num_tasks]

    def run(self, pipeline, use_ray: bool = True) -> None:
        files = self.files()
        print(f"Task {self.task_id}/{self.num_tasks}: {len(files)} files")
        if not files:
            return
        if use_ray:
            import ray

            ray.init(ignore_reinit_error=True)
            daft.set_runner_ray()
        pipeline(daft.read_parquet(files)).write_parquet(
            self.output, write_mode="overwrite"
        )
