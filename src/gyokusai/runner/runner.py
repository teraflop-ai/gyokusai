import faulthandler
import os
import socket
import time

os.environ.setdefault("RAY_DEDUP_LOGS", "0")

import daft


def configure_logging():
    from daft.logging import setup_logger

    faulthandler.enable(all_threads=True)
    setup_logger(level="DEBUG", daft_only=True, datefmt="%Y-%m-%d %H:%M:%S")


class SlurmRunner:
    def __init__(
        self,
        input: str,
        output: str,
        num_tasks: int | None = None,
        extension: str = ".parquet",
    ):
        env = os.environ.get
        self.task_id = int(env("SLURM_ARRAY_TASK_ID", 0))
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
        started = time.monotonic()
        configure_logging()
        print(
            f"Task {self.task_id}/{self.num_tasks}: "
            f"host={socket.gethostname()} pid={os.getpid()}",
            flush=True,
        )
        try:
            print("Selecting input files", flush=True)
            files = self.files()
            print(f"Selected {len(files)} files", flush=True)
            if not files:
                return

            if use_ray:
                import ray

                print("Starting Ray", flush=True)
                ray.init(
                    ignore_reinit_error=True,
                    log_to_driver=True,
                    runtime_env={"worker_process_setup_hook": configure_logging},
                )
                daft.set_runner_ray()
                print(f"Ray resources: {ray.cluster_resources()}", flush=True)

            print(f"Executing pipeline; writing to {self.output}", flush=True)
            pipeline(daft.read_parquet(files)).write_parquet(
                self.output, write_mode="overwrite"
            )
        except Exception:
            print(f"FAILED task {self.task_id}", flush=True)
            raise

        print(
            f"Completed task {self.task_id} in {time.monotonic() - started:.1f}s",
            flush=True,
        )
