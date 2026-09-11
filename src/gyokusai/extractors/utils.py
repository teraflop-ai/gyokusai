import subprocess


def run_process(cmd: list[str], html: str, timeout: float):
    try:
        return subprocess.run(
            cmd,
            input=html.encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=timeout,
        ).stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return None
