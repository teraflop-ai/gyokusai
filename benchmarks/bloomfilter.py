from time import perf_counter

import daft
import pyperf

from gyokusai.deduplication.bloom import URLDeduplicator

N = 10_000_000


def bench_url(loops, df, duplicates):
    elapsed = 0.0
    for _ in range(loops):
        dedup = URLDeduplicator(capacity=N)
        if duplicates:
            sum(1 for _ in dedup.run(df.iter_rows()))

        start = perf_counter()
        sum(1 for _ in dedup.run(df.iter_rows()))
        elapsed += perf_counter() - start
    return elapsed


if __name__ == "__main__":
    runner = pyperf.Runner(processes=3, values=3, loops=1)
    df = daft.from_pydict({
        "url": [f"https://example.com/{i}" for i in range(N)]
    })
    runner.bench_time_func(
        "url_new", bench_url, df, False, inner_loops=N
    )
    runner.bench_time_func(
        "url_duplicates", bench_url, df, True, inner_loops=N
    )