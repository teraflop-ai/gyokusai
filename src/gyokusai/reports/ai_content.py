import daft
from daft import DataFrame, col

from gyokusai.filters.url_blacklist import extract_domain


def ai_domain_review_report(
    df: DataFrame,
    url_column: str = "url",
    score_column: str = "ai_content_score",
    page_score_threshold: float = 0.8,
    min_pages: int = 10,
    min_candidate_fraction: float = 0.5,
) -> DataFrame:
    """Return domain-level score candidates for human review."""
    df = df.with_column("domain", extract_domain(col(url_column))).where(col("domain") != "")
    above_threshold = (col(score_column) >= page_score_threshold).cast(daft.DataType.int64())
    df = df.groupby("domain").agg(
        col("domain").count().alias("page_count"),
        above_threshold.sum().alias("pages_above_threshold"),
        col(score_column).mean().alias("mean_score"),
    )
    df = df.with_column(
        "fraction_above_threshold",
        col("pages_above_threshold") / col("page_count"),
    )
    return df.where(
        (col("page_count") >= min_pages)
        & (col("fraction_above_threshold") >= min_candidate_fraction)
    ).select(
        "domain",
        "page_count",
        "pages_above_threshold",
        "fraction_above_threshold",
        "mean_score",
    )
