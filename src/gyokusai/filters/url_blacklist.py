from urllib.parse import urlparse

import daft
from daft import DataFrame, col
from datasets import load_dataset

from gyokusai.config import (
    DEFAULT_BLACKLIST_COLUMN_NAME,
    DEFAULT_BLACKLIST_REPO_ID,
    DEFAULT_BLACKLIST_SPLIT,
)


def _parse_netloc_for_host(netloc):
    # Parses user:pass@host or host:port into host
    return netloc.rsplit("@", 1)[-1].split(":", 1)[0].strip().lower().rstrip("/")


def _download_default_blacklist(
    repo_id: str = DEFAULT_BLACKLIST_REPO_ID,
    split: str = DEFAULT_BLACKLIST_SPLIT,
    column: str = DEFAULT_BLACKLIST_COLUMN_NAME,
) -> frozenset[str]:
    ds = load_dataset(repo_id, split=split)
    domains = set()
    for raw in ds[column]:
        if not raw:
            continue
        # entries may already be bare domains or full URLs
        netloc = urlparse(raw).netloc or raw
        host = _parse_netloc_for_host(netloc)
        domains.add(host)
    return frozenset(domains)


@daft.func
def extract_domain(url: str | None) -> str:
    """Extract the bare, lowercased hostname from a URL.

    Handles CommonCrawl-style URLs with ports, userinfo, and session ids
    in the path, e.g.:
    "http://01.vistaalegredoalto.sp.gov.br:5661/issweb/paginas/public/
     consulta/especie;jsessionid=mY8n3kSFVokUvnGHh3GEpWUm.undefined"
    -> "01.vistaalegredoalto.sp.gov.br"
    """
    netloc = urlparse(url or "").netloc
    return _parse_netloc_for_host(netloc)


@daft.func
def is_blacklisted_domain(domain: str, blacklist: frozenset) -> bool:
    """Check a domain against a blacklist, matching the domain itself
    or any of its parent domains (so 'ads.badsite.com' matches a
    blacklist entry of 'badsite.com')."""

    labels = domain.split(".")
    for i in range(len(labels)):
        candidate = ".".join(labels[i:])
        if candidate in blacklist:
            return True
    return False


class URLBlacklistFilter:
    def __init__(
        self,
        input_column: str = "url",
        blacklist: set[str] | None = None,
        blacklist_repo_id: str = DEFAULT_BLACKLIST_REPO_ID,
        blacklist_split: str = DEFAULT_BLACKLIST_SPLIT,
        blacklist_column: str = DEFAULT_BLACKLIST_COLUMN_NAME,
        name: str = "URLBlacklistFilter",
    ):
        self.input_column = input_column
        self.name = name

        if blacklist is not None:
            domains = blacklist
            self.blacklist = frozenset(d.lower().strip().rstrip("/") for d in domains)
        else:
            self.blacklist = _download_default_blacklist(
                repo_id=blacklist_repo_id,
                split=blacklist_split,
                column=blacklist_column,
            )

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.with_column("_domain", extract_domain(col(self.input_column)))
        df = df.where(~is_blacklisted_domain(col("_domain"), self.blacklist))
        df = df.exclude("_domain")
        return df
