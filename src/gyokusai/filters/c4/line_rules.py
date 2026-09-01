"""Line-level C4 heuristics. Never directly called - only called through page_rules"""

_TERMINAL_PUNCTUATION = (".", "!", "?", '"', "”")


def _ends_with_terminal_punctuation(line: str) -> bool:
    """Line ends in a period, exclamation mark, question mark, or end quote."""
    return line.rstrip().endswith(_TERMINAL_PUNCTUATION)


def _has_min_words(line: str, min_words: int = 3) -> bool:
    return len(line.split()) >= min_words


def _mentions_javascript(line: str) -> bool:
    return "javascript" in line.lower()
