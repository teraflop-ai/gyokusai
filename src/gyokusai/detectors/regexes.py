import re2 as re

CODE_PATTERN = re.compile(
    r"\b(?:if|else|for|while|def|class|include|switch|case|default|const|static|"
    r"try|catch|exception|continue|open|close|import|var|None|null|true|True|"
    r"false|False|print|return|sudo|apt-get|wget|\+|-|\*|/|=)\b"
    r"|[{};]|\w+\s*\(.*\)|\w+\s*=\s*\w+"
)
