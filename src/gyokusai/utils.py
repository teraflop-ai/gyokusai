import json
import os
from pathlib import Path
from typing import Literal

import daft
from daft import DataType, col, element
from daft.functions import length, list_filter, list_map, split, strip
from resiliparse.parse.encoding import bytes_to_str, detect_encoding


@daft.func
def decode_html(html: bytes) -> str:
    """
    OpenWebMath: https://arxiv.org/abs/2310.06786
    Decodes the html if possible.
    First try to decode with utf-8, then try to detect the encoding.
    """
    try:
        html = bytes_to_str(html, "utf-8")
    except Exception as e:
        encoding = detect_encoding(html)
        if encoding is None or encoding == "utf-8":
            return
        try:
            html = bytes_to_str(html, encoding)
        except Exception as e:
            return
    return html


def checkpoint_uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).resolve().as_uri()


def daft_dtype(
    precision: Literal["float32", "binary", "int8"] = "float32",
):
    if precision == "float32":
        daft_dtype = DataType.float32()
    elif precision == "binary":
        daft_dtype = DataType.binary()
    elif precision == "int8":
        daft_dtype = DataType.int8()
    else:
        raise ValueError(f"Daft datatype not supported: {precision}")
    return daft_dtype


def get_filepaths(directory, extension=".parquet"):
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(extension):
                file_paths.append(os.path.join(root, filename))
    return sorted(file_paths)


def load_personas(_DATASET_ID) -> daft.DataFrame:
    return daft.read_huggingface(_DATASET_ID)


def json_arg(v):
    if os.path.isfile(v):
        return json.load(open(v))
    return json.loads(v)


def sentences(column: str):
    lines = list_map(split(col(column), "\n"), strip(element()))
    return list_filter(lines, length(element()) > 0)


def paragraphs(column: str):
    paras = list_map(split(col(column), "\n\n"), strip(element()))
    return list_filter(paras, length(element()) > 0)
