import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import py7zr

METADATA_URL_TMPL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL_TMPL = "https://archive.org/download/{identifier}/{path}"
CHUNK_SIZE = 1024 * 1024 * 1024  # Gigabyte


def get_file_list(identifier):
    url = METADATA_URL_TMPL.format(identifier=identifier)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        meta = json.loads(resp.read().decode("utf-8"))

    files = []
    for file in meta.get("files", []):
        name = file.get("name", "")
        if name.lower().endswith(".7z"):
            # TODO Implement MD5 check on files
            files.append(
                {"name": name, "size": int(file.get("size", 0)), "md5": file.get("md5")}
            )
    return files


def already_downloaded(dest, expected_size):
    if not os.path.exists(dest):
        return False
    actual_size = os.path.getsize(dest)
    if expected_size and actual_size != expected_size:
        return False
    return True


def download_file(identifier, fileinfo, out_dir, retries=3):
    filename = fileinfo["name"]
    file_url = DOWNLOAD_URL_TMPL.format(identifier=identifier, path=filename)
    local_name = os.path.basename(filename)
    dest = os.path.join(out_dir, local_name)
    tmp_dest = dest + ".part"

    if already_downloaded(dest, fileinfo["size"]):
        return (local_name, "skipped, already downloaded")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(file_url)
            with (
                urllib.request.urlopen(req, timeout=60) as response,
                open(tmp_dest, "wb") as out,
            ):
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(tmp_dest, dest)
            return (dest, "downloaded")
        except:
            if os.path.exists(tmp_dest):
                os.remove(tmp_dest)
            if attempt == retries:
                return (dest, "Failed")
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", default="stackexchange_20260331")
    parser.add_argument("--out", default="/p/data1/datasets/stackexchange")
    parser.add_argument(
        "--only", default=None, help="comma-delimited list of sites to restrict to"
    )
    args = parser.parse_args()

    files = get_file_list(args.identifier)
    if not files:
        print("No .7z files found. Check identifier")
        sys.exit(1)

    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        files = [
            f for f in files if os.path.basename(f["name"]).split(".")[0] in wanted
        ]

    total_bytes = sum(f["size"] for f in files)
    print(f"Found {len(files)} .7z files, total {total_bytes / 1e9:.2f} GB")
    print(f"Downloading to: {os.path.abspath(args.out)}\n")

    successes, failures = 0, 0
    for f in files:
        _, status = download_file(args.identifier, f, args.out)
        if status == "Failed":
            failures += 1
        else:
            successes += 1

    print(f"Done. Downloaded {successes}, Failed {failures}")


def unzip_7z(archive_path: str) -> Path:
    archive_path = Path(archive_path)
    out_dir = archive_path.with_suffix("")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=out_dir)
    return out_dir


if __name__ == "__main__":
    main()
