#!/usr/bin/env python3
"""Downloads a pinned ONNX Runtime release for a platform, checks its SHA-256
and extracts it. Prints the extracted directory (lib/, LICENSE,
ThirdPartyNotices.txt), which `package.py --onnxruntime` and the tests'
`CELLARIS_SMOKE_ORT` accept.

    python3 tools/fetch_onnxruntime.py --platform linux-x86_64 --dest /tmp/ort
    python3 tools/fetch_onnxruntime.py --list

The digests are the ones GitHub publishes for the release assets of
microsoft/onnxruntime. Microsoft no longer builds ONNX Runtime for Intel
macOS; there, install it with Homebrew instead (`brew install onnxruntime`).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.30.0"
BASE = f"https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}"

# platform -> (asset, sha256)
PINS: dict[str, tuple[str, str]] = {
    "linux-x86_64": (
        f"onnxruntime-linux-x64-{VERSION}.tgz",
        "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd",
    ),
    "linux-arm64": (
        f"onnxruntime-linux-aarch64-{VERSION}.tgz",
        "e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da",
    ),
    "macos-arm64": (
        f"onnxruntime-osx-arm64-{VERSION}.tgz",
        "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012",
    ),
    "windows-x86_64": (
        f"onnxruntime-win-x64-{VERSION}.zip",
        "c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949",
    ),
    "windows-arm64": (
        f"onnxruntime-win-arm64-{VERSION}.zip",
        "e53db8a50b23ae35be901cc93428baf997dc8d420333b097b2eae53d3ea9f2d3",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(names: list[str]) -> None:
    for name in names:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts:
            raise SystemExit(f"refusing to extract {name!r}")


def extract(archive: Path, dest: Path, asset: str) -> Path:
    """Extracts `archive` (downloaded as `asset`) and returns its root."""
    dest.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as zip_file:
            safe_members(zip_file.namelist())
            zip_file.extractall(dest)
    else:
        with tarfile.open(archive) as tar:
            safe_members(tar.getnames())
            if hasattr(tarfile, "data_filter"):
                tar.extractall(dest, filter="data")
            else:
                tar.extractall(dest)
    stem = asset.removesuffix(".zip").removesuffix(".tgz")
    root = dest / stem
    if not (root / "lib").is_dir():
        raise SystemExit(f"{archive.name} did not hold {stem}/lib")
    for notice in ("LICENSE", "ThirdPartyNotices.txt"):
        if not (root / notice).is_file():
            raise SystemExit(f"{archive.name} did not hold {stem}/{notice}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--platform", choices=sorted(PINS))
    parser.add_argument("--dest", type=Path, default=Path("onnxruntime"))
    parser.add_argument(
        "--archive", type=Path, help="use this already-downloaded archive instead"
    )
    parser.add_argument("--list", action="store_true", help="print the pins")
    args = parser.parse_args()
    if args.list:
        for platform, (asset, digest) in sorted(PINS.items()):
            print(f"{platform}\t{BASE}/{asset}\t{digest}")
        return
    if not args.platform:
        parser.error("--platform is required")
    asset, expected = PINS[args.platform]
    archive = args.archive
    if archive is None:
        args.dest.mkdir(parents=True, exist_ok=True)
        archive = args.dest / asset
        if not archive.exists():
            request = urllib.request.Request(
                f"{BASE}/{asset}", headers={"User-Agent": "cellaris-ci"}
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                archive.write_bytes(response.read())
    actual = sha256(archive)
    if actual != expected:
        if args.archive is None:
            archive.unlink(missing_ok=True)
        raise SystemExit(f"{asset}: sha256 {actual}, expected {expected}")
    print(extract(archive, args.dest, asset))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
