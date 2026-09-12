#!/usr/bin/env python3
"""Build cellaris.eop, the package EOV installs.

    python package.py --onnxruntime path/to/onnxruntime.dll

A plain tar of plugin.toml, the built library and the ONNX runtime beside it.
"""

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

LIBRARY = {"win32": "cellaris.dll", "darwin": "libcellaris.dylib"}.get(
    sys.platform, "libcellaris.so"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--onnxruntime", type=Path, help="bundle this runtime too")
    parser.add_argument("--out", type=Path, default=Path("cellaris.eop"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent
    if not args.skip_build:
        subprocess.run(["cargo", "build", "--release"], cwd=root, check=True)

    built = root / "target" / "release" / LIBRARY
    if not built.exists():
        raise SystemExit(f"no {built}; build first or pass --skip-build wrongly")

    members = {"./plugin.toml": root / "plugin.toml", f"./{LIBRARY}": built}
    if args.onnxruntime:
        members[f"./{args.onnxruntime.name}"] = args.onnxruntime

    with tarfile.open(args.out, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, source in members.items():
            payload = source.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    print(f"{args.out}  {args.out.stat().st_size:,} bytes")
    for name in members:
        print(f"  {name}")
    print("\nInstall: copy it to ~/.eov/plugins/ and put markers.onnx in")
    print("~/.eov/plugins/cellaris/")


if __name__ == "__main__":
    main()
