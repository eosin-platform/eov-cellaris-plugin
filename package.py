#!/usr/bin/env python3
"""Build cellaris.eop, the package eov installs.

    python3 package.py
    python3 package.py --onnxruntime path/to/onnxruntime-linux-x64-1.22.0
    python3 package.py --onnxruntime path/to/libonnxruntime.so.1.22.0 \\
        --onnxruntime-license LICENSE --onnxruntime-notices ThirdPartyNotices.txt
    python3 package.py --model markers.onnx

The package is a tar of plugin.toml, the panel UI, the plugin library and
the plugin's licences; its SHA-256 goes into <out>.sha256 beside it. ONNX Runtime can be bundled (pass a release archive's
extracted directory, or the library and its licence files); its licence and
third-party notices are then always included, as its licence requires. A
model can be bundled too; its SHA-256 is printed so it can be published with
the package.

Run it from an eov checkout (the plugin builds against plugins/../plugin_api).
Nothing is installed: copy the package to eov's plugin directory, or run
`eov plugin install ./cellaris.eop`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLUGIN_ID = "cellaris"


def library_name(platform: str) -> str:
    if platform == "win32":
        return f"{PLUGIN_ID}.dll"
    if platform == "darwin":
        return f"lib{PLUGIN_ID}.dylib"
    return f"lib{PLUGIN_ID}.so"


def is_runtime(name: str, platform: str) -> bool:
    lower = name.lower()
    if "providers" in lower:
        return False
    if platform == "win32":
        return lower == "onnxruntime.dll"
    if platform == "darwin":
        return lower.startswith("libonnxruntime") and lower.endswith(".dylib")
    return lower == "libonnxruntime.so" or lower.startswith("libonnxruntime.so.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_runtime(given: Path, platform: str) -> Path:
    """The runtime library in a release directory (or the file given)."""
    if given.is_file():
        return given.resolve()
    for directory in (given, given / "lib"):
        if not directory.is_dir():
            continue
        names = sorted(
            (p for p in directory.iterdir() if is_runtime(p.name, platform)),
            # The real file, not its symlinks: the longest versioned name.
            key=lambda p: (not p.is_symlink(), len(p.name)),
            reverse=True,
        )
        if names:
            return names[0].resolve()
    raise SystemExit(f"no ONNX Runtime library for {platform} in {given}")


def find_notice(given: Path | None, runtime_arg: Path, names: list[str]) -> Path:
    if given is not None:
        if not given.is_file():
            raise SystemExit(f"{given} does not exist")
        return given
    start = runtime_arg if runtime_arg.is_dir() else runtime_arg.parent
    for directory in (start, start.parent):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise SystemExit(
        f"ONNX Runtime may only be redistributed with its {names[0]}: none found "
        f"near {runtime_arg}. Pass the release directory, or --onnxruntime-license "
        "and --onnxruntime-notices."
    )


def check_manifest(text: str) -> None:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        return
    manifest = tomllib.loads(text)
    for key in ("id", "name", "version", "environment"):
        if key not in manifest:
            raise SystemExit(f"plugin.toml has no {key}")
    if manifest["id"] != PLUGIN_ID:
        raise SystemExit(f"plugin.toml id is {manifest['id']!r}, not {PLUGIN_ID!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("cellaris.eop"))
    parser.add_argument("--target", help="a Rust target triple to build for")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--onnxruntime",
        type=Path,
        help="bundle ONNX Runtime: an extracted release directory or the library",
    )
    parser.add_argument("--onnxruntime-license", type=Path)
    parser.add_argument("--onnxruntime-notices", type=Path)
    parser.add_argument("--model", type=Path, help="bundle this markers.onnx")
    args = parser.parse_args()

    platform = sys.platform
    if args.target:
        if "windows" in args.target:
            platform = "win32"
        elif "apple" in args.target:
            platform = "darwin"
        else:
            platform = "linux"

    manifest = ROOT / "Cargo.toml"
    if not args.skip_build:
        command = ["cargo", "build", "--release", "--manifest-path", str(manifest)]
        if args.target:
            command += ["--target", args.target]
        subprocess.run(command, check=True)

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if args.target:
        target_dir = target_dir / args.target
    library = target_dir / "release" / library_name(platform)
    if not library.exists():
        raise SystemExit(f"no {library}: build first, or drop --skip-build")

    manifest_text = (ROOT / "plugin.toml").read_text(encoding="utf-8")
    check_manifest(manifest_text)

    members: dict[str, Path] = {
        "plugin.toml": ROOT / "plugin.toml",
        library_name(platform): library,
    }
    for ui in sorted((ROOT / "ui").glob("*.slint")):
        members[f"ui/{ui.name}"] = ui
    for licence in ("LICENSE-APACHE-2.0", "LICENSE-MIT", "LICENSE-GPLv3"):
        members[f"licenses/cellaris/{licence}"] = ROOT / licence

    notes = []
    if args.onnxruntime:
        runtime = find_runtime(args.onnxruntime, platform)
        members[runtime.name] = runtime
        members["licenses/onnxruntime/LICENSE"] = find_notice(
            args.onnxruntime_license, args.onnxruntime, ["LICENSE", "LICENSE.txt"]
        )
        members["licenses/onnxruntime/ThirdPartyNotices.txt"] = find_notice(
            args.onnxruntime_notices, args.onnxruntime, ["ThirdPartyNotices.txt"]
        )
        notes.append(f"ONNX Runtime {runtime.name}  sha256 {sha256(runtime)}")
    if args.model:
        if not args.model.is_file():
            raise SystemExit(f"{args.model} does not exist")
        members["markers.onnx"] = args.model
        notes.append(f"model {args.model.name}  sha256 {sha256(args.model)}")
        notes.append(
            "check the model's licence and its training data's terms before "
            "publishing this package"
        )

    with tarfile.open(args.out, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, source in members.items():
            payload = source.read_bytes()
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(payload)
            info.mode = 0o755 if name == library_name(platform) or "onnxruntime" in name else 0o644
            tar.addfile(info, io.BytesIO(payload))

    # Read it back: every member is there and the library is not empty.
    with tarfile.open(args.out) as tar:
        names = {m.name.removeprefix("./"): m for m in tar.getmembers()}
    missing = [name for name in members if name not in names]
    if missing or names[library_name(platform)].size == 0:
        raise SystemExit(f"the package is incomplete: {missing}")

    digest = sha256(args.out)
    sidecar = args.out.with_name(args.out.name + ".sha256")
    sidecar.write_text(f"{digest}  {args.out.name}\n", encoding="ascii")
    print(f"{args.out}  {args.out.stat().st_size:,} bytes  sha256 {digest}")
    for name in members:
        print(f"  {name}")
    for note in notes:
        print(f"  {note}")
    print(
        "\nInstall with `eov plugin install ./" + args.out.name + "`, or copy it to "
        "eov's plugin directory (~/.eov/plugins by default)."
    )
    if not args.model:
        print(
            "Put markers.onnx in the plugin's data directory, or choose it in the "
            "Cellaris panel."
        )


if __name__ == "__main__":
    main()
