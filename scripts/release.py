#!/usr/bin/env python3
"""Release tooling for Cellaris, in the schema eov's installer reads.

    python3 scripts/release.py validate-source --version 1.0.0
    python3 scripts/release.py manifest --version 1.0.0 \\
        --artifacts-dir artifacts --staging-dir release-files \\
        --output release-files/release.toml
    python3 scripts/release.py verify-assets --version 1.0.0 \\
        --manifest release-files/release.toml

`manifest` stages the six platform packages (cellaris-v<version>-<platform>
.eop, each with its .sha256 sidecar) and writes release.toml: schema 1, a
[plugin] table, and one [platform.<os>.<arch>] table per package with its
version, SHA-256, immutable download URL and eov requirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "eosin-platform/eov-cellaris-plugin"
PLATFORMS = (
    ("platform.linux.x86_64", "linux-x86_64"),
    ("platform.linux.arm64", "linux-arm64"),
    ("platform.macos.arm64", "macos-arm64"),
    ("platform.macos.x86_64", "macos-x86_64"),
    ("platform.windows.x86_64", "windows-x86_64"),
    ("platform.windows.arm64", "windows-arm64"),
)
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")


class ReleaseError(Exception):
    pass


def artifact(version: str, platform: str) -> str:
    return f"cellaris-v{version}-{platform}.eop"


def url(version: str, name: str) -> str:
    return f"https://github.com/{REPOSITORY}/releases/download/v{version}/{name}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"cannot read {path}: {error}") from error


def validate_source(version: str) -> dict:
    """Checks that Cargo.toml, plugin.toml and the version agree; returns
    plugin.toml."""
    if not SEMVER.fullmatch(version):
        raise ReleaseError(f"not a semantic version: {version!r}")
    cargo = read_toml(ROOT / "Cargo.toml")["package"]
    plugin = read_toml(ROOT / "plugin.toml")
    if cargo.get("version") != version or plugin.get("version") != version:
        raise ReleaseError(
            f"versions disagree with {version}: Cargo.toml {cargo.get('version')!r}, "
            f"plugin.toml {plugin.get('version')!r}"
        )
    if plugin.get("id") != "cellaris":
        raise ReleaseError("plugin.toml id is not cellaris")
    if plugin.get("repository") != f"https://github.com/{REPOSITORY}":
        raise ReleaseError("plugin.toml repository is not the release repository")
    environment = plugin.get("environment", {}).get("version")
    if not isinstance(environment, str):
        raise ReleaseError("plugin.toml has no [environment].version")
    if "research use only" not in plugin.get("description", "").lower():
        raise ReleaseError("plugin.toml's description must say research use only")
    return plugin


def command_validate(args: argparse.Namespace) -> None:
    validate_source(args.version)
    print(f"source agrees on {args.version}")


def command_manifest(args: argparse.Namespace) -> None:
    plugin = validate_source(args.version)
    environment = plugin["environment"]["version"]
    found = {p.name: p for p in args.artifacts_dir.rglob("*") if p.is_file()}
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "[manifest]",
        "schema = 1",
        'kind = "plugin"',
        f"version = {json.dumps(args.version)}",
        f'repository = "https://github.com/{REPOSITORY}"',
        "",
        "[plugin]",
        f"id = {json.dumps(plugin['id'])}",
        f"name = {json.dumps(plugin['name'])}",
        f"version = {json.dumps(args.version)}",
        f'repository = "https://github.com/{REPOSITORY}"',
        f"description = {json.dumps(plugin['description'])}",
        f"environment = {json.dumps(environment)}",
        "",
    ]
    for section, platform in PLATFORMS:
        name = artifact(args.version, platform)
        if name not in found:
            raise ReleaseError(f"missing {name}")
        digest = sha256(found[name])
        sidecar = found.get(name + ".sha256")
        if sidecar is None:
            raise ReleaseError(f"missing {name}.sha256")
        recorded = sidecar.read_text(encoding="ascii").split()
        if recorded[:2] != [digest, name]:
            raise ReleaseError(f"{sidecar.name} does not match {name}")
        shutil.copy2(found[name], args.staging_dir / name)
        shutil.copy2(sidecar, args.staging_dir / sidecar.name)
        lines += [
            f"[{section}]",
            f"version = {json.dumps(args.version)}",
            f"sha256 = {json.dumps(digest)}",
            f"url = {json.dumps(url(args.version, name))}",
            f"environment = {json.dumps(environment)}",
            "",
        ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    check_manifest(read_toml(args.output), args.version, environment)
    print(f"wrote {args.output}")


def check_manifest(data: dict, version: str, environment: str) -> None:
    meta = data.get("manifest", {})
    if meta.get("schema") != 1 or meta.get("kind") != "plugin":
        raise ReleaseError("release.toml is not a schema-1 plugin manifest")
    if data.get("plugin", {}).get("version") != version:
        raise ReleaseError("[plugin] has the wrong version")
    for section, platform in PLATFORMS:
        entry = data
        for part in section.split("."):
            entry = entry.get(part, {})
        if entry.get("version") != version or entry.get("environment") != environment:
            raise ReleaseError(f"[{section}] is inconsistent")
        if entry.get("url") != url(version, artifact(version, platform)):
            raise ReleaseError(f"[{section}] does not use the immutable URL")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ReleaseError(f"[{section}] has no valid SHA-256")


def command_verify(args: argparse.Namespace) -> None:
    plugin = validate_source(args.version)
    data = read_toml(args.manifest)
    check_manifest(data, args.version, plugin["environment"]["version"])
    for section, _ in PLATFORMS:
        entry = data
        for part in section.split("."):
            entry = entry[part]
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "cellaris-release"})
        with urllib.request.urlopen(request, timeout=120) as response:
            digest = hashlib.sha256(response.read()).hexdigest()
        if digest != entry["sha256"]:
            raise ReleaseError(f"{entry['url']} does not match its SHA-256")
    print(f"verified {len(PLATFORMS)} published packages")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-source")
    validate.add_argument("--version", required=True)
    validate.set_defaults(run=command_validate)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--artifacts-dir", type=Path, required=True)
    manifest.add_argument("--staging-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(run=command_manifest)
    verify = commands.add_parser("verify-assets")
    verify.add_argument("--version", required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.set_defaults(run=command_verify)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
