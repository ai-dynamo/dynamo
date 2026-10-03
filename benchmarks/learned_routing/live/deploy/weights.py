# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage, verify and YaRN-patch the pinned Qwen3-32B checkpoint (stdlib only).

``copy SRC DST``   parallel copy of the manifest's files into a fresh node-local directory.
``verify DIR``     sizes and content hashes against the manifest (LFS sha256 or git blob sha1).
``patch-config DIR --rope-scaling JSON``
                   write the YaRN ``rope_scaling`` into DIR/config.json (the Qwen model card's
                   "modify the model files" route), keeping the verified original as
                   config.json.orig. Only the node-local copy is ever patched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 16 << 20


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "learned-routing.model-manifest.v1":
        raise SystemExit(f"error: {path} is not a model manifest")
    return manifest


def file_digest(path: Path, entry: dict) -> tuple[str, str]:
    if "sha256" in entry:
        h = hashlib.sha256()
        kind = "sha256"
    else:
        h = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
        kind = "git_blob_sha1"
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            h.update(chunk)
    return kind, h.hexdigest()


def verify(
    root: Path,
    manifest: dict,
    jobs: int,
    sizes_only: bool,
    skip: frozenset[str] = frozenset(),
) -> dict:
    problems = []
    entries = [e for e in manifest["files"] if e["path"] not in skip]
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
        elif path.stat().st_size != entry["size"]:
            problems.append(
                f"size {entry['path']} {path.stat().st_size} != {entry['size']}"
            )
    checked = 0
    if not problems and not sizes_only:

        def check(entry):
            kind, digest = file_digest(root / entry["path"], entry)
            return entry["path"], digest == entry[kind]

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for name, ok in pool.map(check, entries):
                checked += 1
                if not ok:
                    problems.append(f"hash {name}")
    return {
        "root": str(root),
        "files": len(entries),
        "hashed": checked,
        "sizes_only": sizes_only,
        "problems": problems,
        "ok": not problems,
    }


def copy(src: Path, dst: Path, manifest: dict, jobs: int) -> dict:
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"error: {dst} exists and is not empty")
    dst.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def one(entry):
        target = dst / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / entry["path"], target)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, manifest["files"]))
    elapsed = time.monotonic() - started
    total = sum(e["size"] for e in manifest["files"])
    return {
        "src": str(src),
        "dst": str(dst),
        "bytes": total,
        "seconds": round(elapsed, 1),
        "gb_per_s": round(total / max(elapsed, 1e-9) / 1e9, 2),
    }


def patch_config(root: Path, manifest: dict, rope_scaling: dict) -> dict:
    config_path = root / "config.json"
    original = root / "config.json.orig"
    entry = next(e for e in manifest["files"] if e["path"] == "config.json")
    source = original if original.exists() else config_path
    kind, digest = file_digest(source, entry)
    if digest != entry[kind]:
        raise SystemExit(f"error: {source} does not match the pinned config.json")
    if not original.exists():
        shutil.copyfile(config_path, original)
    config = json.loads(original.read_text())
    config["rope_scaling"] = rope_scaling
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return {
        "config": str(config_path),
        "rope_scaling": rope_scaling,
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "patched_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("copy", "verify", "patch-config"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--jobs", type=int, default=8)
        if name == "copy":
            cmd.add_argument("src", type=Path)
        cmd.add_argument("dir", type=Path)
        if name == "verify":
            cmd.add_argument("--sizes-only", action="store_true")
            cmd.add_argument(
                "--patched-config",
                action="store_true",
                help="config.json carries the YaRN patch; verify config.json.orig",
            )
        if name == "patch-config":
            cmd.add_argument("--rope-scaling", required=True, help="JSON object")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "copy":
        result = copy(args.src, args.dir, manifest, args.jobs)
        ok = True
    elif args.command == "verify":
        skip = frozenset({"config.json"}) if args.patched_config else frozenset()
        result = verify(args.dir, manifest, args.jobs, args.sizes_only, skip)
        if args.patched_config and result["ok"]:
            entry = next(e for e in manifest["files"] if e["path"] == "config.json")
            kind, digest = file_digest(args.dir / "config.json.orig", entry)
            result["config_orig_ok"] = digest == entry[kind]
            result["ok"] = result["config_orig_ok"]
        ok = result["ok"]
    else:
        result = patch_config(args.dir, manifest, json.loads(args.rope_scaling))
        ok = True
    result["command"] = args.command
    print(json.dumps(result, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
