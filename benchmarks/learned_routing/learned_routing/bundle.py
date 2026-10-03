# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained ``lr-eval`` bundles for Slurm CPU nodes, and ingestion of their results.

Layout of ``lr-eval --bundle-out DIR``::

    DIR/MANIFEST.json      what is inside, build ID, glibc floor, task count, file sizes
    DIR/run.sh             runs the shard: bash DIR/run.sh [--shard i/n] [lr-eval options]
    DIR/site/              frozen import tree: dynamo (bindings + components), learned_routing,
                           aisimulate, aisimulate_core, yaml, packaging, typing_extensions
                           (+ dist-info)
    DIR/wheels/            learned_routing sdist and wheel (uv build of DIR/src/)
    DIR/config/engine.json
    DIR/traces/<sha16>/    the cells' trace sources
    DIR/cells.jsonl        the cells, trace paths rewritten relative to DIR
    DIR/specs.jsonl        the policy specs
    DIR/runs/cache/e0/     the local E0 table (saves recomputation)

``site/dynamo/_core.abi3.so`` is the exact local extension, not a rebuild, so the remote bindings
build ID equals the local one and remote cache entries are valid local cache entries. ``run.sh``
runs ``python -S`` with ``PYTHONPATH=DIR/site``: no venv or pip on the node, only CPython 3.12
(``uv`` can provide one) and glibc at least the floor in the manifest (otherwise use a container
with a newer glibc).

Before writing the manifest, :func:`check_site` runs ``python -S`` with only ``DIR/site`` on
``sys.path`` and a minimal environment, imports what a worker imports and builds the bundled
engine's ``MockEngineArgs`` and a ``KvRouterConfig``; a missing module fails the build instead of
every remote replay (build audit B1: ``packaging`` was missing and the old check passed only through
the host's dist-packages).

``lr-eval --ingest DIR`` copies each remote result into the local cache after recomputing its
cache key from its own fields and checking the harness version and bindings build ID.
"""

from __future__ import annotations

import fcntl
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from learned_routing import HARNESS_VERSION
from learned_routing.cache import ResultCache, bindings_build_id, cache_key
from learned_routing.canon import atomic_write_text, sha256_json
from learned_routing.cells import Cell, engine_content, source_sha256
from learned_routing.paths import Layout

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "*.egg-info")
SITE_PACKAGES = (
    "learned_routing",
    "aisimulate",
    "aisimulate_core",
    "yaml",
    "packaging",
)
DIST_INFOS = ("aisimulate", "pyyaml", "packaging", "typing_extensions")
# What a worker imports and builds; run by check_site under python -S with only site/ on sys.path.
SITE_CHECK = """
import json, sys
engine = json.load(open(sys.argv[1]))
import learned_routing.worker, learned_routing.eval_cli, learned_routing.goodput, learned_routing.e0
from dynamo.replay import run_trace_replay, run_synthetic_trace_replay
from dynamo.mocker import MockEngineArgs
from dynamo.llm import KvRouterConfig
MockEngineArgs.from_json(json.dumps(engine["mock_engine_args"]))
KvRouterConfig()
roots = tuple(path.rstrip("/") + "/" for path in sys.argv[2:4])
leaks = sorted(
    name for name, module in sys.modules.items()
    if getattr(module, "__file__", None) and not module.__file__.startswith(roots)
)
print(json.dumps({"modules": len(sys.modules), "outside_site_and_stdlib": leaks}))
sys.exit(1 if leaks else 0)
"""

RUN_SH = """#!/usr/bin/env bash
# Run this bundle's lr-eval shard. Extra arguments go to lr-eval (e.g. --shard 0/4,
# --max-wall-seconds 3000, --slots 32). Results land in runs/cache and results.jsonl here;
# bring them home with: lr-eval --ingest <this directory>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${{PYTHON:-}}"
if [ -z "$PY" ]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PY="$(command -v python3.12)"
  elif command -v uv >/dev/null 2>&1; then
    PY="$(uv python find 3.12 2>/dev/null || (uv python install 3.12 >&2 && uv python find 3.12))"
  else
    echo "run.sh: need python3.12 or uv on PATH (or set PYTHON)" >&2
    exit 1
  fi
fi
SLOTS="${{LR_SLOTS:-$(nproc)}}"
export PYTHONPATH="$HERE/site" PYTHONDONTWRITEBYTECODE=1 DYN_LOG="${{DYN_LOG:-warn}}" LR_ROOT="$HERE"
exec "$PY" -S -m learned_routing.eval_cli --root "$HERE" \\
  --policy-spec "$HERE/specs.jsonl" --cells "$HERE/cells.jsonl" \\
  --repeats {repeats} --repeat-offset {offset} \\
  --out "$HERE/results.jsonl" --slots "$SLOTS" --num-slots "$SLOTS" "$@"
"""


class BundleError(RuntimeError):
    pass


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise BundleError(f"cannot locate package {name!r}")
    if spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    return Path(spec.origin)


def _dist_info(name: str) -> Path | None:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    path = getattr(dist, "_path", None)
    return Path(path) if path else None


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True, symlinks=False)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _glibc_floor(so: Path) -> str | None:
    try:
        out = subprocess.run(
            ["objdump", "-T", str(so)], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    versions = re.findall(r"GLIBC_(\d+(?:\.\d+)*)", out)
    if not versions:
        return None
    return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def build_site(site: Path) -> dict:
    """Frozen import tree of everything lr-eval needs at run time."""
    import dynamo.replay  # noqa: F401  (resolves the components tree)

    core = Path(importlib.util.find_spec("dynamo._core").origin)
    components = Path(importlib.util.find_spec("dynamo.replay").origin).parents[1]
    _copy(core.parent, site / "dynamo")
    _copy(components, site / "dynamo")
    for name in SITE_PACKAGES:
        _copy(_package_dir(name), site / name)
    _copy(_package_dir("typing_extensions"), site / "typing_extensions.py")
    for name in DIST_INFOS:
        info = _dist_info(name)
        if info is not None:
            _copy(info, site / info.name)
    return {"core_so": str(core), "components": str(components)}


def check_site(site: Path, engine_json: Path, python: str | None = None) -> dict:
    """Import and construct what a worker needs from ``site`` alone; raise BundleError on failure.

    Runs ``python -S -s -P`` from ``site`` with ``PYTHONPATH=site`` and no other environment, so
    neither the venv, the interpreter's site-packages nor the working directory can satisfy an
    import, and requires every loaded module to come from ``site`` or the standard library.
    """
    import sysconfig

    site, engine_json = Path(site).resolve(), Path(engine_json).resolve()
    python = python or sys.executable
    stdlib = sysconfig.get_paths()["stdlib"]
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(site),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DYN_LOG": "warn",
    }
    result = subprocess.run(
        [
            python,
            "-S",
            "-s",
            "-P",
            "-c",
            SITE_CHECK,
            str(engine_json),
            str(site),
            stdlib,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(site),
        timeout=300,
    )
    out = result.stdout.strip().splitlines()
    report = {"python": python, "returncode": result.returncode}
    if out:
        try:
            report.update(json.loads(out[-1]))
        except json.JSONDecodeError:
            pass
    if result.returncode != 0:
        raise BundleError(
            f"bundle site check failed under {python} -S: "
            f"{(result.stderr.strip().splitlines() or ['?'])[-1]} {out[-1:] or ''}"
        )
    return report


def _build_wheels(wheels: Path, scratch: Path) -> dict:
    """sdist + wheel of learned_routing, built from a copy so the source tree stays clean."""
    package = Path(
        importlib.util.find_spec("learned_routing").submodule_search_locations[0]
    ).parent
    uv = shutil.which("uv")
    if uv is None:
        return {"built": False, "reason": "uv not on PATH"}
    src = scratch / "learned_routing_src"
    _copy(package / "pyproject.toml", src / "pyproject.toml")
    _copy(package / "learned_routing", src / "learned_routing")
    result = subprocess.run(
        [uv, "build", "--sdist", "--wheel", "--out-dir", str(wheels), str(src)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"built": False, "reason": result.stderr[-2000:]}
    return {
        "built": True,
        "files": sorted(
            p.name for p in wheels.iterdir() if p.suffix in (".whl", ".gz")
        ),
    }


def build_bundle(
    out_dir: Path,
    layout: Layout,
    *,
    specs,
    cells: list[Cell],
    repeats: int,
    repeat_offset: int,
) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise BundleError(f"{out_dir} exists and is not empty; choose a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    build = bindings_build_id(layout.cache_dir)
    site_info = build_site(out_dir / "site")
    wheels = _build_wheels(out_dir / "wheels", out_dir / "src")
    bundle_layout = Layout(out_dir.resolve())
    engines: dict[str, str] = {}
    rewritten = []
    sha_check = {}
    for cell in cells:
        raw = dict(cell.raw)
        engine = cell.engine()
        engine_sha = sha256_json(engine_content(engine))
        if engine_sha not in engines:
            name = "engine.json" if not engines else f"engine-{engine_sha[:12]}.json"
            _copy(cell.engine_path(), out_dir / "config" / name)
            engines[engine_sha] = f"config/{name}"
        raw["engine_ref"] = engines[engine_sha]
        if not cell.is_synthetic:
            source = cell.trace_source()
            sha = source_sha256(source, cell.trace_format)
            relative = f"traces/{sha[:16]}/{source.name}"
            if not (out_dir / relative).exists():
                _copy(source, out_dir / relative)
            raw["trace_files"] = [relative]
        rewritten.append(raw)
        local_sha = cell.content_sha()
        remote_sha = Cell(raw=raw, layout=bundle_layout).content_sha()
        sha_check[cell.cell_id] = local_sha == remote_sha
    if not all(sha_check.values()):
        bad = [cid for cid, ok in sha_check.items() if not ok]
        raise BundleError(f"bundled cells changed content identity: {bad[:5]}")
    (out_dir / "cells.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rewritten)
    )
    (out_dir / "specs.jsonl").write_text(
        "".join(json.dumps(s.to_dict(), sort_keys=True) + "\n" for s in specs)
    )
    if layout.e0_dir.exists():
        for path in layout.e0_dir.glob("*.json"):
            _copy(path, out_dir / "runs" / "cache" / "e0" / path.name)
    site_check = check_site(out_dir / "site", out_dir / "config" / "engine.json")
    run_sh = out_dir / "run.sh"
    run_sh.write_text(RUN_SH.format(repeats=repeats, offset=repeat_offset))
    run_sh.chmod(0o755)
    core_copy = out_dir / "site" / "dynamo" / Path(site_info["core_so"]).name
    manifest = {
        "bundle": str(out_dir),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "harness_version": HARNESS_VERSION,
        "build_id": build["build_id"],
        "core_so_sha256": build["core_so_sha256"],
        "aisimulate": build["aisimulate"],
        "glibc_floor": _glibc_floor(core_copy),
        "python": f"CPython {platform.python_version()} (abi3 bindings; vendored yaml C speedups are cp312 and optional)",
        "host": platform.node(),
        "cells": len(cells),
        "specs": len(specs),
        "repeats": repeats,
        "repeat_offset": repeat_offset,
        "tasks": len(cells) * len(specs) * repeats,
        "cell_content_sha_match": True,
        "wheels": wheels,
        "site": site_info,
        "site_check": site_check,
        "bytes": _dir_bytes(out_dir),
        "usage": "bash run.sh [--shard i/n] [--max-wall-seconds S]; then locally: lr-eval --ingest <bundle>",
    }
    atomic_write_text(
        out_dir / "MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True)
    )
    return manifest


def _merge_e0(bundle: Path, layout: Layout) -> int:
    merged = 0
    for path in (bundle / "runs" / "cache" / "e0").glob("*.json"):
        target = layout.e0_dir / path.name
        layout.e0_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(target.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            local = json.loads(target.read_text()) if target.exists() else {}
            remote = json.loads(path.read_text())
            new = {k: v for k, v in remote.items() if k not in local}
            if new:
                local.update(new)
                atomic_write_text(target, json.dumps(local, sort_keys=True))
                merged += len(new)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    return merged


def ingest(
    bundle: Path, layout: Layout, *, out: str | None = None, any_build: bool = False
) -> dict:
    bundle = Path(bundle)
    remote = ResultCache(bundle / "runs" / "cache")
    local = ResultCache(layout.cache_dir)
    local_build = bindings_build_id(layout.cache_dir)["build_id"]
    ingested = already = 0
    rejected: list[dict] = []
    for key in sorted(remote.keys()):
        if key.endswith(".per_request.jsonl"):
            continue
        record = json.loads(remote.path(key).read_text())
        reason = None
        recomputed = cache_key(
            policy_sha=record.get("policy_sha", ""),
            cell_id=record.get("cell_id", ""),
            cell_sha=record.get("cell_sha", ""),
            repeat=record.get("repeat", -1),
            protocol=record.get("replicate_protocol", ""),
            harness_version=record.get("harness_version", ""),
            build_id=record.get("build_id", ""),
        )
        if record.get("error"):
            reason = "errored"
        elif record.get("cache_key") != key or recomputed != key:
            reason = "cache_key does not match the record's own fields"
        elif record.get("harness_version") != HARNESS_VERSION:
            reason = (
                f"harness_version {record.get('harness_version')} != {HARNESS_VERSION}"
            )
        elif record.get("build_id") != local_build and not any_build:
            reason = f"build_id {str(record.get('build_id'))[:12]} != local {local_build[:12]}"
        if reason:
            rejected.append({"key": key, "reason": reason})
            continue
        if local.path(key).exists():
            already += 1
            continue
        per_request = remote.per_request_path(key)
        if per_request.exists():
            _copy(per_request, local.per_request_path(key))
            record["per_request_path"] = str(local.per_request_path(key))
        record["ingested_from"] = str(bundle.resolve())
        local.put(key, record)
        ingested += 1
    appended = 0
    results = bundle / "results.jsonl"
    if out and results.exists():
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as handle:
            for line in results.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    row["ingested_from"] = str(bundle.resolve())
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    appended += 1
    return {
        "bundle": str(bundle),
        "ingested": ingested,
        "already_cached": already,
        "rejected": rejected,
        "results_appended": appended,
        "e0_values_merged": _merge_e0(bundle, layout),
        "python": sys.version.split()[0],
    }
