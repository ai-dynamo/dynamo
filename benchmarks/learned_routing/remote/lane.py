# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bookkeeping for the Slurm CPU lane (stdlib only; run with the WT venv python).

Subcommands::

    lane.py alloc-record --job ID [--set key=value ...] [--json '{...}']
        add or update one allocation in CR/facts/remote.json (keyed by job ID); the exact cancel
        command is filled in automatically
    lane.py alloc-refresh [--job ID ...]
        read state, node, start and end of recorded jobs back from sacct on the cluster's login node
    lane.py set KEY JSON
        set one top-level key of CR/facts/remote.json
    lane.py trace-list BUNDLE
        write BUNDLE/traces.sha256 (sha256sum -c format) from the cells' declared trace SHA-256s,
        after checking each declared value against the local file and its provenance in
        CR/traces/MANIFEST.json
    lane.py train-bundle BUNDLE --train T.jsonl --val V.jsonl --space S.yaml ... --jobs J.jsonl
        turn an lr-eval bundle into an lr-train bundle: split cells, add numpy + cma (site-train/),
        spaces and the job list
    lane.py latest-bundle BUILD_ID
        print the newest remote bundle recorded for this bindings build (rsync --link-dest base)
    lane.py note-bundle NAME BUILD_ID REMOTE_PATH
        remember a pushed bundle for later --link-dest reuse

Every write to remote.json holds an exclusive flock on remote.json.lock and replaces the file
atomically, so concurrent submit scripts cannot lose each other's entries.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import siteenv

siteenv.load()
CR = Path(siteenv.require("LR_CR"))
FACTS = CR / "facts" / "remote.json"
SSH_ALIAS = siteenv.require("LR_SSH_ALIAS")
SCHEMA = "learned-routing.remote.v1"
TERMINAL = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


@contextmanager
def facts_lock():
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(FACTS) + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = json.loads(FACTS.read_text()) if FACTS.exists() else {"schema": SCHEMA}
        yield data
        data["updated"] = now()
        tmp = FACTS.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, FACTS)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def cancel_command(job: str) -> str:
    return f"ssh {SSH_ALIAS} scancel {job}"


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cmd_alloc_record(args) -> int:
    fields = json.loads(args.json) if args.json else {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        fields[key] = _parse_value(value)
    job = str(args.job)
    with facts_lock() as data:
        allocations = data.setdefault("allocations", [])
        entry = next((a for a in allocations if str(a.get("job_id")) == job), None)
        if entry is None:
            entry = {"job_id": job, "recorded": now()}
            allocations.append(entry)
        entry.update(fields)
        entry["cancel"] = cancel_command(job)
    print(cancel_command(job))
    return 0


def _sacct(jobs: list[str]) -> dict[str, dict]:
    if not jobs:
        return {}
    remote = (
        "sacct -X -n -P -j "
        + ",".join(jobs)
        + " -o JobID,State,NodeList,Submit,Start,End,Elapsed,AllocCPUS,ReqMem,Partition,Account,QOS,Timelimit"
    )
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_ALIAS, remote],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    keys = [
        "job_id",
        "state",
        "node",
        "submit",
        "start",
        "end",
        "elapsed",
        "alloc_cpus",
        "req_mem",
        "partition",
        "account",
        "qos",
        "time_limit",
    ]
    rows = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == len(keys):
            row = dict(zip(keys, parts))
            row["state"] = row["state"].split()[0]
            rows[row["job_id"]] = row
    return rows


def cmd_alloc_refresh(args) -> int:
    with facts_lock() as data:
        allocations = data.setdefault("allocations", [])
        wanted = (
            [str(j) for j in args.job]
            if args.job
            else [
                str(a["job_id"])
                for a in allocations
                if str(a.get("state", "")).split()[0] not in TERMINAL
            ]
        )
        rows = _sacct(wanted)
        for entry in allocations:
            row = rows.get(str(entry["job_id"]))
            if row:
                entry.update({k: v for k, v in row.items() if k != "job_id"})
                entry["refreshed"] = now()
        live = [
            a["job_id"]
            for a in allocations
            if str(a.get("state", "")).split()[0] not in TERMINAL
        ]
    print(json.dumps({"refreshed": sorted(rows), "live": live}))
    return 0


def cmd_set(args) -> int:
    value = (
        json.loads(Path(args.value[1:]).read_text())
        if args.value.startswith("@")
        else json.loads(args.value)
    )
    with facts_lock() as data:
        data[args.key] = value
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_shas() -> set[str]:
    manifest = json.loads((CR / "traces" / "MANIFEST.json").read_text())
    return {f["sha256"] for f in manifest["files"] if f.get("sha256")}


def _provenance(local_source: Path, manifest_shas: set[str], sha: str) -> str:
    """How a staged trace traces back to CR/traces/MANIFEST.json."""
    if sha in manifest_shas:
        return "manifest"
    for meta in (
        local_source.with_suffix(".meta.json"),
        local_source.with_name(local_source.name + ".meta.json"),
    ):
        if not meta.exists():
            continue
        info = json.loads(meta.read_text())
        if info.get("sha256") not in (None, sha):
            return f"meta sha mismatch ({meta.name})"
        source = info.get("source") or {}
        if source.get("sha256") in manifest_shas:
            return "derived from manifest source " + source["sha256"][:16]
        base = info.get("base_manifest")
        if (
            base
            and Path(base).exists()
            and _sha256(Path(base)) == info.get("base_manifest_sha256")
        ):
            return "lowered from " + str(Path(base).relative_to(CR))
        return f"meta without a manifest-backed source ({meta.name})"
    if local_source.name.startswith(sha):
        return "content-addressed name only"
    return "unknown"


def cmd_trace_list(args) -> int:
    bundle = Path(args.bundle)
    manifest_shas = _manifest_shas()
    lines, report = [], []
    seen = set()
    for line in (bundle / "cells.jsonl").read_text().splitlines():
        cell = json.loads(line)
        for rel in cell.get("trace_files") or []:
            if rel in seen:
                continue
            seen.add(rel)
            staged = bundle / rel
            actual = _sha256(staged)
            declared = cell.get("trace_sha256") or []
            declared = declared if isinstance(declared, list) else [declared]
            if declared and declared != [actual]:
                raise SystemExit(f"{rel}: content {actual[:16]} != declared {declared}")
            # The bundle names each trace traces/<sha16>/<basename>; find the local original.
            local = next(CR.joinpath("traces").rglob(staged.name), None)
            provenance = (
                _provenance(local, manifest_shas, actual)
                if local
                else "local original not found"
            )
            lines.append(f"{actual}  {rel}\n")
            report.append(
                {
                    "path": rel,
                    "sha256": actual,
                    "bytes": staged.stat().st_size,
                    "provenance": provenance,
                }
            )
    (bundle / "traces.sha256").write_text("".join(lines))
    (bundle / "traces.provenance.json").write_text(json.dumps(report, indent=1) + "\n")
    bad = [
        r
        for r in report
        if r["provenance"].startswith(("meta sha", "unknown", "local original"))
    ]
    print(
        json.dumps(
            {
                "traces": len(report),
                "bytes": sum(r["bytes"] for r in report),
                "bad_provenance": bad,
            }
        )
    )
    return 1 if bad else 0


def _copy_package(name: str, dest: Path) -> list[str]:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"cannot locate package {name}")
    root = Path(list(spec.submodule_search_locations)[0])
    site_packages = root.parent
    copied = []
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "tests")
    shutil.copytree(root, dest / root.name, ignore=ignore, dirs_exist_ok=True)
    copied.append(root.name)
    for extra in site_packages.glob(f"{name}.libs"):
        shutil.copytree(extra, dest / extra.name, dirs_exist_ok=True)
        copied.append(extra.name)
    for info in site_packages.glob(f"{name}-*.dist-info"):
        shutil.copytree(info, dest / info.name, dirs_exist_ok=True)
        copied.append(info.name)
    return copied


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_train_bundle(args) -> int:
    bundle = Path(args.bundle)
    by_id = {c["cell_id"]: c for c in _jsonl(bundle / "cells.jsonl")}

    def ids(path):
        return [cell["cell_id"] for cell in _jsonl(Path(path))]

    for name, sources in (("train.jsonl", args.train), ("val.jsonl", args.val or [])):
        wanted = [cid for path in sources for cid in ids(path)]
        missing = [cid for cid in wanted if cid not in by_id]
        if missing:
            raise SystemExit(f"{name}: cells missing from bundle: {missing[:5]}")
        (bundle / name).write_text(
            "".join(json.dumps(by_id[c], sort_keys=True) + "\n" for c in wanted)
        )
    site_train = bundle / "site-train"
    site_train.mkdir(exist_ok=True)
    copied = _copy_package("numpy", site_train) + _copy_package("cma", site_train)
    spaces = bundle / "spaces"
    spaces.mkdir(exist_ok=True)
    for space in args.space:
        shutil.copy2(space, spaces / Path(space).name)
    jobs = _jsonl(Path(args.jobs))
    for job in jobs:
        if not (spaces / Path(job["space"]).name).exists():
            raise SystemExit(
                f"job {job['run']}: space {job['space']} was not passed with --space"
            )
        job["space"] = "spaces/" + Path(job["space"]).name
    (bundle / "train_jobs.jsonl").write_text(
        "".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs)
    )
    print(
        json.dumps(
            {
                "train_cells": len(ids(args.train[0]))
                if len(args.train) == 1
                else None,
                "site_train": copied,
                "jobs": [j["run"] for j in jobs],
            }
        )
    )
    return 0


def _bundle_index() -> Path:
    return CR / "runs" / "remote" / "bundles.json"


def cmd_note_bundle(args) -> int:
    index = _bundle_index()
    index.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(index.read_text()) if index.exists() else []
    data.append(
        {
            "name": args.name,
            "build_id": args.build_id,
            "remote": args.remote,
            "pushed": now(),
        }
    )
    index.write_text(json.dumps(data, indent=1) + "\n")
    return 0


def cmd_latest_bundle(args) -> int:
    index = _bundle_index()
    data = json.loads(index.read_text()) if index.exists() else []
    match = [b for b in data if b["build_id"] == args.build_id]
    if match:
        print(match[-1]["remote"])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lane.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("alloc-record")
    a.add_argument("--job", required=True)
    a.add_argument("--set", action="append")
    a.add_argument("--json")
    a.set_defaults(fn=cmd_alloc_record)
    r = sub.add_parser("alloc-refresh")
    r.add_argument("--job", action="append")
    r.set_defaults(fn=cmd_alloc_refresh)
    s = sub.add_parser("set")
    s.add_argument("key")
    s.add_argument("value", help="JSON text, or @file.json")
    s.set_defaults(fn=cmd_set)
    t = sub.add_parser("trace-list")
    t.add_argument("bundle")
    t.set_defaults(fn=cmd_trace_list)
    b = sub.add_parser("train-bundle")
    b.add_argument("bundle")
    b.add_argument("--train", nargs="+", required=True)
    b.add_argument("--val", nargs="*")
    b.add_argument("--space", nargs="+", required=True)
    b.add_argument("--jobs", required=True)
    b.set_defaults(fn=cmd_train_bundle)
    n = sub.add_parser("note-bundle")
    n.add_argument("name")
    n.add_argument("build_id")
    n.add_argument("remote")
    n.set_defaults(fn=cmd_note_bundle)
    lb = sub.add_parser("latest-bundle")
    lb.add_argument("build_id")
    lb.set_defaults(fn=cmd_latest_bundle)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
