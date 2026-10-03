# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record live-lane jobs in CR/facts/live.json (flock + atomic replace; other keys preserved).

    live_facts.py add-job --job-id 123 --field partition=PARTITION --field cancel="ssh ALIAS scancel 123"
    live_facts.py set-job --job-id 123 --field state=COMPLETED --field ended_utc=...

``jobs`` is a list of objects keyed by ``job_id``; ``set-job`` merges fields into an existing
entry. Values that parse as JSON are stored as JSON, everything else as strings. The default
``--path`` is ``$LR_CR/facts/live.json`` (``LR_CR`` from deploy/site.env via common.sh, or the
environment).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path


def default_path() -> Path:
    cr = os.environ.get("LR_CR")
    if not cr:
        raise SystemExit("error: set LR_CR (the campaign root) or pass --path")
    return Path(cr) / "facts" / "live.json"


def parse_fields(items: list[str]) -> dict:
    fields = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"error: --field needs key=value, got {item!r}")
        try:
            fields[key] = json.loads(value)
        except json.JSONDecodeError:
            fields[key] = value
    return fields


def update(path: Path, job_id: str, fields: dict, create: bool) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        data.setdefault("schema", "learned-routing.live.v1")
        jobs = data.setdefault("jobs", [])
        entry = next((j for j in jobs if str(j.get("job_id")) == job_id), None)
        if entry is None:
            if not create:
                raise SystemExit(f"error: job {job_id} is not recorded in {path}")
            entry = {
                "job_id": job_id,
                "recorded_utc": time.strftime("%FT%TZ", time.gmtime()),
            }
            jobs.append(entry)
        elif create:
            raise SystemExit(f"error: job {job_id} is already recorded in {path}")
        entry.update(fields)
        entry["updated_utc"] = time.strftime("%FT%TZ", time.gmtime())
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)
        return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["add-job", "set-job"])
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args(argv)
    entry = update(
        args.path or default_path(),
        args.job_id,
        parse_fields(args.field),
        args.command == "add-job",
    )
    print(json.dumps(entry, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
