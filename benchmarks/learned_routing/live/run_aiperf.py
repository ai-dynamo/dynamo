#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run AIPerf on a ``gen_aiperf_inputs`` output, exactly as its manifest specifies.

The tool fills ``${URL}`` and ``${ARTIFACT_DIR}`` into the manifest's argv, sets the manifest's
environment, and runs from the input directory, so ``--input-file`` resolves. It refuses an
existing artifact directory. Next to that directory it writes ``<artifact>.run.json``: the argv,
the environment, the AIPerf version, the input and manifest SHA-256s, the wall-clock start and
end, and the exit code.

Standard library only: it runs from the AIPerf environment itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--url", required=True, help="frontend base URL, e.g. http://host:8000"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--aiperf", default="aiperf", help="aiperf executable")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument(
        "--verify-input", action="store_true", help="re-hash the input file"
    )
    args = parser.parse_args(argv)

    inputs = args.inputs.resolve()
    artifact = args.artifact_dir.resolve()
    if artifact.exists():
        raise SystemExit(
            f"{artifact} exists; every run needs a fresh artifact directory"
        )
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if args.verify_input:
        for name, expected in manifest["files"].items():
            actual = sha256_file(inputs / name)
            if actual != expected:
                raise SystemExit(f"{name}: sha256 {actual} != manifest {expected}")
    spec = manifest["aiperf"]
    command = [
        args.aiperf if part == "aiperf" and i == 0 else part
        for i, part in enumerate(spec["argv"])
    ]
    command = [
        part.replace("${URL}", args.url).replace("${ARTIFACT_DIR}", str(artifact))
        for part in command
    ]
    env = dict(os.environ)
    env.update(spec["env"])
    version = (
        subprocess.run(
            [command[0], "--version"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        .stdout.strip()
        .splitlines()
    )
    record = {
        "inputs": str(inputs),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "files": manifest["files"],
        "argv": command,
        "env": spec["env"],
        "aiperf_version": version[-1] if version else None,
        "expected_aiperf_version": spec["version"],
        "start_unix_ns": time.time_ns(),
    }
    run_json = artifact.with_name(artifact.name + ".run.json")
    run_json.write_text(json.dumps(record, indent=1) + "\n")
    try:
        completed = subprocess.run(
            command, cwd=inputs, env=env, timeout=args.timeout_s, check=False
        )
        record["exit_code"] = completed.returncode
    except subprocess.TimeoutExpired:
        record["exit_code"] = None
        record["timed_out"] = True
    record["end_unix_ns"] = time.time_ns()
    run_json.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"run": str(run_json), "exit_code": record["exit_code"]}))
    return 0 if record["exit_code"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
