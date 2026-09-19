#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, collect, and run the implemented CPU wire suite."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    try:
        return subprocess.run(command, check=True, timeout=1800, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, flush=True)
        if error.stderr:
            print(error.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["vllm", "all"], default="all")
    parser.add_argument("--level", choices=["wire"], default="wire")
    parser.add_argument("--list", action="store_true", help="collect without execution")
    parser.add_argument(
        "--export", type=Path, help="export executable tests for a CPU container"
    )
    parser.add_argument(
        "--artifacts", type=Path, help="execute previously exported tests"
    )
    args = parser.parse_args()
    if args.artifacts:
        entries = json.loads((args.artifacts / "tests.json").read_text())
    else:
        root = Path(__file__).resolve().parents[3]
        entries = []
        for package, target in [
            ("dynamo-sidecar-testkit", "conformance"),
            ("dynamo-vllm-mocker", "sidecar"),
        ]:
            command = [
                "cargo",
                "test",
                "--locked",
                "-p",
                package,
                "--test",
                target,
                "--no-run",
                "--message-format=json",
            ]
            result = run(command, cwd=root, capture_output=True, text=True)
            for line in result.stdout.splitlines():
                message = json.loads(line)
                if (
                    message.get("reason") == "compiler-artifact"
                    and message.get("executable")
                    and message["profile"]["test"]
                    and message["target"]["name"] == target
                ):
                    entries.append(
                        {"name": f"{package}-{target}", "path": message["executable"]}
                    )
            if not any(entry["name"] == f"{package}-{target}" for entry in entries):
                raise RuntimeError(f"Cargo did not produce {package}/{target}")
        if args.export:
            args.export.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                destination = args.export / entry["name"]
                run(["strip", "-o", str(destination), entry["path"]])
                entry["path"] = entry["name"]
            (args.export / "tests.json").write_text(
                json.dumps(entries, indent=2) + "\n"
            )
            return
    expected = {"dynamo-sidecar-testkit-conformance", "dynamo-vllm-mocker-sidecar"}
    if {entry["name"] for entry in entries} != expected or len(entries) != len(
        expected
    ):
        raise RuntimeError("Incomplete or duplicate exported wire suite")
    for entry in entries:
        binary = (
            args.artifacts / entry["path"] if args.artifacts else Path(entry["path"])
        )
        collected = run(
            [str(binary), "--list", "--format=terse"], capture_output=True, text=True
        )
        print(collected.stdout, end="", flush=True)
        cases = [
            line for line in collected.stdout.splitlines() if line.endswith(": test")
        ]
        if not cases:
            raise RuntimeError(f"No cases collected from {binary}")
        if not args.list:
            result = run(
                [str(binary), "--nocapture"],
                capture_output=True,
                text=True,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""},
            )
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", flush=True)
            summary = re.findall(
                r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;",
                result.stdout,
            )
            if summary != [(str(len(cases)), "0", "0")]:
                raise RuntimeError(
                    f"Failed, ignored, or incompletely executed test result from {binary}"
                )


if __name__ == "__main__":
    main()
