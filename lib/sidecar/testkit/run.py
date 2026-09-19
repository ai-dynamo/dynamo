#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect and execute isolated unit and CPU wire sidecar tests."""

import argparse
import json
import os
import re
import shutil
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


def specifications(level):
    specs = []
    if level in ("unit", "pre-merge", "all"):
        for package in ("dynamo-sidecar-common", "dynamo-vllm-sidecar"):
            extra = (
                ["--features", "tonic-v14"]
                if package == "dynamo-sidecar-common"
                else []
            )
            specs.append((package, "lib", extra, ["unit_"], "unit"))
    if level in ("wire", "pre-merge", "all"):
        specs += [
            ("dynamo-sidecar-testkit", "conformance", [], [], "wire"),
            ("dynamo-vllm-mocker", "sidecar", [], [], "wire"),
        ]
        for package in ("dynamo-sidecar-common", "dynamo-vllm-sidecar"):
            extra = (
                ["--features", "tonic-v14"]
                if package == "dynamo-sidecar-common"
                else []
            )
            specs.append(
                (package, "lib", extra, ["--skip", "unit_", "--test-threads=1"], "wire")
            )
    return specs


def build(root, package, target, extra):
    selector = ["--lib"] if target == "lib" else ["--test", target]
    result = run(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            package,
            *selector,
            *extra,
            "--no-run",
            "--message-format=json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        message = json.loads(line)
        target_name = package.replace("-", "_") if target == "lib" else target
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("executable")
            and message["profile"]["test"]
            and message["target"]["name"] == target_name
        ):
            return message["executable"]
    raise RuntimeError(f"Cargo did not produce {package}/{target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["vllm", "all"], default="all")
    parser.add_argument(
        "--level",
        choices=["unit", "wire", "pre-merge", "all"],
        default="all",
    )
    parser.add_argument("--list", action="store_true", help="collect without execution")
    parser.add_argument(
        "--export",
        type=Path,
        help="export executable tests for a CPU container",
    )
    parser.add_argument(
        "--artifacts", type=Path, help="execute previously exported tests"
    )
    args = parser.parse_args()
    if args.artifacts:
        entries = json.loads((args.artifacts / "tests.json").read_text())
        selected = {spec[-1] for spec in specifications(args.level)}
        entries = [entry for entry in entries if entry["level"] in selected]
        expected = {
            (
                level,
                f"{package}-{target}",
                tuple(filters),
            )
            for package, target, _, filters, level in specifications(args.level)
        }
        actual = {
            (entry["level"], entry["name"], tuple(entry["filters"]))
            for entry in entries
        }
        if actual != expected or len(entries) != len(expected):
            raise RuntimeError(
                f"Incomplete or duplicate exported suite: expected {expected}, got {actual}"
            )
    else:
        root = Path(__file__).resolve().parents[3]
        entries = []
        for package, target, extra, filters, level in specifications(args.level):
            name = f"{package}-{target}"
            entries.append(
                {
                    "name": name,
                    "path": build(root, package, target, extra),
                    "filters": filters,
                    "level": level,
                }
            )
        if args.export:
            args.export.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                destination = args.export / entry["name"]
                run(["strip", "-o", str(destination), entry["path"]])
                entry["path"] = entry["name"]
            shutil.copy2(__file__, args.export / "run.py")
            (args.export / "tests.json").write_text(
                json.dumps(entries, indent=2) + "\n"
            )
            return
    if not entries:
        raise RuntimeError(f"No exported tests for {args.level}")
    for entry in entries:
        binary = (
            args.artifacts / entry["path"] if args.artifacts else Path(entry["path"])
        )
        collected = run(
            [str(binary), *entry["filters"], "--list", "--format=terse"],
            capture_output=True,
            text=True,
        )
        print(collected.stdout, end="", flush=True)
        cases = [
            line for line in collected.stdout.splitlines() if line.endswith(": test")
        ]
        if not cases:
            raise RuntimeError(f"No cases collected from {binary}")
        if not args.list:
            environment = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
            result = run(
                [str(binary), *entry["filters"], "--nocapture"],
                capture_output=True,
                text=True,
                env=environment,
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
