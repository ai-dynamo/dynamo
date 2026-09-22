#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect and execute sidecar tests; all selects CPU layers, native is explicit."""

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


def specifications(level, framework="all"):
    if framework not in ("vllm", "all"):
        raise ValueError(f"Unsupported framework: {framework}")
    specs = []
    if level in ("unit", "pre-merge", "all"):
        for package in ("dynamo-sidecar-common", "dynamo-vllm-sidecar"):
            extra = (
                ["--features", "tonic-v14"]
                if package == "dynamo-sidecar-common"
                else []
            )
            specs.append((package, "lib", extra, ["unit_"], "unit"))
    if level in ("wire", "pre-merge", "integration", "all"):
        specs += [
            (
                "dynamo-sidecar-testkit",
                "conformance",
                [],
                ["--skip", "sglang::"],
                "wire",
            ),
            ("dynamo-vllm-mocker", "sidecar", [], [], "wire"),
        ]
        if framework == "all":
            specs += [
                ("dynamo-sidecar-testkit", "conformance", [], ["sglang::"], "wire"),
                ("dynamo-sglang-mocker", "sidecar", [], [], "wire"),
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
    if level in ("process", "integration", "all"):
        specs.append(
            (
                "dynamo-sidecar-testkit",
                "cross_process",
                ["--features", "process-tests"],
                ["--test-threads=1"],
                "process",
            )
        )
    if level == "native":
        specs.append(
            (
                "dynamo-sidecar-testkit",
                "native_engine",
                ["--features", "native-tests"],
                [],
                "native",
            )
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
    target_name = package.replace("-", "_") if target == "lib" else target
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("executable")
            and message["profile"]["test"]
            and message["target"]["name"] == target_name
        ):
            return message["executable"]
    raise RuntimeError(f"Cargo did not produce {package}/{target}")


def artifact_name(package, target, filters, level):
    if level == "native":
        return "native_engine"
    if target == "conformance":
        backend = "sglang" if filters == ["sglang::"] else "vllm"
        return f"{package}-{target}-{backend}"
    return f"{package}-{target}"


def select_entries(entries, specs, supported_specs):
    def spec_key(spec):
        package, target, _, filters, level = spec
        return level, artifact_name(package, target, filters, level), tuple(filters)

    def entry_key(entry):
        return entry["level"], entry["name"], tuple(entry["filters"])

    expected = {spec_key(spec) for spec in specs}
    allowed = {spec_key(spec) for spec in supported_specs}
    actual = [entry_key(entry) for entry in entries]
    if (
        len(actual) != len(set(actual))
        or not set(actual).issubset(allowed)
        or not expected.issubset(actual)
    ):
        raise RuntimeError(
            "Incomplete, unexpected, or duplicate exported suite: "
            f"expected {expected}, got {actual}"
        )
    return [entry for entry in entries if entry_key(entry) in expected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework",
        choices=["vllm", "all"],
        default="all",
        help=(
            "all also preserves baseline SGLang wire/Mocker coverage; "
            "process/native remain vLLM-only"
        ),
    )
    parser.add_argument(
        "--level",
        choices=[
            "unit",
            "wire",
            "process",
            "integration",
            "native",
            "pre-merge",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--list", action="store_true", help="collect without execution")
    artifacts = parser.add_mutually_exclusive_group()
    artifacts.add_argument(
        "--export",
        type=Path,
        help="export executable tests for a CPU container or native engine job",
    )
    artifacts.add_argument(
        "--artifacts", type=Path, help="execute previously exported tests"
    )
    args = parser.parse_args()
    if args.export and args.list:
        parser.error(
            "--export builds artifacts; use --artifacts with --list to collect them"
        )
    specs = specifications(args.level, args.framework)
    if args.artifacts:
        args.artifacts = args.artifacts.resolve()
        entries = select_entries(
            json.loads((args.artifacts / "tests.json").read_text()),
            specs,
            specifications("all", "all") + specifications("native", "all"),
        )
    else:
        root = Path(__file__).resolve().parents[3]
        entries = []
        for package, target, extra, filters, level in specs:
            name = artifact_name(package, target, filters, level)
            entries.append(
                {
                    "name": name,
                    "path": build(root, package, target, extra),
                    "filters": filters,
                    "level": level,
                }
            )
        if any(entry["level"] == "process" for entry in entries):
            run(
                [
                    "cargo",
                    "build",
                    "--locked",
                    "-p",
                    "dynamo-vllm-sidecar",
                    "--bin",
                    "dynamo-vllm-sidecar",
                ],
                cwd=root,
            )
            process = next(entry for entry in entries if entry["level"] == "process")
            binary = Path(process["path"]).parent.parent / "dynamo-vllm-sidecar"
            os.environ["DYNAMO_VLLM_SIDECAR"] = str(binary)
        if args.export:
            args.export.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                destination = args.export / entry["name"]
                run(["strip", "-o", str(destination), entry["path"]])
                entry["path"] = entry["name"]
            if any(entry["level"] == "process" for entry in entries):
                run(
                    [
                        "strip",
                        "-o",
                        str(args.export / "dynamo-vllm-sidecar"),
                        os.environ["DYNAMO_VLLM_SIDECAR"],
                    ]
                )
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
            if entry["level"] == "native":
                raise RuntimeError(
                    "Run native tests through tests/sidecar/test_native_integration.py "
                    "with DYNAMO_SIDECAR_NATIVE_TEST set to the exported binary"
                )
            environment = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
            if args.artifacts and entry["level"] == "process":
                environment["DYNAMO_VLLM_SIDECAR"] = str(
                    args.artifacts / "dynamo-vllm-sidecar"
                )
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
                    "Failed, ignored, or incompletely executed test result "
                    f"from {binary}"
                )


if __name__ == "__main__":
    main()
