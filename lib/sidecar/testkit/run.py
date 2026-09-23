#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect and execute isolated sidecar units by backend and earliest CI lane."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

LANES = ("pre-merge", "post-merge", "nightly")
LANE_PREFIX = "__sidecar_lane_"
MANIFEST_VERSION = 1
UNIT_BACKENDS = {
    "vllm": {
        "name": "dynamo-vllm-sidecar-lib",
        "package": "dynamo-vllm-sidecar",
        "framework": "vllm",
        "extra": [],
    },
}


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    try:
        return subprocess.run(command, check=True, timeout=1800, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            if "--message-format=json" in command:
                for line in error.stdout.splitlines():
                    message = json.loads(line)
                    if message.get("reason") == "compiler-message":
                        print(message["message"]["rendered"], end="", flush=True)
            else:
                print(error.stdout, flush=True)
        if error.stderr:
            print(error.stderr, flush=True)
        raise


def specifications(suite="unit", framework="all"):
    if suite != "unit":
        raise ValueError(f"Unsupported test suite: {suite}")
    if framework != "all" and framework not in UNIT_BACKENDS:
        raise ValueError(f"Unit suites are not implemented for {framework}")
    backends = UNIT_BACKENDS if framework == "all" else [framework]
    return [
        {
            "name": "dynamo-sidecar-common-lib",
            "package": "dynamo-sidecar-common",
            "framework": "common",
            "extra": ["--features", "tonic-v14"],
        },
        *[UNIT_BACKENDS[name] for name in backends],
    ]


def selected_lanes(lane):
    if lane == "all":
        return LANES
    return LANES[: LANES.index(lane) + 1]


def libtest_args(lane):
    return [
        argument
        for excluded in LANES
        if excluded not in selected_lanes(lane)
        for argument in ("--skip", f"::{LANE_PREFIX}{excluded.replace('-', '_')}")
    ]


def inventory(output, framework):
    cases = []
    for line in output.splitlines():
        if not line.endswith(": test"):
            continue
        name = line.removesuffix(": test")
        parts = name.split("::")
        markers = [part for part in parts if part.startswith(LANE_PREFIX)]
        if not markers and not any(part.startswith("unit_") for part in parts):
            continue
        if len(markers) != 1 or parts[-1] != markers[0]:
            raise RuntimeError(f"Expected one terminal lane marker on {name}")
        lane = markers[0].removeprefix(LANE_PREFIX).replace("_", "-")
        if lane not in LANES:
            raise RuntimeError(f"Unknown lane on {name}: {lane}")
        cases.append(
            {
                "name": name,
                "scenario": "::".join(parts[:-1]),
                "lane": lane,
                "category": (
                    "common"
                    if framework == "common"
                    else "shared"
                    if any(
                        part == "shared" or part.startswith("unit_shared_")
                        for part in parts
                    )
                    else "native"
                ),
            }
        )
    names = [case["name"] for case in cases]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Empty or duplicate governed test inventory")
    return sorted(cases, key=lambda case: case["name"])


def collect(binary, framework):
    result = run(
        [str(binary), "--list", "--format=terse"], capture_output=True, text=True
    )
    cases = inventory(result.stdout, framework)
    ignored = run(
        [str(binary), "--ignored", "--list", "--format=terse"],
        capture_output=True,
        text=True,
    )
    ignored_names = set(ignored.stdout.splitlines())
    if any(f"{case['name']}: test" in ignored_names for case in cases):
        raise RuntimeError(f"Governed unit tests must not be ignored: {binary}")
    return cases


def select_cases(cases, lane):
    return [case for case in cases if case["lane"] in selected_lanes(lane)]


def execute(binary, cases):
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = run(
        [str(binary), "--exact", *[case["name"] for case in cases], "--nocapture"],
        capture_output=True,
        text=True,
        env=environment,
    )
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", flush=True)
    summary = re.findall(
        r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", result.stdout
    )
    if summary != [(str(len(cases)), "0", "0")]:
        raise RuntimeError(
            f"Failed, ignored, or incompletely executed test result from {binary}"
        )


def build(root, spec):
    result = run(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            spec["package"],
            "--lib",
            *spec["extra"],
            "--no-run",
            "--message-format=json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    target_name = spec["package"].replace("-", "_")
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("executable")
            and message["profile"]["test"]
            and message["target"]["name"] == target_name
        ):
            return message["executable"]
    raise RuntimeError(f"Cargo did not produce {spec['package']}/lib")


def select_entries(manifest, specs):
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        raise RuntimeError("Unsupported exported manifest; export the unit suite again")
    entries = manifest["entries"]
    allowed = {spec["name"]: spec for spec in specifications()}
    names = [entry["name"] for entry in entries]
    expected = {spec["name"] for spec in specs}
    if (
        len(names) != len(set(names))
        or not set(names).issubset(allowed)
        or not expected.issubset(names)
    ):
        raise RuntimeError("Incomplete, unexpected, or duplicate exported suite")
    for entry in entries:
        spec = allowed[entry["name"]]
        if (
            entry["framework"] != spec["framework"]
            or entry["suite"] != "unit"
            or entry["path"] != entry["name"]
        ):
            raise RuntimeError(f"Invalid exported suite metadata: {entry['name']}")
    return [entry for entry in entries if entry["name"] in expected]


def workspace_tests(root, lane):
    specs = specifications()
    for spec in specs:
        collect(build(root, spec), spec["framework"])
    owners = [spec["package"] for spec in specs]
    run(
        [
            "cargo",
            "test",
            "--locked",
            "--workspace",
            "--all-targets",
            *[arg for owner in owners for arg in ("--exclude", owner)],
        ],
        cwd=root,
    )
    # Unit owners use libtest; unrelated custom harnesses keep their original arguments.
    run(
        [
            "cargo",
            "test",
            "--locked",
            "--all-targets",
            *[arg for owner in owners for arg in ("-p", owner)],
            "--",
            *libtest_args(lane),
        ],
        cwd=root,
    )


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["unit"], default="unit")
    parser.add_argument("--framework", choices=["vllm", "sglang", "all"], default="all")
    parser.add_argument("--lane", choices=[*LANES, "all"])
    parser.add_argument(
        "--level", choices=["unit", "pre-merge", "all"], help="deprecated alias"
    )
    parser.add_argument("--list", action="store_true", help="collect without execution")
    parser.add_argument(
        "--libtest-args", action="store_true", help="print lane filters, one per line"
    )
    parser.add_argument(
        "--workspace-tests",
        action="store_true",
        help="run workspace tests with sidecar lane filters",
    )
    artifacts = parser.add_mutually_exclusive_group()
    artifacts.add_argument("--export", type=Path, help="export all unit lanes")
    artifacts.add_argument("--artifacts", type=Path, help="use exported test binaries")
    args = parser.parse_args(argv)
    if args.level:
        alias = "pre-merge" if args.level == "pre-merge" else "all"
        if args.lane is not None and args.lane != alias:
            parser.error("--level conflicts with --lane")
        args.lane = alias
    args.lane = args.lane or "all"
    if args.framework != "all" and args.framework not in UNIT_BACKENDS:
        parser.error(f"{args.framework} unit suites are not implemented yet")
    if args.export and args.list:
        parser.error("--export builds artifacts; use --artifacts with --list")
    if args.libtest_args and (args.export or args.artifacts or args.list):
        parser.error("--libtest-args cannot be combined with collection or artifacts")
    if args.workspace_tests and (
        args.export
        or args.artifacts
        or args.list
        or args.libtest_args
        or args.framework != "all"
    ):
        parser.error(
            "--workspace-tests runs all frameworks and cannot use artifact or listing options"
        )
    return args


def main():
    args = arguments()
    if args.workspace_tests:
        workspace_tests(Path(__file__).resolve().parents[3], args.lane)
        return
    if args.libtest_args:
        for argument in libtest_args(args.lane):
            print(argument)
        return
    specs = specifications(args.suite, args.framework)
    if args.artifacts:
        args.artifacts = args.artifacts.resolve()
        entries = select_entries(
            json.loads((args.artifacts / "tests.json").read_text()), specs
        )
        for entry in entries:
            binary = args.artifacts / entry["path"]
            if collect(binary, entry["framework"]) != entry["cases"]:
                raise RuntimeError(f"Exported inventory does not match {binary}")
    else:
        root = Path(__file__).resolve().parents[3]
        entries = []
        for spec in specs:
            binary = build(root, spec)
            entries.append(
                {
                    "name": spec["name"],
                    "path": binary,
                    "suite": args.suite,
                    "framework": spec["framework"],
                    "cases": collect(binary, spec["framework"]),
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
                json.dumps({"version": MANIFEST_VERSION, "entries": entries}, indent=2)
                + "\n"
            )
            return
    selected = [(entry, select_cases(entry["cases"], args.lane)) for entry in entries]
    if not any(cases for _, cases in selected):
        raise RuntimeError(f"No unit tests selected for {args.framework}/{args.lane}")
    for entry, cases in selected:
        for case in cases:
            print(
                f"{entry['framework']} {case['category']} {case['lane']} "
                f"{case['scenario']}",
                flush=True,
            )
        if cases and not args.list:
            binary = (
                args.artifacts / entry["path"]
                if args.artifacts
                else Path(entry["path"])
            )
            execute(binary, cases)
    print(f"Selected {sum(len(cases) for _, cases in selected)} unit tests")


if __name__ == "__main__":
    main()
