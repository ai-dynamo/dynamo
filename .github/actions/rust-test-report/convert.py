#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert existing cargo console logs to JUnit after test execution.

No per-test times are inferred. Ambiguous or truncated suites stay in the log,
but are excluded from XML. This is a temporary adapter for stable Rust.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
XML_INVALID = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
HEADER = re.compile(r"^\s*(?:Running (.+?) \((.+)\)|Doc-tests (.+))\s*$")
START = re.compile(r"^running (\d+) tests?$")
CASE = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored(?:, .*)?)$")
PREFIX = re.compile(r"^test (.+?) \.\.\. .*$")
OUTCOME = re.compile(r"^(ok|FAILED|ignored(?:, .*)?)$")
SUMMARY = re.compile(
    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; "
    r"(\d+) ignored; (\d+) measured; \d+ filtered out; finished in ([\d.]+)s$"
)


@dataclass
class Suite:
    name: str
    expected: int
    cases: dict[str, str] = field(default_factory=dict)
    tail: str = ""
    ambiguous: bool = False
    pending: str | None = None


def clean(text: str) -> str:
    return XML_INVALID.sub("", ANSI.sub("", text))


def parse_log(
    lines: Iterable[str], group: str, serial: bool = False
) -> tuple[ET.Element, list[str]]:
    """Accept only suites whose observed outcomes match the libtest totals."""
    report = ET.Element("testsuites")
    warnings: list[str] = []
    header = ""
    active: Suite | None = None
    for raw in lines:
        line = clean(raw).rstrip("\r\n")
        match = HEADER.fullmatch(line)
        if match:
            if active is not None:
                warnings.append(f"Incomplete suite: {active.name}")
                active = None
            source, executable, doctest = match.groups()
            if doctest:
                header = f"doctest/{doctest}"
            else:
                binary = re.sub(r"-[0-9a-f]{16}(?:\.exe)?$", "", Path(executable).name)
                header = f"{binary}/{source}"
            continue
        match = START.fullmatch(line)
        if match:
            if active is not None:
                warnings.append(f"Incomplete suite: {active.name}")
            # Without a cargo header, test-looking application logs are ambiguous.
            active = Suite(f"{group}/{header}", int(match[1])) if header else None
            continue
        if active is None:
            continue
        active.tail = (active.tail + line + "\n")[-16000:]
        match = CASE.fullmatch(line)
        if match:
            name, outcome = match.groups()
            name = name.removesuffix(" - should panic")
            if name in active.cases or active.pending is not None:
                active.ambiguous = True
            active.cases[name] = outcome
            continue
        # With one test thread, --nocapture can put application output between
        # the test-name prefix and its outcome. Never infer this association
        # for parallel tests, where another test can own the output.
        if serial:
            match = PREFIX.fullmatch(line)
            if match:
                if active.pending is not None:
                    active.ambiguous = True
                active.pending = match[1].removesuffix(" - should panic")
                continue
            if OUTCOME.fullmatch(line):
                if active.pending is None or active.pending in active.cases:
                    active.ambiguous = True
                else:
                    active.cases[active.pending] = line
                active.pending = None
                continue
        match = SUMMARY.fullmatch(line)
        if not match:
            continue
        passed, failed, ignored, measured = map(int, match.groups()[:4])
        observed = (
            sum(status == "ok" for status in active.cases.values()),
            sum(status == "FAILED" for status in active.cases.values()),
            sum(status.startswith("ignored") for status in active.cases.values()),
        )
        if (
            active.ambiguous
            or active.pending is not None
            or measured
            or observed != (passed, failed, ignored)
            or len(active.cases) != active.expected
        ):
            warnings.append(f"Ambiguous suite omitted: {active.name}")
        elif active.cases:
            suite = ET.SubElement(
                report,
                "testsuite",
                name=active.name,
                tests=str(len(active.cases)),
                failures=str(failed),
                errors="0",
                skipped=str(ignored),
                time=match[5],
            )
            for name, outcome in active.cases.items():
                case = ET.SubElement(
                    suite, "testcase", name=name, classname=active.name
                )
                if outcome == "FAILED":
                    ET.SubElement(case, "failure", message="Rust test failed").text = (
                        "Suite output tail; complete output is in the log artifact.\n"
                        + active.tail
                    )
                elif outcome.startswith("ignored"):
                    ET.SubElement(case, "skipped", message=outcome)
        active = None
        header = ""
    if active is not None:
        warnings.append(f"Incomplete suite omitted: {active.name}")
    return report, warnings


def convert_log(log_path: Path, workspace: str, serial: bool) -> list[str]:
    """Convert one completed log without executing a test command."""
    xml_path = log_path.with_suffix(".xml")
    xml_path.unlink(missing_ok=True)
    group = f"{workspace}/{log_path.stem}"
    with log_path.open(encoding="utf-8", errors="replace") as log:
        report, warnings = parse_log(log, group, serial=serial)
    if len(report):
        temporary = xml_path.with_suffix(".xml.tmp")
        ET.ElementTree(report).write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(xml_path)
    else:
        warnings.append(f"No complete libtest results in {log_path.name}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--serial-log", action="append", default=[], help="Single-threaded log stem"
    )
    args = parser.parse_args()
    warnings: list[str] = []
    for log_path in sorted(args.log_dir.glob("*.log")):
        warnings.extend(
            convert_log(log_path, args.workspace, log_path.stem in args.serial_log)
        )
    for warning in warnings:
        print(f"::warning::{warning}", file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(f"\nRust console reports: `{args.workspace}`.\n")
            stream.write("Per-test durations are unavailable from console output.\n")
            for warning in warnings:
                stream.write(f"\n- {warning}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
