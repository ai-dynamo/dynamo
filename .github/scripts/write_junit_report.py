# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a minimal JUnit XML report for jobs without native JUnit output."""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def normalize_artifact_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return name or "junit-results"


def write_report(
    output: Path, suite_name: str, test_name: str, status: str, message: str
) -> None:
    status = status or "failure"
    skipped = status in {"cancelled", "skipped"}
    failed = status != "success" and not skipped
    failures = 1 if failed else 0
    skipped_count = 1 if skipped else 0

    testsuites = ET.Element(
        "testsuites",
        tests="1",
        failures=str(failures),
        errors="0",
        skipped=str(skipped_count),
    )
    testsuite = ET.SubElement(
        testsuites,
        "testsuite",
        name=suite_name,
        tests="1",
        failures=str(failures),
        errors="0",
        skipped=str(skipped_count),
        time="0",
    )
    testcase = ET.SubElement(
        testsuite,
        "testcase",
        classname=suite_name,
        name=test_name,
        time="0",
    )
    if skipped:
        ET.SubElement(
            testcase, "skipped", message=message or f"{suite_name} was skipped"
        )
    elif failed:
        failure = ET.SubElement(
            testcase,
            "failure",
            message=message or f"{suite_name} completed with status {status}",
        )
        failure.text = message or f"{suite_name} completed with status {status}."

    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(testsuites)
    ET.ElementTree(testsuites).write(output, encoding="utf-8", xml_declaration=True)


def write_github_output(output_name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output or not output_name:
        return

    with open(github_output, "a", encoding="utf-8") as out:
        out.write(f"{output_name}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--test-name", default="")
    parser.add_argument("--status", required=True)
    parser.add_argument("--output", default="test-results/junit.xml")
    parser.add_argument("--message", default="")
    parser.add_argument("--artifact-name", default="")
    parser.add_argument("--artifact-output-name", default="")
    args = parser.parse_args()

    test_name = args.test_name or args.suite_name
    write_report(
        Path(args.output), args.suite_name, test_name, args.status, args.message
    )

    if args.artifact_name:
        write_github_output(
            args.artifact_output_name, normalize_artifact_name(args.artifact_name)
        )


if __name__ == "__main__":
    main()
