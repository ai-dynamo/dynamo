#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pytest Marker Report (Production Grade)

- Collects pytest tests without executing them
- Prints markers and validates category coverage
- Optionally mocks unavailable dependencies so tests in import paths do
  not fail collection
- Provides structured output suitable for CI (text, JSON)
"""

from __future__ import annotations

import argparse
import configparser
import importlib
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock

import pytest

try:
    import tomllib  # Python >=3.11
except ImportError:
    import tomli as tomllib  # type: ignore

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOG = logging.getLogger("pytest-marker-report")
# Disable all logging except CRITICAL to suppress noise from test code collection
logging.disable(logging.WARNING)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REQUIRED_CATEGORIES: Dict[str, Set[str]] = {
    "Lifecycle": {"pre_merge", "post_merge", "nightly", "weekly", "release"},
    "Test Type": {
        "unit",
        "integration",
        "e2e",
        "benchmark",
        "stress",
        "multimodal",
        "performance",
    },
    "Hardware": {"gpu_0", "gpu_1", "gpu_2", "gpu_4", "gpu_8", "h100", "k8s"},
}

STUB_MODULES = [
    "pytest_httpserver",
    "pytest_httpserver.HTTPServer",
    "pytest_benchmark",
    "pytest_benchmark.logger",
    "pytest_benchmark.plugin",
    "kubernetes",
    "kubernetes_asyncio",
    "kubernetes_asyncio.client",
    "kubernetes_asyncio.client.exceptions",
    "kubernetes.client",
    "kubernetes.config",
    "kubernetes.config.config_exception",
    "kr8s",
    "kr8s.objects",
    "tritonclient",
    "tritonclient.grpc",
    "aiohttp",
    "aiofiles",
    "httpx",
    "tabulate",
    "prometheus_api_client",
    "huggingface_hub",
    "huggingface_hub.model_info",
    "transformers",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "pmdarima",
    "prophet",
    "filterpy",
    "filterpy.kalman",
    "scipy",
    "scipy.interpolate",
    "nats",
    "dynamo._core",
    "psutil",
    "requests",
    "numpy",
    "gradio",
    "aiconfigurator",
    "aiconfigurator.webapp",
    "aiconfigurator.webapp.components",
    "aiconfigurator.webapp.components.profiling",
    "boto3",
    "botocore",
    "botocore.client",
    "botocore.exceptions",
]

# Project paths for local imports
PROJECT_PATHS = [
    os.getcwd(),
    os.path.join(os.getcwd(), "components", "src"),
    os.path.join(os.getcwd(), "lib", "bindings", "python", "src"),
]
sys.path[:0] = PROJECT_PATHS  # prepend to sys.path

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def sanitize(s: str, max_len: int = 200) -> str:
    """Safe, trimmed string for output."""
    s = re.sub(r"[^\x20-\x7E\n\t]", "", str(s))
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def missing_categories(markers: Set[str]) -> List[str]:
    """Return required categories missing in a test's markers."""
    return [
        cat for cat, allowed in REQUIRED_CATEGORIES.items() if not (markers & allowed)
    ]


# --------------------------------------------------------------------------- #
# Dependency Stubbing
# --------------------------------------------------------------------------- #


class DependencyStubber:
    """Stub unavailable modules to allow test collection without real dependencies."""

    def __init__(self):
        self.stubbed: Set[str] = set()

    def _create_module_stub(self, name: str) -> MagicMock:
        """Create a stub module with proper Python module attributes."""
        stub = MagicMock()
        stub.__path__ = []
        stub.__name__ = name
        stub.__loader__ = None
        stub.__spec__ = None
        stub.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        return stub

    def ensure_available(self, module_name: str) -> ModuleType:
        """Ensure a module is available, stubbing it if not installed."""
        if module_name in sys.modules:
            return sys.modules[module_name]

        parts = module_name.split(".")
        parent_stubbed = any(
            ".".join(parts[:i]) in self.stubbed for i in range(1, len(parts))
        )

        if not parent_stubbed:
            try:
                return importlib.import_module(module_name)
            except (ImportError, AttributeError):
                pass

        # Create parent packages if needed
        for i in range(1, len(parts)):
            sub = ".".join(parts[:i])
            if sub not in sys.modules:
                pkg = ModuleType(sub)
                pkg.__path__ = []
                sys.modules[sub] = pkg
                self.stubbed.add(sub)

        # Create stub module with proper attributes
        stub = self._create_module_stub(module_name)
        sys.modules[module_name] = stub
        self.stubbed.add(module_name)
        return stub


# --------------------------------------------------------------------------- #
# Data Structures
# --------------------------------------------------------------------------- #


@dataclass
class TestRecord:
    nodeid: str
    markers: List[str]
    missing: List[str]


@dataclass
class Report:
    total_checked: int
    total_skipped_mypy: int
    total_missing: int
    tests: List[TestRecord]
    undeclared_markers: Optional[List[str]] = None
    missing_in_project_config: Optional[List[str]] = None


# --------------------------------------------------------------------------- #
# Pytest Plugin
# --------------------------------------------------------------------------- #


class MarkerReportPlugin:
    def __init__(self):
        self.records: List[TestRecord] = []
        self.checked = 0
        self.skipped_mypy = 0

    def pytest_collection_modifyitems(self, session, config, items):
        for item in items:
            markers = {m.name for m in item.iter_markers()}
            if "mypy" in markers:
                self.skipped_mypy += 1
                continue

            record = TestRecord(
                nodeid=sanitize(item.nodeid),
                markers=sorted(markers),
                missing=missing_categories(markers),
            )
            self.records.append(record)
            self.checked += 1

    def build_report(self) -> Report:
        return Report(
            total_checked=self.checked,
            total_skipped_mypy=self.skipped_mypy,
            total_missing=sum(bool(r.missing) for r in self.records),
            tests=self.records,
        )


# --------------------------------------------------------------------------- #
# Marker Validation
# --------------------------------------------------------------------------- #


def load_declared_markers(project_root: Path = Path(".")) -> Set[str]:
    """Load declared pytest markers from pytest.ini and pyproject.toml."""
    declared: Set[str] = set()

    # pytest.ini
    ini_path = project_root / "pytest.ini"
    if ini_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(str(ini_path))
        markers = cfg.get("pytest", "markers", fallback="")
        declared.update(
            line.split(":", 1)[0].strip()
            for line in markers.splitlines()
            if line.strip()
        )

    # pyproject.toml
    toml_path = project_root / "pyproject.toml"
    if toml_path.exists():
        try:
            with toml_path.open("rb") as f:
                data = tomllib.load(f)
            markers_list = (
                data.get("tool", {})
                .get("pytest", {})
                .get("ini_options", {})
                .get("markers", [])
            )
            declared.update(
                line.split(":", 1)[0].strip() for line in markers_list if line.strip()
            )
        except Exception as e:
            LOG.warning("Failed reading pyproject.toml markers: %s", e)

    return declared


def validate_marker_definitions(report: Report, declared: Set[str]) -> None:
    """Fill report with metadata about declared/undeclared markers."""
    used = {m for t in report.tests for m in t.markers}
    required = {m for s in REQUIRED_CATEGORIES.values() for m in s}

    report.undeclared_markers = sorted(used - declared) or None
    report.missing_in_project_config = sorted(required - declared) or None


class MarkerStrictValidator:
    """Strict validation for marker definitions and naming conventions."""

    NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")

    @staticmethod
    def validate(report: Report, declared: Set[str]) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        errors: List[str] = []

        if report.undeclared_markers:
            errors.append(
                "Undeclared markers used: " + ", ".join(report.undeclared_markers)
            )

        if report.missing_in_project_config:
            errors.append(
                "Required markers missing in pytest.ini/pyproject.toml: "
                + ", ".join(report.missing_in_project_config)
            )

        bad_names = sorted(
            m for m in declared if not MarkerStrictValidator.NAME_PATTERN.fullmatch(m)
        )
        if bad_names:
            errors.append(
                "Invalid marker names (must match [a-z0-9_]+): " + ", ".join(bad_names)
            )

        return errors


# --------------------------------------------------------------------------- #
# CLI & Runner
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(description="pytest marker validator")
    parser.add_argument("--json", help="Write JSON report to file")
    parser.add_argument(
        "--no-stub", action="store_true", help="Disable dependency stubbing"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation (undeclared markers, missing config, naming)",
    )
    parser.add_argument(
        "--tests", default="tests", help="Path to test directory (default: tests)"
    )
    return parser.parse_args()


def run_collection(test_path: str, use_stubbing: bool) -> tuple[int, Report]:
    """Run pytest collection and return exit code and report."""
    if use_stubbing:
        stubber = DependencyStubber()
        for module in STUB_MODULES:
            stubber.ensure_available(module)

        # Special case: pytest-benchmark needs a real Warning subclass
        try:
            sys.modules["pytest_benchmark.logger"].PytestBenchmarkWarning = type(  # type: ignore[attr-defined]
                "PytestBenchmarkWarning", (Warning,), {}
            )
        except (KeyError, AttributeError):
            pass

        LOG.info("Stubbed %d modules", len(stubber.stubbed))

    plugin = MarkerReportPlugin()
    exitcode = pytest.main(
        [
            "--collect-only",
            "-qq",
            "--disable-warnings",
            # Override config from pyproject.toml to avoid picking up options
            # that require plugins/modules not installed in this environment
            "-o",
            "addopts=",
            "-o",
            "filterwarnings=",
            test_path,
        ],
        plugins=[plugin],
    )
    return exitcode, plugin.build_report()


def print_human_report(report: Report) -> None:
    """Print human-readable report to stdout."""
    print("\n" + "=" * 80)
    print(f"{'TEST ID':<60} | MARKERS")
    print("=" * 80)

    for rec in report.tests:
        print(f"{rec.nodeid:<60} | {', '.join(rec.markers)}")

    # Print tests with missing markers before summary
    missing_tests = [rec for rec in report.tests if rec.missing]
    if missing_tests:
        print("\n" + "=" * 80)
        print("TESTS MISSING REQUIRED MARKERS")
        print("=" * 80)
        for rec in missing_tests:
            print(f"{rec.nodeid}")
            print(f"  Missing: {', '.join(rec.missing)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Tests checked: {report.total_checked}")
    print(f"  Mypy skipped:  {report.total_skipped_mypy}")
    print(f"  Missing sets:  {report.total_missing}")
    print("=" * 80)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    exitcode, report = run_collection(args.tests, not args.no_stub)

    # Load and validate marker definitions
    declared = load_declared_markers(Path("."))
    validate_marker_definitions(report, declared)

    print_human_report(report)

    # Strict mode validation
    if args.strict:
        strict_errors = MarkerStrictValidator.validate(report, declared)
        if strict_errors:
            for e in strict_errors:
                LOG.error("[STRICT] %s", e)
            return 1

    # Write JSON report if requested
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        LOG.info("Wrote JSON report to %s", args.json)

    # Fail if any tests are missing required markers
    return 1 if report.total_missing > 0 else exitcode


if __name__ == "__main__":
    raise SystemExit(main())
