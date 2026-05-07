#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pytest Marker Report (Production Grade)

- Collects pytest tests without executing them
- Prints markers and validates category coverage
- Auto-stubs unavailable dependencies via a sys.meta_path import hook
  (`_AutoStubFinder`) so test files import without their real third-party
  deps. First-party namespaces (`dynamo.*`, `tests.*`, `components.*`)
  are excluded from auto-stubbing so real bugs surface as ImportError.
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
from typing import Callable, Dict, List, Optional, Set

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
    "Hardware": {
        "gpu_0",
        "gpu_1",
        "gpu_2",
        "gpu_4",
        "gpu_8",
        "h100",
        "k8s",
        "xpu_1",
        "xpu_2",
    },
}

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


def _make_stub_class(name: str) -> type:
    """Permissive class usable as a base, a pydantic field type, or a callable.

    - __init__ accepts arbitrary args so `Cls(*a, **kw)` works.
    - Metaclass __getattr__ auto-creates stub-class attributes so
      `Cls.SOME_CONSTANT` and `Cls.NestedType` both work.
    - __init_subclass__ tolerates arbitrary keyword args from typing tricks.
    - __get_pydantic_core_schema__ returns any_schema for pydantic field use.
    """

    class _StubMeta(type):
        def __getattr__(cls, attr):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            sub = _make_stub_class(f"{cls.__name__}.{attr}")
            setattr(cls, attr, sub)
            return sub

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        pass

    def _init_subclass(cls, **kwargs):  # type: ignore[no-untyped-def]
        pass

    def _getattr(self, attr):  # type: ignore[no-untyped-def]
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _make_stub_class(attr)()

    def _call(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _make_stub_class("call_result")()

    def _get_schema(cls, source, handler):  # type: ignore[no-untyped-def]
        try:
            from pydantic_core import core_schema

            return core_schema.any_schema()
        except ImportError:
            return None

    return _StubMeta(
        name,
        (),
        {
            "__init__": _init,
            "__init_subclass__": classmethod(_init_subclass),
            "__getattr__": _getattr,
            "__call__": _call,
            "__get_pydantic_core_schema__": classmethod(_get_schema),
        },
    )


class _StubModule(ModuleType):
    """Module whose unknown attributes resolve to real, pydantic-friendly classes.

    Real classes (vs MagicMock) so that:
      - class Foo(stub.X): works (X is a type)
      - field: stub.X in pydantic works (X has __get_pydantic_core_schema__)
      - stub.X.attr = classmethod(...) descriptor-binds correctly
    Submodule attribute access prefers an entry already in sys.modules so
    `pkg.sub` returns the submodule instance (not a class) when both are
    present.
    """

    def __getattr__(self, name: str) -> object:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        sub_name = f"{self.__name__}.{name}"
        if sub_name in sys.modules:
            sub = sys.modules[sub_name]
            setattr(self, name, sub)
            return sub
        cls = _make_stub_class(name)
        setattr(self, name, cls)
        return cls


# First-party namespaces that must NEVER be auto-stubbed. A typo in
# `dynamo.runtime.foo` should surface as ModuleNotFoundError, not silently
# resolve to a stub. `pytest` and `_pytest.*` are excluded so pytest's own
# machinery keeps working; third-party `pytest_*` plugins (pytest_asyncio,
# pytest_benchmark, pytest_httpserver) are NOT in this list — they fall
# through to auto-stub when missing from the pre-commit env.
_NEVER_STUB_PREFIXES = ("dynamo.", "dynamo_run.", "tests.", "components.", "_pytest.")
_NEVER_STUB_EXACT: Set[str] = {"dynamo", "dynamo_run", "tests", "components", "pytest"}

# Native extensions we DO want to auto-stub even though they sit under a
# never-stub namespace (the .so isn't built in the pre-commit env).
_ALWAYS_STUB_EXACT: Set[str] = {"dynamo._core", "nixl._api"}

# Lazy specials: applied by _StubLoader.exec_module after the bare stub is
# created. Add an entry only when test code reads a module attribute that a
# bare auto-vivified stub class can't satisfy (e.g. a real Warning subclass
# for `issubclass(..., Warning)`, or a non-dunder attribute that
# _StubModule.__getattr__ would refuse to create).
_LOADER_SPECIALS: Dict[str, Callable[[ModuleType], None]] = {
    "vllm": lambda m: setattr(m, "__version__", "0.0.0"),
    "pytest_benchmark.logger": lambda m: setattr(
        m,
        "PytestBenchmarkWarning",
        type("PytestBenchmarkWarning", (Warning,), {}),
    ),
}


class _StubLoader:
    """Loader paired with `_AutoStubFinder` — instantiates `_StubModule`."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return _StubModule(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        module.__path__ = []  # type: ignore[attr-defined]
        module.__loader__ = self
        module.__package__ = (  # type: ignore[assignment]
            module.__name__.rsplit(".", 1)[0]
            if "." in module.__name__
            else module.__name__
        )
        special = _LOADER_SPECIALS.get(module.__name__)
        if special is not None:
            special(module)


class _AutoStubFinder:
    """Last-resort meta-path finder: stubs any import a real loader couldn't resolve.

    Append to the END of `sys.meta_path` so `BuiltinImporter`, `FrozenImporter`,
    and `PathFinder` all get first crack. Real installed packages win;
    auto-stubbing only fires on `ModuleNotFoundError`. First-party namespaces
    (`dynamo.*`, `tests.*`, `components.*`) are excluded so a typo or a real
    bug in our code raises instead of getting silently masked — except for
    `dynamo._core` / `nixl._api`, native extensions that intentionally fall
    through when the .so isn't built.
    """

    def __init__(self) -> None:
        self.stubbed: Set[str] = set()
        self._loader = _StubLoader()

    def _should_skip(self, fullname: str) -> bool:
        if fullname in _ALWAYS_STUB_EXACT:
            return False
        if fullname in _NEVER_STUB_EXACT:
            return True
        if fullname.startswith(_NEVER_STUB_PREFIXES):
            return True
        # CPython stdlib top-level names — never stub. Stdlib code wraps
        # platform-specific imports (`winreg`, `_winapi`, `msilib`,
        # `_posixsubprocess`, etc.) in try/except ImportError;
        # auto-stubbing makes those blocks take the wrong branch (Windows
        # code path on Linux, mimetypes registry read against a stub
        # OpenKey, etc.). `sys.stdlib_module_names` lists ALL stdlib top
        # levels regardless of platform availability — same set on every
        # platform, so it correctly identifies "this is missing because
        # stdlib expects it to be missing here."
        if fullname.split(".", 1)[0] in sys.stdlib_module_names:
            return True
        return False

    def find_spec(
        self,
        fullname: str,
        path: Optional[List[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if self._should_skip(fullname):
            return None
        # `importlib.util.find_spec(X)` is a presence check, not an import
        # request. Returning a stub here makes `if find_spec(X): use(X)`
        # patterns false-positive — caller assumes the package is installed
        # and dies later (e.g. `dash/dash.py` does this for dash_design_kit
        # and follows up with `metadata.version(...)` which fails). Only
        # respond to the real import machinery.
        if _called_from_util_find_spec():
            return None
        self.stubbed.add(fullname)
        # is_package=True populates spec.submodule_search_locations so the
        # spec carries an `__path__`; the loader resets it to [] so any
        # submodule cascades back through us instead of resolving to disk.
        return importlib.machinery.ModuleSpec(fullname, self._loader, is_package=True)


def _called_from_util_find_spec() -> bool:
    """True if the call stack contains `importlib.util.find_spec`.

    The real import machinery uses `_bootstrap._find_spec`; the public
    `importlib.util.find_spec` is the presence-check API. They share our
    `find_spec(...)` signature, so distinguishing them by stack walk is the
    only signal available.
    """
    frame = sys._getframe(2)  # skip _called_from_util_find_spec + find_spec
    while frame is not None:
        code = frame.f_code
        # Python 3.12+ freezes importlib — co_filename is `<frozen
        # importlib.util>`. Match by frozen marker as well as the source
        # path to stay portable across versions.
        if code.co_name == "find_spec" and (
            "importlib.util" in code.co_filename
            or code.co_filename.endswith(("importlib/util.py", "importlib\\util.py"))
        ):
            return True
        frame = frame.f_back
    return False


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
            if markers & {"mypy", "skip", "skipif"}:
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
        "--tests",
        nargs="*",
        default=["tests", "components/src"],
        help="Paths to test directories (default: tests components/src)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all tests with their markers (default: only failures and summary)",
    )
    return parser.parse_args()


def run_collection(test_paths: list[str], use_stubbing: bool) -> tuple[int, Report]:
    """Run pytest collection and return exit code and report."""
    auto: Optional[_AutoStubFinder] = None
    if use_stubbing:
        # Force-remove native extensions that may be partially loaded but broken.
        for mod in ("dynamo._core", "nixl._api"):
            sys.modules.pop(mod, None)

        auto = _AutoStubFinder()
        sys.meta_path.append(auto)  # END of meta_path — real finders win first

        # Default pydantic models to arbitrary_types_allowed so stubbed
        # classes used as field annotations don't blow up schema generation.
        # This patches REAL pydantic (the pre-commit env installs it); when
        # pydantic itself is auto-stubbed, _make_stub_class already produces
        # pydantic-friendly classes via __get_pydantic_core_schema__.
        try:
            import pydantic as _pydantic_root

            _pydantic_root.BaseModel.model_config = _pydantic_root.ConfigDict(  # type: ignore[assignment]
                arbitrary_types_allowed=True
            )
        except (ImportError, AttributeError):
            pass

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
            *test_paths,
        ],
        plugins=[plugin],
    )

    # Visibility: print which packages got auto-stubbed during collection.
    # Reviewers spot-check this; surprising names ("why is `pandas` here?")
    # surface immediately. Goes to stderr so --json output stays clean.
    if auto is not None and auto.stubbed:
        roots: Dict[str, int] = {}
        for name in auto.stubbed:
            root = name.split(".", 1)[0]
            roots[root] = roots.get(root, 0) + 1
        summary = ", ".join(f"{k}({v})" for k, v in sorted(roots.items()))
        print(
            f"[auto-stub] {len(auto.stubbed)} modules across "
            f"{len(roots)} packages: {summary}",
            file=sys.stderr,
        )

    return exitcode, plugin.build_report()


def print_human_report(report: Report, *, verbose: bool = False) -> None:
    """Print human-readable report to stdout.

    By default only prints tests with missing markers and the summary.
    Pass verbose=True to print all tests with their markers.
    """
    if verbose:
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

    print_human_report(report, verbose=args.verbose)

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
