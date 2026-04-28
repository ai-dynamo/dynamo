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
    "transformers.models",
    "transformers.models.qwen2_vl",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl",
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
    "pynvml",
    "gpu_memory_service",
    "gpu_memory_service.client",
    "gpu_memory_service.client.memory_manager",
    "gpu_memory_service.client.rpc",
    "gpu_memory_service.client.session",
    "gpu_memory_service.client.torch",
    "gpu_memory_service.client.torch.allocator",
    "gpu_memory_service.client.torch.module",
    "gpu_memory_service.client.torch.tensor",
    "gpu_memory_service.common",
    "gpu_memory_service.common.locks",
    "gpu_memory_service.common.cuda_utils",
    "gpu_memory_service.common.protocol",
    "gpu_memory_service.common.protocol.messages",
    "gpu_memory_service.common.protocol.wire",
    "gpu_memory_service.common.types",
    "gpu_memory_service.common.utils",
    "gpu_memory_service.failover_lock",
    "gpu_memory_service.failover_lock.flock",
    "gpu_memory_service.integrations",
    "gpu_memory_service.integrations.common",
    "gpu_memory_service.integrations.common.utils",
    "gpu_memory_service.integrations.sglang",
    "gpu_memory_service.integrations.sglang.patches",
    "gpu_memory_service.integrations.sglang.memory_saver",
    "gpu_memory_service.integrations.vllm",
    "gpu_memory_service.integrations.vllm.worker",
    "gpu_memory_service.server",
    "gpu_memory_service.server.allocations",
    "gpu_memory_service.server.fsm",
    "gpu_memory_service.server.gms",
    "gpu_memory_service.server.rpc",
    "gpu_memory_service.server.session",
    "prometheus_client",
    "prometheus_client.parser",
    "sklearn",
    "sklearn.linear_model",
    "torch",
    "PIL",
    "PIL.Image",
    "fsspec",
    "fsspec.implementations",
    "fsspec.implementations.dirfs",
    "sglang",
    "sglang.srt",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.openai",
    "sglang.srt.entrypoints.openai.protocol",
    "sglang.srt.function_call",
    "sglang.srt.function_call.core_types",
    "sglang.srt.function_call.function_call_parser",
    "sglang.srt.function_call.json_array_parser",
    "sglang.srt.function_call.utils",
    "sglang.srt.parser",
    "sglang.srt.parser.conversation",
    "sglang.srt.parser.reasoning_parser",
    "sglang.srt.utils",
    "sglang.srt.utils.hf_transformers_utils",
    "sglang.srt.disaggregation",
    "sglang.srt.disaggregation.kv_events",
    "sglang.srt.disaggregation.utils",
    "sglang.srt.server_args",
    "sglang.srt.server_args_config_parser",
    "vllm",
    "vllm.config",
    "vllm.distributed",
    "vllm.distributed.ec_transfer",
    "vllm.distributed.ec_transfer.ec_connector",
    "vllm.distributed.ec_transfer.ec_connector.base",
    "vllm.distributed.kv_events",
    "vllm.engine",
    "vllm.engine.arg_utils",
    "vllm.entrypoints",
    "vllm.entrypoints.chat_utils",
    "vllm.entrypoints.openai",
    "vllm.entrypoints.openai.chat_completion",
    "vllm.entrypoints.openai.chat_completion.protocol",
    "vllm.entrypoints.openai.engine",
    "vllm.entrypoints.openai.engine.protocol",
    "vllm.inputs",
    "vllm.logprobs",
    "vllm.lora",
    "vllm.lora.request",
    "vllm.multimodal",
    "vllm.multimodal.inputs",
    "vllm.outputs",
    "vllm.reasoning",
    "vllm.reasoning.mistral_reasoning_parser",
    "vllm.reasoning.qwen3_reasoning_parser",
    "vllm.renderers",
    "vllm.renderers.embed_utils",
    "vllm.sampling_params",
    "vllm.tokenizers",
    "vllm.tokenizers.mistral",
    "vllm.tool_parsers",
    "vllm.tool_parsers.hermes_tool_parser",
    "vllm.tool_parsers.mistral_tool_parser",
    "vllm.utils",
    "vllm.utils.async_utils",
    "vllm.utils.hashing",
    "vllm.utils.system_utils",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.async_scheduler",
    "vllm.v1.core.sched.output",
    "vllm.v1.engine",
    "vllm.v1.engine.async_llm",
    "vllm.v1.engine.exceptions",
    "vllm.v1.engine.input_processor",
    "vllm.v1.engine.output_processor",
    "vllm.v1.metrics",
    "vllm.v1.metrics.loggers",
    "vllm.v1.metrics.stats",
    "vllm.v1.request",
    "msgspec",
    "msgspec.structs",
    "mistral_common",
    "mistral_common.tokens",
    "mistral_common.tokens.tokenizers",
    "mistral_common.tokens.tokenizers.base",
    "safetensors",
    "nixl",
    "nixl._api",
    "nixl._bindings",
    "aiohttp.web",
    "aiconfigurator.sdk",
    "aiconfigurator.sdk.task",
    "plotly",
    "plotly.graph_objects",
    "plotly.subplots",
    "pybase64",
    "zmq",
    "zmq.asyncio",
    "blake3",
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


class DependencyStubber:
    """Stub unavailable modules to allow test collection without real dependencies."""

    def __init__(self):
        self.stubbed: Set[str] = set()

    def _create_module_stub(self, name: str) -> ModuleType:
        """Create a stub module with proper Python module attributes."""
        stub = _StubModule(name)
        stub.__path__ = []  # type: ignore[attr-defined]
        stub.__loader__ = None
        # Real ModuleSpec so importlib.util.find_spec() doesn't raise
        # ValueError when callers introspect the stub.
        stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
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
                pkg = _StubModule(sub)
                pkg.__path__ = []  # type: ignore[attr-defined]
                pkg.__spec__ = importlib.machinery.ModuleSpec(sub, loader=None)
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
    if use_stubbing:
        # Force-remove native extensions that may be partially loaded but broken.
        for mod in ("dynamo._core", "nixl._api"):
            sys.modules.pop(mod, None)

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

        # Default pydantic models to arbitrary_types_allowed so stubbed
        # classes used as field annotations don't blow up schema generation.
        try:
            import pydantic as _pydantic_root

            _pydantic_root.BaseModel.model_config = _pydantic_root.ConfigDict(  # type: ignore[assignment]
                arbitrary_types_allowed=True
            )
        except (ImportError, AttributeError):
            pass

        # Some dynamo code reads `vllm.__version__`. Stubs strip dunders;
        # set the attribute explicitly so `from vllm import __version__` works.
        if "vllm" in sys.modules and "vllm" in stubber.stubbed:
            sys.modules["vllm"].__version__ = "0.0.0"  # type: ignore[attr-defined]

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
            *test_paths,
        ],
        plugins=[plugin],
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
