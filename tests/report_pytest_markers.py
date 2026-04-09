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
from importlib.machinery import ModuleSpec
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
    "gpu_memory_service.integrations.sglang.memory_saver",
    "gpu_memory_service.integrations.vllm",
    "gpu_memory_service.integrations.vllm.worker",
    "gpu_memory_service.server",
    "gpu_memory_service.server.allocations",
    "gpu_memory_service.server.gms",
    "gpu_memory_service.server.rpc",
    "gpu_memory_service.server.session",
    "prometheus_client",
    "prometheus_client.parser",
    "sklearn",
    "sklearn.linear_model",
    "torch",
    "torch.cuda",
    "torch.distributed",
    "torch.nn",
    "torch.nn.functional",
    "torch.multiprocessing",
    "sglang",
    "sglang.srt",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.openai",
    "sglang.srt.entrypoints.openai.protocol",
    "sglang.srt.managers",
    "sglang.srt.managers.io_struct",
    "sglang.srt.openai_api",
    "sglang.srt.openai_api.protocol",
    "sglang.srt.utils",
    "sglang.srt.utils.hf_transformers_utils",
    "vllm",
    "vllm.entrypoints",
    "vllm.entrypoints.chat_utils",
    "vllm.entrypoints.openai",
    "vllm.entrypoints.openai.protocol",
    "vllm.sampling_params",
    "pydantic",
    "pydantic.fields",
    "pydantic.functional_validators",
    "fsspec",
    "fsspec.spec",
    "msgpack",
    "sglang.srt.disaggregation",
    "sglang.srt.disaggregation.decode",
    "sglang.srt.function_call",
    "sglang.srt.function_call.function_call_parser",
    "sglang.srt.server_args",
    "vllm.config",
    "vllm.distributed",
    "vllm.distributed.parallel_state",
    "vllm.entrypoints.openai.chat_completion",
    "vllm.v1",
    "vllm.v1.metrics",
    "vllm.v1.metrics.stats",
    "aiconfigurator.generator",
    "aiconfigurator.generator.config",
    "aiconfigurator.generator.api",
    "aiconfigurator.generator.api.dynamo",
    "fsspec.implementations",
    "fsspec.implementations.local",
    "nixl",
    "sglang.srt.disaggregation.utils",
    "sglang.srt.parser",
    "sglang.srt.server_args_config_parser",
    "vllm.distributed.kv_events",
    "vllm.entrypoints.openai.chat_completion.protocol",
    "vllm.inputs",
    "vllm.v1.engine",
    "vllm.v1.engine.core",
    "vllm.v1.engine.async_llm",
    "vllm.engine",
    "vllm.engine.arg_utils",
    "vllm.entrypoints.openai.engine",
    "vllm.entrypoints.openai.engine.async_llm_engine",
    "vllm.lora",
    "vllm.lora.request",
    "nixl._api",
    "fsspec.implementations.dirfs",
    "sglang.srt.parser.reasoning_parser",
    "aiconfigurator.generator.module_bridge",
    "aiconfigurator.generator.naive",
    "safetensors",
    "safetensors.torch",
    "vllm.entrypoints.openai.engine.protocol",
    "vllm.outputs",
    "vllm.utils",
    "vllm.v1.engine.exceptions",
    "vllm.inputs.data",
    "vllm.reasoning",
    "vllm.reasoning.reasoner",
    "nixl._bindings",
    "aiconfigurator.sdk",
    "aiconfigurator.sdk.inference_client",
    "aiconfigurator.sdk.task",
    "aiconfigurator.sdk.task.config",
    "msgspec",
    "vllm.renderers",
    "vllm.renderers.base",
    "vllm.tool_parsers",
    "vllm.tool_parsers.abstract_tool_parser",
    "zmq",
    "zmq.asyncio",
    "pydantic_core",
    "sglang.srt.disaggregation.kv_events",
    "vllm.tokenizers",
    "vllm.v1.engine.input_processor",
    "pybase64",
    "typing_extensions",
    "vllm.utils.async_utils",
    "vllm.v1.engine.output_processor",
    "sglang.srt.parser.conversation",
    "vllm.logprobs",
    "vllm.logprobs.logprobs",
    "vllm.multimodal",
    "vllm.multimodal.inputs",
    "vllm.entrypoints.openai.chat_completion.serving",
    "vllm.entrypoints.openai.serving_chat",
    "vllm.entrypoints.openai.completion",
    "vllm.entrypoints.openai.completion.protocol",
    "vllm.entrypoints.openai.serving_completion",
    "vllm.entrypoints.openai.serving_models",
    "vllm.utils.system_utils",
    "blake3",
    "vllm.distributed.ec_transfer",
    "vllm.distributed.ec_transfer.ec_connector",
    "vllm.distributed.ec_transfer.ec_connector.base",
    "vllm.v1.core",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.output",
    "vllm.v1.request",
    "vllm.reasoning.qwen3_reasoning_parser",
    "vllm.reasoning.mistral_reasoning_parser",
    "vllm.tool_parsers.hermes_tool_parser",
    "vllm.tokenizers.mistral",
    "vllm.tool_parsers.mistral_tool_parser",
    "PIL",
    "PIL.Image",
    "mistral_common",
    "mistral_common.tokens",
    "mistral_common.tokens.tokenizers",
    "mistral_common.tokens.tokenizers.base",
    "vllm.v1.metrics.loggers",
    "aiconfigurator.cli",
    "aiconfigurator.cli.main",
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
        stub.__spec__ = ModuleSpec(name, None)
        stub.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        return stub

    def force_stub(self, module_name: str) -> None:
        """Force-replace a module with a stub, even if already loaded."""
        stub = self._create_module_stub(module_name)
        sys.modules[module_name] = stub
        self.stubbed.add(module_name)

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
        stubber = DependencyStubber()

        # Force-stub modules where the real .so/.pyd may be present but
        # incomplete (e.g. stale native extension missing newer symbols).
        stubber.force_stub("dynamo._core")
        stubber.force_stub("nixl._api")

        for module in STUB_MODULES:
            stubber.ensure_available(module)

        # Special case: pytest-benchmark needs a real Warning subclass
        try:
            sys.modules["pytest_benchmark.logger"].PytestBenchmarkWarning = type(  # type: ignore[attr-defined]
                "PytestBenchmarkWarning", (Warning,), {}
            )
        except (KeyError, AttributeError):
            pass

        # Special case: pydantic's BaseModel must be a real class so that
        # class Foo(BaseModel) and type annotations don't explode.
        if "pydantic" in sys.modules and "pydantic" in stubber.stubbed:
            _pydantic = sys.modules["pydantic"]

            class _FakeBaseModel:
                model_config: dict = {}

                def __init__(self, **kwargs: object) -> None:
                    for k, v in kwargs.items():
                        setattr(self, k, v)

                def __init_subclass__(cls, **kwargs: object) -> None:
                    super().__init_subclass__(**kwargs)

            _pydantic.BaseModel = _FakeBaseModel  # type: ignore[attr-defined]
            _pydantic.Field = lambda *a, **kw: None  # type: ignore[attr-defined]
            _pydantic.model_validator = lambda *a, **kw: lambda f: f  # type: ignore[attr-defined]
            _pydantic.field_validator = lambda *a, **kw: lambda f: f  # type: ignore[attr-defined]
            _pydantic.ConfigDict = lambda **kw: {}  # type: ignore[attr-defined]
            _pydantic.ValidationError = type("ValidationError", (Exception,), {})  # type: ignore[attr-defined]

        # Special case: typing_extensions must re-export real typing constructs
        # so that class Foo(TypedDict) and type annotations work.
        if "typing_extensions" in stubber.stubbed:
            import typing

            _te = sys.modules["typing_extensions"]
            for attr in (
                "TypedDict",
                "Required",
                "NotRequired",
                "Protocol",
                "runtime_checkable",
                "Annotated",
                "get_type_hints",
            ):
                setattr(_te, attr, getattr(typing, attr, lambda *a, **kw: None))

        # Special case: vllm needs __version__
        if "vllm" in sys.modules and "vllm" in stubber.stubbed:
            sys.modules["vllm"].__version__ = "0.0.0"  # type: ignore[attr-defined]

        # Special case: vllm EC connector classes are used as dataclass bases,
        # which requires real classes (MagicMock lacks __mro__).
        for mod_attr in [
            ("vllm.distributed.ec_transfer.ec_connector.base", "ECConnectorMetadata"),
            ("vllm.distributed.ec_transfer.ec_connector.base", "ECConnectorRole"),
            ("vllm.v1.core.sched.output", "SchedulerOutput"),
        ]:
            mod_name, cls_name = mod_attr
            if mod_name in stubber.stubbed:
                setattr(sys.modules[mod_name], cls_name, type(cls_name, (), {}))

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

    # Fail if any tests are missing required markers.
    # Collection errors (exitcode 2) are tolerated because different
    # environments (pre-commit, CI, local) have different dependencies
    # installed, making zero-error collection impractical to guarantee.
    if report.total_missing > 0:
        return 1
    return 0 if exitcode == 2 else exitcode


if __name__ == "__main__":
    raise SystemExit(main())
