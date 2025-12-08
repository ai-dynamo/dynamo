#!/usr/bin/env python3
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# --- Configuration ---
REQUIRED_CATEGORIES = {
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

# Add current directory to sys.path to ensure local modules are found
cwd = os.path.abspath(os.getcwd())
sys.path.insert(0, cwd)
sys.path.insert(0, os.path.join(cwd, "components", "src"))
sys.path.insert(0, os.path.join(cwd, "lib", "bindings", "python", "src"))


# --- Mocking Helper ---
def mock_package(name):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


# --- Mock missing modules ---
sys.modules["pytest_httpserver"] = MagicMock()
sys.modules["pytest_httpserver.HTTPServer"] = MagicMock()

# pytest_benchmark (required for parsing filterwarnings in pyproject.toml)
mock_package("pytest_benchmark")
m_pb_logger = mock_package("pytest_benchmark.logger")
# Mock the plugin module explicitly so pytest can 'load' it if it tries
mock_package("pytest_benchmark.plugin")


# Define a real class inheriting from Warning so issubclass(cat, Warning) works
class PytestBenchmarkWarning(Warning):
    pass


m_pb_logger.PytestBenchmarkWarning = PytestBenchmarkWarning

# kr8s
m_kr8s = mock_package("kr8s")
m_kr8s.objects = MagicMock()
sys.modules["kr8s.objects"] = m_kr8s.objects

# aiohttp
sys.modules["aiohttp"] = MagicMock()

# aiofiles
sys.modules["aiofiles"] = MagicMock()

# httpx
sys.modules["httpx"] = MagicMock()

# tabulate
sys.modules["tabulate"] = MagicMock()

# tritonclient
mock_package("tritonclient")
sys.modules["tritonclient.grpc"] = MagicMock()

# kubernetes
mock_package("kubernetes")
m_k8s_client = mock_package("kubernetes.client")
m_k8s_client.V1Node = MagicMock()
mock_package("kubernetes.config")
m_k8s_config_exc = mock_package("kubernetes.config.config_exception")
m_k8s_config_exc.ConfigException = MagicMock()

# kubernetes_asyncio
m_k8s_asyncio = mock_package("kubernetes_asyncio")
m_k8s_asyncio.client = MagicMock()
m_k8s_asyncio.config = MagicMock()

# prometheus_api_client
sys.modules["prometheus_api_client"] = MagicMock()

# huggingface_hub
m_hfh = mock_package("huggingface_hub")
sys.modules["huggingface_hub.model_info"] = MagicMock()

# transformers
m_transformers = mock_package("transformers")
m_transformers.AutoConfig = MagicMock()

# pandas
sys.modules["pandas"] = MagicMock()

# matplotlib
m_mpl = mock_package("matplotlib")
m_mpl.pyplot = MagicMock()
m_mpl.cm = MagicMock()
sys.modules["matplotlib.pyplot"] = m_mpl.pyplot

# pmdarima
sys.modules["pmdarima"] = MagicMock()

# prophet
sys.modules["prophet"] = MagicMock()

# scipy
m_scipy = mock_package("scipy")
m_scipy.interpolate = MagicMock()
sys.modules["scipy.interpolate"] = m_scipy.interpolate

# nats
sys.modules["nats"] = MagicMock()

# dynamo._core (binary extension)
sys.modules["dynamo._core"] = MagicMock()


class MarkerReportPlugin:
    def __init__(self):
        self.count = 0
        self.errors = []

    def pytest_collection_modifyitems(self, session, config, items):
        """
        Hook called after collection has been performed.
        We report the markers for each item here.
        """
        print(f"\nGenerating marker report for {len(items)} collected tests...\n")
        print("=" * 80)
        print(f"{'TEST ID':<60} | {'MARKERS'}")
        print("=" * 80)

        for item in items:
            # item.iter_markers() includes markers from classes, modules, etc. (inherited)
            # We sort them for consistent output
            marker_names = {m.name for m in item.iter_markers()}

            if "mypy" in marker_names:
                continue

            markers = sorted(marker_names)
            marker_str = ", ".join(markers)
            print(f"{item.nodeid:<60} | {marker_str}")
            self.count += 1

            # Check for required categories
            missing = []
            for category, allowed in REQUIRED_CATEGORIES.items():
                if not marker_names.intersection(allowed):
                    missing.append(category)

            if missing:
                self.errors.append(
                    f"FAIL: {item.nodeid}\n      Missing: {', '.join(missing)}"
                )

        print("-" * 80)

        if self.errors:
            print("\n" + "=" * 80)
            print("VALIDATION ERRORS")
            print("=" * 80)
            for err in self.errors:
                print(err)
            print("=" * 80)
            print(
                f"FAILED. Found {len(self.errors)} tests with missing required markers."
            )

    def pytest_sessionfinish(self, session, exitstatus):
        """
        Called after the whole session finishes.
        """
        print(f"\nReport generated for {self.count} tests.")
        if self.errors:
            session.exitstatus = 1
        else:
            session.exitstatus = 0


def run_report():
    # Arguments for pytest
    args = [
        "--collect-only",
        "-q",
        "--disable-warnings",
        "-W ignore::pytest.PytestAssertRewriteWarning",  # Ignore assertion rewrite warnings for mocked modules
        "tests",  # Target tests directory
    ]

    plugin = MarkerReportPlugin()

    # Run pytest programmatically
    return pytest.main(args, plugins=[plugin])


if __name__ == "__main__":
    sys.exit(run_report())
