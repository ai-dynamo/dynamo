# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Generator, Optional

import pytest
from filelock import FileLock

from tests.utils.constants import TEST_MODELS, DefaultPort
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import (
    ServicePorts,
    allocate_port,
    allocate_ports,
    deallocate_port,
    deallocate_ports,
)

_logger = logging.getLogger(__name__)


EXECUTION_CADENCE_MARKERS = [
    "pre_merge: marks tests to run before merging",
    "post_merge: marks tests to run after merge",
    "nightly: marks tests to run nightly",
    "weekly: marks tests to run weekly",
]

GPU_SCALE_MARKERS = [
    "gpu_0: marks tests that don't require GPU",
    "gpu_1: marks tests to run on GPU",
    "gpu_2: marks tests to run on 2GPUs",
    "gpu_4: marks tests to run on 4GPUs",
    "gpu_8: marks tests to run on 8GPUs",
]

TEST_SCOPE_MARKERS = [
    "e2e: marks tests as end-to-end tests",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

TEST_TYPE_MARKERS = [
    "stress: stress tests",
    "performance: performance benchmarks",
]

FRAMEWORK_MARKERS = [
    "vllm: marks tests as requiring vllm",
    "trtllm: marks tests as requiring trtllm",
    "sglang: marks tests as requiring sglang",
]

FEATURE_MARKERS = [
    "multimodal: marks tests as multimodal (image/video) tests",
    "router: marks tests for router component",
    "planner: marks tests for planner component",
    "kvbm: marks tests for KV behavior and model determinism",
    "kvbm_v2: marks tests using KVBM V2",
    "kvbm_concurrency: marks concurrency stress tests for KVBM (runs separately)",
    "fault_tolerance: marks tests as fault tolerance tests",
]


def pytest_configure(config):
    # Defining markers to avoid `<marker> not found in 'markers' configuration option`
    # errors when pyproject.toml is not available in the container (e.g. some CI jobs).
    # IMPORTANT: Keep this marker list in sync with [tool.pytest.ini_options].markers
    # in pyproject.toml. If you add or remove markers there, mirror the change here.
    markers = [
        "parallel: marks tests that can run in parallel with pytest-xdist",
        "slow: marks tests as known to be slow",
        "h100: marks tests to run on H100",
        "model: model id used by a test or parameter",
        "custom_build: marks tests that require custom builds or special setup (e.g., MoE models)",
        "k8s: marks tests as requiring Kubernetes",
        # Third-party plugin markers
        "timeout: test timeout in seconds (pytest-timeout plugin)",
    ]
    markers.extend(EXECUTION_CADENCE_MARKERS + GPU_SCALE_MARKERS + TEST_SCOPE_MARKERS + TEST_TYPE_MARKERS + FRAMEWORK_MARKERS + FEATURE_MARKERS)
    for marker in markers:
        config.addinivalue_line("markers", marker)


LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,  # ISO 8601 UTC format
)



# If True: only count tests that are marked as end-to-end.
ONLY_E2E = False

# Extract marker names without descriptions for matrix building
FRAMEWORKS = [m.split(":")[0] for m in FRAMEWORK_MARKERS]
FEATURES = [m.split(":")[0] for m in FEATURE_MARKERS]
GPU_INFRA = [m.split(":")[0] for m in GPU_SCALE_MARKERS]


# Reporting feature/framework test coverage by scanning markers
FRAMEWORK_SET = set(FRAMEWORKS)
FEATURE_SET = set(FEATURES)
GPU_INFRA_SET = set(GPU_INFRA)

# ---- Globals populated during collection ----
_agg_matrix: Dict[str, Dict[str, bool]] | None = None
_agg_tests: Dict[str, Dict[str, List[str]]] | None = None

# 3D data: gpu -> feature -> framework -> {bool, tests}
_per_gpu_matrix: Dict[str, Dict[str, Dict[str, bool]]] | None = None
_per_gpu_tests: Dict[str, Dict[str, Dict[str, List[str]]]] | None = None

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """
    This function is called to modify the list of tests to run.
    Also builds the feature X framework coverage matrix.
    """
    global _agg_matrix, _agg_tests, _per_gpu_matrix, _per_gpu_tests
    
    # Collect models via explicit pytest mark from final filtered items only
    models_to_download = set()
    for item in items:
        # Only collect from items that are not skipped
        if any(
            getattr(m, "name", "") == "skip" for m in getattr(item, "own_markers", [])
        ):
            continue
        model_mark = item.get_closest_marker("model")
        if model_mark and model_mark.args:
            models_to_download.add(model_mark.args[0])

    # Store models to download in pytest config for fixtures to access
    if models_to_download:
        config.models_to_download = models_to_download
    
    # Build feature×framework coverage matrix
    agg_matrix: Dict[str, Dict[str, bool]] = {
        feat: {fw: False for fw in FRAMEWORKS} for feat in FEATURES
    }
    agg_tests: Dict[str, Dict[str, List[str]]] = {
        feat: {fw: [] for fw in FRAMEWORKS} for feat in FEATURES
    }

    per_gpu_matrix: Dict[str, Dict[str, Dict[str, bool]]] = {
        gpu: {feat: {fw: False for fw in FRAMEWORKS} for feat in FEATURES}
        for gpu in GPU_INFRA
    }
    per_gpu_tests: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        gpu: {feat: {fw: [] for fw in FRAMEWORKS} for feat in FEATURES}
        for gpu in GPU_INFRA
    }

    for item in items:
        marker_names = {m.name for m in item.iter_markers()}

        if ONLY_E2E and "e2e" not in marker_names:
            continue

        item_frameworks = marker_names & FRAMEWORK_SET
        item_features = marker_names & FEATURE_SET
        item_gpus = marker_names & GPU_INFRA_SET

        if not item_frameworks or not item_features:
            continue

        nodeid = item.nodeid

        for feat in item_features:
            for fw in item_frameworks:
                agg_matrix[feat][fw] = True
                agg_tests[feat][fw].append(nodeid)

        for gpu in item_gpus:
            for feat in item_features:
                for fw in item_frameworks:
                    per_gpu_matrix[gpu][feat][fw] = True
                    per_gpu_tests[gpu][feat][fw].append(nodeid)

    _agg_matrix = agg_matrix
    _agg_tests = agg_tests
    _per_gpu_matrix = per_gpu_matrix
    _per_gpu_tests = per_gpu_tests


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    CLI options to control where we write machine-readable coverage reports.
    """
    group = parser.getgroup("feature-matrix")

    group.addoption(
        "--feature-matrix-json",
        action="store",
        dest="feature_matrix_json",
        default=None,
        help="Path to write feature×framework(+gpu) coverage matrix (JSON).",
    )

    group.addoption(
        "--feature-matrix-md",
        action="store",
        dest="feature_matrix_md",
        default=None,
        help="Path to write aggregated feature×framework coverage matrix (Markdown).",
    )

def _render_terminal_matrix(tr, matrix: Dict[str, Dict[str, bool]]) -> None:
    tr.write_line("")
    tr.write_line("Feature × Framework test coverage (aggregated across gpu_*):")
    tr.write_line("")

    header = ["Feature"] + FRAMEWORKS
    col_widths = [max(len(h), 14) for h in header]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    tr.write_line(fmt_row(header))
    tr.write_line(fmt_row(["-" * w for w in col_widths]))

    for feat in FEATURES:
        row = [feat]
        for fw in FRAMEWORKS:
            row.append("✔" if matrix[feat][fw] else "·")
        tr.write_line(fmt_row(row))

    tr.write_line("")

def _write_json_report(
    path: str | None,
    agg_matrix: Dict[str, Dict[str, bool]],
    agg_tests: Dict[str, Dict[str, List[str]]],
    per_gpu_matrix: Dict[str, Dict[str, Dict[str, bool]]],
    per_gpu_tests: Dict[str, Dict[str, Dict[str, List[str]]]],
) -> None:
    if not path:
        return

    payload = {
        "dimensions": {
            "frameworks": FRAMEWORKS,
            "features": FEATURES,
            "gpu": GPU_INFRA,
        },
        "aggregated": {
            feat: {
                fw: {
                    "has_test": agg_matrix[feat][fw],
                    "tests": agg_tests[feat][fw],
                }
                for fw in FRAMEWORKS
            }
            for feat in FEATURES
        },
        "by_gpu": {
            gpu: {
                feat: {
                    fw: {
                        "has_test": per_gpu_matrix[gpu][feat][fw],
                        "tests": per_gpu_tests[gpu][feat][fw],
                    }
                    for fw in FRAMEWORKS
                }
                for feat in FEATURES
            }
            for gpu in GPU_INFRA
        },
    }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _write_markdown_report(
    path: str | None,
    agg_matrix: Dict[str, Dict[str, bool]],
) -> None:
    if not path:
        return

    lines: list[str] = []
    lines.append("| Feature | " + " | ".join(FRAMEWORKS) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(FRAMEWORKS))) + "|")

    for feat in FEATURES:
        row = [feat]
        for fw in FRAMEWORKS:
            row.append("✅" if agg_matrix[feat][fw] else "❌")
        lines.append("| " + " | ".join(row) + " |")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    global _agg_matrix, _agg_tests, _per_gpu_matrix, _per_gpu_tests

    if (
        _agg_matrix is None
        or _agg_tests is None
        or _per_gpu_matrix is None
        or _per_gpu_tests is None
    ):
        return

    # Human-readable aggregated matrix
    _render_terminal_matrix(terminalreporter, _agg_matrix)

    # Machine-readable artifacts
    json_path = config.getoption("feature_matrix_json")
    md_path = config.getoption("feature_matrix_md")

    _write_json_report(
        json_path,
        _agg_matrix,
        _agg_tests,
        _per_gpu_matrix,
        _per_gpu_tests,
    )
    _write_markdown_report(md_path, _agg_matrix)



@pytest.fixture()
def set_ucx_tls_no_mm():
    """Set UCX env defaults for all tests."""
    mp = pytest.MonkeyPatch()
    # CI note:
    # - Affected test: tests/fault_tolerance/cancellation/test_vllm.py::test_request_cancellation_vllm_decode_cancel
    # - Symptom on L40 CI: UCX/NIXL mm transport assertion during worker init
    #   (uct_mem.c:482: mem.memh != UCT_MEM_HANDLE_NULL) when two workers
    #   start on the same node (maybe a shared-memory segment collision/limits).
    # - Mitigation: disable UCX "mm" shared-memory transport globally for tests
    mp.setenv("UCX_TLS", "^mm")
    yield
    mp.undo()


def download_models(model_list=None, ignore_weights=False):
    """Download models - can be called directly or via fixture

    Args:
        model_list: List of model IDs to download. If None, downloads TEST_MODELS.
        ignore_weights: If True, skips downloading model weight files. Default is False.
    """
    if model_list is None:
        model_list = TEST_MODELS

    # Check for HF_TOKEN in environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logging.info("HF_TOKEN found in environment")
    else:
        logging.warning(
            "HF_TOKEN not found in environment. "
            "Some models may fail to download or you may encounter rate limits. "
            "Get a token from https://huggingface.co/settings/tokens"
        )

    try:
        from huggingface_hub import snapshot_download

        for model_id in model_list:
            logging.info(
                f"Pre-downloading {'model (no weights)' if ignore_weights else 'model'}: {model_id}"
            )

            try:
                if ignore_weights:
                    # Weight file patterns to exclude (based on hub.rs implementation)
                    weight_patterns = [
                        "*.bin",
                        "*.safetensors",
                        "*.h5",
                        "*.msgpack",
                        "*.ckpt.index",
                    ]

                    # Download everything except weight files
                    snapshot_download(
                        repo_id=model_id,
                        token=hf_token,
                        ignore_patterns=weight_patterns,
                    )
                else:
                    # Download the full model snapshot (includes all files)
                    snapshot_download(
                        repo_id=model_id,
                        token=hf_token,
                    )
                logging.info(f"Successfully pre-downloaded: {model_id}")

            except Exception as e:
                logging.error(f"Failed to pre-download {model_id}: {e}")
                # Don't fail the fixture - let individual tests handle missing models

    except ImportError:
        logging.warning(
            "huggingface_hub not installed. "
            "Models will be downloaded during test execution."
        )


@pytest.fixture(scope="session")
def predownload_models(pytestconfig):
    """Fixture wrapper around download_models for models used in collected tests"""
    # Get models from pytest config if available, otherwise fall back to TEST_MODELS
    models = getattr(pytestconfig, "models_to_download", None)
    if models:
        logging.info(
            f"Downloading {len(models)} models needed for collected tests\nModels: {models}"
        )
        download_models(model_list=list(models))
    else:
        # Fallback to original behavior if extraction failed
        download_models()
    yield


@pytest.fixture(scope="session")
def predownload_tokenizers(pytestconfig):
    """Fixture wrapper around download_models for tokenizers used in collected tests"""
    # Get models from pytest config if available, otherwise fall back to TEST_MODELS
    models = getattr(pytestconfig, "models_to_download", None)
    if models:
        logging.info(
            f"Downloading tokenizers for {len(models)} models needed for collected tests\nModels: {models}"
        )
        download_models(model_list=list(models), ignore_weights=True)
    else:
        # Fallback to original behavior if extraction failed
        download_models(ignore_weights=True)
    yield


@pytest.fixture(autouse=True)
def logger(request):
    log_path = os.path.join(request.node.name, "test.log.txt")
    logger = logging.getLogger()
    shutil.rmtree(request.node.name, ignore_errors=True)
    os.makedirs(request.node.name, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    yield
    handler.close()
    logger.removeHandler(handler)


class EtcdServer(ManagedProcess):
    def __init__(self, request, port=2379, timeout=300):
        # Allocate free ports if port is 0
        use_random_port = port == 0
        if use_random_port:
            # Need two ports: client port and peer port for parallel execution
            # Start from 2380 (etcd default 2379 + 1)
            port, peer_port = allocate_ports(2, 2380)
        else:
            peer_port = None

        self.port = port
        self.peer_port = peer_port  # Store for cleanup
        self.use_random_port = use_random_port  # Track if we allocated the port
        port_string = str(port)
        etcd_env = os.environ.copy()
        etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
        data_dir = tempfile.mkdtemp(prefix="etcd_")

        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--advertise-client-urls",
            f"http://0.0.0.0:{port_string}",
        ]

        # Add peer port configuration only for random ports (parallel execution)
        if peer_port is not None:
            peer_port_string = str(peer_port)
            command.extend(
                [
                    "--listen-peer-urls",
                    f"http://0.0.0.0:{peer_port_string}",
                    "--initial-advertise-peer-urls",
                    f"http://localhost:{peer_port_string}",
                    "--initial-cluster",
                    f"default=http://localhost:{peer_port_string}",
                ]
            )

        command.extend(
            [
                "--data-dir",
                data_dir,
            ]
        )
        super().__init__(
            env=etcd_env,
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_all_matching_process_names=not use_random_port,  # For distributed tests, do not terminate all matching processes
            health_check_ports=[port],
            data_dir=data_dir,
            log_dir=request.node.name,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated ports when server exits."""
        try:
            # Only deallocate ports that were dynamically allocated (not default ports)
            if self.use_random_port:
                ports_to_release = [self.port]
                if self.peer_port is not None:
                    ports_to_release.append(self.peer_port)
                deallocate_ports(ports_to_release)
        except Exception as e:
            logging.warning(f"Failed to release EtcdServer port: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)


class NatsServer(ManagedProcess):
    def __init__(self, request, port=4222, timeout=300, disable_jetstream=False):
        # Allocate a free port if port is 0
        use_random_port = port == 0
        if use_random_port:
            # Start from 4223 (nats-server default 4222 + 1)
            port = allocate_port(4223)

        self.port = port
        self.use_random_port = use_random_port  # Track if we allocated the port
        self._request = request  # Store for restart
        self._timeout = timeout
        self._disable_jetstream = disable_jetstream
        data_dir = tempfile.mkdtemp(prefix="nats_") if not disable_jetstream else None
        command = [
            "nats-server",
            "--trace",
            "-p",
            str(port),
        ]
        if not disable_jetstream and data_dir:
            command.extend(["-js", "--store_dir", data_dir])
        super().__init__(
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_all_matching_process_names=not use_random_port,  # For distributed tests, do not terminate all matching processes
            data_dir=data_dir,
            health_check_ports=[port],
            health_check_funcs=[self._nats_ready],
            log_dir=request.node.name,
        )

    def _nats_ready(self, timeout: float = 5) -> bool:
        """Verify NATS server is ready by connecting and optionally checking JetStream."""
        import asyncio

        import nats

        async def check():
            try:
                nc = await nats.connect(
                    f"nats://localhost:{self.port}",
                    connect_timeout=min(timeout, 2),
                )
                try:
                    if not self._disable_jetstream:
                        # Verify JetStream is initialized
                        js = nc.jetstream()
                        await js.account_info()
                    return True
                finally:
                    await nc.close()
            except Exception:
                return False

        # Handle both sync and async contexts
        try:
            asyncio.get_running_loop()  # Check if we're in async context
            # Already in async context - run in a thread to avoid blocking
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, check()).result(timeout=timeout)
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(check())

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated port when server exits."""
        try:
            # Only deallocate ports that were dynamically allocated (not default ports)
            if self.use_random_port:
                deallocate_port(self.port)
        except Exception as e:
            logging.warning(f"Failed to release NatsServer port: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)

    def stop(self):
        """Stop the NATS server for restart. Does not release port or clean up fully."""
        _logger.info(f"Stopping NATS server on port {self.port}")
        self._terminate_process_group()
        proc = self.proc  # type: ignore[has-type]
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except Exception as e:
                _logger.warning(f"Error waiting for NATS process to stop: {e}")
            self.proc = None

    def start(self):
        """Restart a stopped NATS server with fresh state."""
        _logger.info(f"Starting NATS server on port {self.port} with fresh state")
        # Clean up old data directory and create fresh one (only if JetStream enabled)
        if not self._disable_jetstream:
            old_data_dir = self.data_dir  # type: ignore[has-type]
            if old_data_dir is not None:
                shutil.rmtree(old_data_dir, ignore_errors=True)
            self.data_dir = tempfile.mkdtemp(prefix="nats_")

        # Rebuild command
        self.command = [
            "nats-server",
            "--trace",
            "-p",
            str(self.port),
        ]
        if not self._disable_jetstream and self.data_dir:
            self.command.extend(["-js", "--store_dir", self.data_dir])

        self._start_process()
        elapsed = self._check_ports(self._timeout)
        self._check_funcs(self._timeout - elapsed)


class SharedManagedProcess:
    """Base class for persistent shared processes across pytest-xdist workers.

    Simplified design: first worker starts the process on a dynamic port, it lives forever
    (until the container dies). No ref counting, no teardown. Subsequent workers just
    reuse via port check. This eliminates race conditions and simplifies the logic.
    """

    def __init__(
        self,
        request,
        tmp_path_factory,
        resource_name: str,
        start_port: int,
        timeout: int = 300,
    ):
        self.request = request
        self.start_port = start_port
        self.port: Optional[int] = None  # Set when entering context
        self.timeout = timeout
        self.resource_name = resource_name
        self._server: Optional[ManagedProcess] = None

        root_tmp = Path(tempfile.gettempdir()) / "pytest_shared_services"
        root_tmp.mkdir(parents=True, exist_ok=True)

        self.port_file = root_tmp / f"{resource_name}_port"
        self.lock_file = str(self.port_file) + ".lock"

    def _create_server(self, port: int) -> ManagedProcess:
        """Create the underlying server instance. Must be implemented by subclasses."""
        raise NotImplementedError

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use (i.e., a process is listening on it)."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0  # 0 means connection succeeded (port in use)
        except Exception:
            return False

    def _read_port(self) -> Optional[int]:
        """Read stored port from file."""
        if self.port_file.exists():
            try:
                return int(self.port_file.read_text().strip())
            except (ValueError, IOError):
                return None
        return None

    def _write_port(self, port: int):
        """Write port to file."""
        self.port_file.write_text(str(port))

    def __enter__(self):
        with FileLock(self.lock_file):
            stored_port = self._read_port()

            # Check if a process is already running on the stored port
            if stored_port is not None and self._is_port_in_use(stored_port):
                # Reuse existing process
                self.port = stored_port
                logging.info(
                    f"[{self.resource_name}] Reusing existing process on port {self.port}"
                )
            else:
                # Start new process
                if stored_port is not None:
                    logging.warning(
                        f"[{self.resource_name}] Stale port file: port {stored_port} not in use, starting fresh"
                    )
                self.port = allocate_port(self.start_port)
                self._write_port(self.port)
                self._server = self._create_server(self.port)
                self._server.__enter__()
                logging.info(
                    f"[{self.resource_name}] Started process on port {self.port}"
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Never tear down - let the process live until the container dies.
        # This avoids race conditions and simplifies the logic.
        pass


class SharedEtcdServer(SharedManagedProcess):
    """EtcdServer with file-based reference counting for multi-process sharing."""

    def __init__(self, request, tmp_path_factory, start_port=2380, timeout=300):
        super().__init__(request, tmp_path_factory, "etcd", start_port, timeout)
        # Create a log directory for session-scoped servers
        self._log_dir = tempfile.mkdtemp(prefix=f"pytest_{self.resource_name}_logs_")

    def _create_server(self, port: int) -> ManagedProcess:
        """Create EtcdServer instance."""
        server = EtcdServer(self.request, port=port, timeout=self.timeout)
        # Override log_dir since request.node.name is empty in session scope
        server.log_dir = self._log_dir
        return server


class SharedNatsServer(SharedManagedProcess):
    """NatsServer with file-based reference counting for multi-process sharing."""

    def __init__(
        self,
        request,
        tmp_path_factory,
        start_port=4223,
        timeout=300,
        disable_jetstream=False,
    ):
        super().__init__(request, tmp_path_factory, "nats", start_port, timeout)
        # Create a log directory for session-scoped servers
        self._log_dir = tempfile.mkdtemp(prefix=f"pytest_{self.resource_name}_logs_")
        self._disable_jetstream = disable_jetstream

    def _create_server(self, port: int) -> ManagedProcess:
        """Create NatsServer instance."""
        server = NatsServer(
            self.request,
            port=port,
            timeout=self.timeout,
            disable_jetstream=self._disable_jetstream,
        )
        # Override log_dir since request.node.name is empty in session scope
        server.log_dir = self._log_dir
        return server


@pytest.fixture
def store_kv(request):
    """
    KV store for runtime. Defaults to "etcd".

    To iterate over multiple stores in a test:
        @pytest.mark.parametrize("store_kv", ["file", "etcd"], indirect=True)
        def test_example(runtime_services):
            ...
    """
    return getattr(request, "param", "etcd")


@pytest.fixture
def request_plane(request):
    """
    Request plane for runtime. Defaults to "nats".

    To iterate over multiple transports in a test:
        @pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
        def test_example(runtime_services):
            ...
    """
    return getattr(request, "param", "nats")


@pytest.fixture
def use_nats_core(request):
    """
    Whether to use NATS Core mode (local indexer) instead of JetStream. Defaults to False.

    When True:
    - NATS server starts without JetStream (-js flag omitted) for faster startup
    - Tests should use enable_local_indexer=True in mocker_args

    When False (default):
    - NATS server starts with JetStream for KV event distribution
    - Tests use JetStream-based indexer synchronization

    To use NATS Core mode:
        @pytest.mark.parametrize("use_nats_core", [True], indirect=True)
        def test_example(runtime_services_dynamic_ports):
            ...
    """
    return getattr(request, "param", False)


@pytest.fixture()
def runtime_services(request, store_kv, request_plane):
    """
    Start runtime services (NATS and/or etcd) based on store_kv and request_plane.

    - If store_kv != "etcd", etcd is not started (returns None)
    - If request_plane != "nats", NATS is not started (returns None)

    Returns a tuple of (nats_process, etcd_process) where each has a .port attribute.
    """
    # Port cleanup is now handled in NatsServer and EtcdServer __exit__ methods
    if request_plane == "nats" and store_kv == "etcd":
        with NatsServer(request) as nats_process:
            with EtcdServer(request) as etcd_process:
                yield nats_process, etcd_process
    elif request_plane == "nats":
        with NatsServer(request) as nats_process:
            yield nats_process, None
    elif store_kv == "etcd":
        with EtcdServer(request) as etcd_process:
            yield None, etcd_process
    else:
        yield None, None


@pytest.fixture()
def runtime_services_dynamic_ports(request, store_kv, request_plane, use_nats_core):
    """Provide NATS and Etcd servers with truly dynamic ports per test.

    This fixture actually allocates dynamic ports by passing port=0 to the servers.
    It also sets the NATS_SERVER and ETCD_ENDPOINTS environment variables so that
    Dynamo processes can find the services on the dynamic ports.

    xdist/parallel safety:
    - Function-scoped: each test gets its own NATS/etcd instances and ports.
    - Each pytest-xdist worker runs tests in a separate process, so env vars do not
      leak across workers.

    - If store_kv != "etcd", etcd is not started (returns None)
    - NATS is always started when etcd is used, because KV events require NATS
      regardless of the request_plane (tcp/nats only affects request transport)
    - JetStream is enabled by default; disabled when use_nats_core=True for faster startup

    Returns a tuple of (nats_process, etcd_process) where each has a .port attribute.
    """
    import os

    # Port cleanup is now handled in NatsServer and EtcdServer __exit__ methods
    # Always start NATS when etcd is used - KV events require NATS regardless of request_plane
    # When use_nats_core=True, disable JetStream for faster startup
    if store_kv == "etcd":
        with NatsServer(
            request, port=0, disable_jetstream=use_nats_core
        ) as nats_process:
            with EtcdServer(request, port=0) as etcd_process:
                # Save original env vars (may be set by session-scoped fixture)
                orig_nats = os.environ.get("NATS_SERVER")
                orig_etcd = os.environ.get("ETCD_ENDPOINTS")

                # Set environment variables for this test's dynamic ports
                os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
                os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

                yield nats_process, etcd_process

                # Restore original env vars (or remove if they weren't set)
                if orig_nats is not None:
                    os.environ["NATS_SERVER"] = orig_nats
                else:
                    os.environ.pop("NATS_SERVER", None)
                if orig_etcd is not None:
                    os.environ["ETCD_ENDPOINTS"] = orig_etcd
                else:
                    os.environ.pop("ETCD_ENDPOINTS", None)
    elif request_plane == "nats":
        with NatsServer(
            request, port=0, disable_jetstream=use_nats_core
        ) as nats_process:
            orig_nats = os.environ.get("NATS_SERVER")
            os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
            yield nats_process, None
            if orig_nats is not None:
                os.environ["NATS_SERVER"] = orig_nats
            else:
                os.environ.pop("NATS_SERVER", None)
    else:
        yield None, None


@pytest.fixture(scope="session")
def runtime_services_session(request, tmp_path_factory):
    """Session-scoped fixture that provides shared NATS and etcd instances for all tests.

    Uses file locking to coordinate between pytest-xdist worker processes.
    First worker starts services on dynamic ports, subsequent workers reuse them.
    Services are never torn down (live until container dies) to avoid race conditions.

    This fixture is xdist-safe when tests use unique namespaces (e.g. random suffixes)
    and do not assume exclusive access to global streams/keys.

    For tests that need to restart NATS (e.g. indexer sync), use `runtime_services_dynamic_ports`
    which provides per-test isolated instances.
    """
    with SharedNatsServer(request, tmp_path_factory) as nats:
        with SharedEtcdServer(request, tmp_path_factory) as etcd:
            # Set environment variables for Rust/Python runtime to use
            os.environ["NATS_SERVER"] = f"nats://localhost:{nats.port}"
            os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd.port}"

            yield nats, etcd

            # Clean up environment variables
            os.environ.pop("NATS_SERVER", None)
            os.environ.pop("ETCD_ENDPOINTS", None)


@pytest.fixture
def file_storage_backend():
    """Fixture that sets up and tears down file storage backend.

    Creates a temporary directory for file-based KV storage and sets
    the DYN_FILE_KV environment variable. Cleans up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        old_env = os.environ.get("DYN_FILE_KV")
        os.environ["DYN_FILE_KV"] = tmpdir
        logging.info(f"Set up file storage backend in: {tmpdir}")
        yield tmpdir
        # Cleanup
        if old_env is not None:
            os.environ["DYN_FILE_KV"] = old_env
        else:
            os.environ.pop("DYN_FILE_KV", None)


########################################################
# Shared Port Allocation (Dynamo deployments)
########################################################


@pytest.fixture(scope="function")
def num_system_ports(request) -> int:
    """Number of system ports to allocate for this test.

    Default: 1 port.

    Tests that need multiple system ports (e.g. SYSTEM_PORT1 + SYSTEM_PORT2) must
    explicitly request them via indirect parametrization:
      @pytest.mark.parametrize("num_system_ports", [2], indirect=True)
    """
    return getattr(request, "param", 1)


@pytest.fixture(scope="function")
def dynamo_dynamic_ports(num_system_ports) -> Generator[ServicePorts, None, None]:
    """Allocate per-test ports for Dynamo deployments.

    - frontend_port: OpenAI-compatible HTTP/gRPC ingress (dynamo.frontend)
    - system_ports: List of worker metrics/system ports (configurable count via num_system_ports)
    """
    frontend_port = allocate_port(DefaultPort.FRONTEND.value)
    system_port_list = allocate_ports(num_system_ports, DefaultPort.SYSTEM1.value)
    all_ports = [frontend_port, *system_port_list]
    try:
        yield ServicePorts(frontend_port=frontend_port, system_ports=system_port_list)
    finally:
        deallocate_ports(all_ports)
