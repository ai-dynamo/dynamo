# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from filelock import FileLock

from tests.utils.constants import TEST_MODELS
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import (
    allocate_port,
    allocate_ports,
    deallocate_port,
    deallocate_ports,
)

_logger = logging.getLogger(__name__)


def pytest_configure(config):
    # Defining markers to avoid `<marker> not found in 'markers' configuration option`
    # errors when pyproject.toml is not available in the container (e.g. some CI jobs).
    # IMPORTANT: Keep this marker list in sync with [tool.pytest.ini_options].markers
    # in pyproject.toml. If you add or remove markers there, mirror the change here.
    markers = [
        "pre_merge: marks tests to run before merging",
        "post_merge: marks tests to run after merge",
        "parallel: marks tests that can run in parallel with pytest-xdist",
        "nightly: marks tests to run nightly",
        "weekly: marks tests to run weekly",
        "gpu_0: marks tests that don't require GPU",
        "gpu_1: marks tests to run on GPU",
        "gpu_2: marks tests to run on 2GPUs",
        "gpu_4: marks tests to run on 4GPUs",
        "gpu_8: marks tests to run on 8GPUs",
        "e2e: marks tests as end-to-end tests",
        "integration: marks tests as integration tests",
        "unit: marks tests as unit tests",
        "stress: marks tests as stress tests",
        "performance: marks tests as performance tests",
        "vllm: marks tests as requiring vllm",
        "trtllm: marks tests as requiring trtllm",
        "sglang: marks tests as requiring sglang",
        "multimodal: marks tests as multimodal (image/video) tests",
        "slow: marks tests as known to be slow",
        "h100: marks tests to run on H100",
        "router: marks tests for router component",
        "planner: marks tests for planner component",
        "kvbm: marks tests for KV behavior and model determinism",
        "kvbm_v2: marks tests using KVBM V2",
        "model: model id used by a test or parameter",
        "custom_build: marks tests that require custom builds or special setup (e.g., MoE models)",
        "k8s: marks tests as requiring Kubernetes",
        "fault_tolerance: marks tests as fault tolerance tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,  # ISO 8601 UTC format
)


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


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """
    This function is called to modify the list of tests to run.
    """
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
            terminate_existing=not use_random_port,  # Disabled for parallel test execution with random ports
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
    def __init__(self, request, port=4222, timeout=300):
        # Allocate a free port if port is 0
        use_random_port = port == 0
        if use_random_port:
            # Start from 4223 (nats-server default 4222 + 1)
            port = allocate_port(4223)

        self.port = port
        self.use_random_port = use_random_port  # Track if we allocated the port
        data_dir = tempfile.mkdtemp(prefix="nats_")
        command = [
            "nats-server",
            "-js",
            "--trace",
            "--store_dir",
            data_dir,
            "-p",
            str(port),
        ]
        super().__init__(
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_existing=not use_random_port,  # Disabled for parallel test execution with random ports
            data_dir=data_dir,
            health_check_ports=[port],
            log_dir=request.node.name,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated port when server exits."""
        try:
            # Only deallocate ports that were dynamically allocated (not default ports)
            if self.use_random_port:
                deallocate_port(self.port)
        except Exception as e:
            logging.warning(f"Failed to release NatsServer port: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)


class SharedManagedProcess:
    """Base class for ManagedProcess with file-based reference counting for multi-process sharing."""

    def __init__(
        self,
        request,
        tmp_path_factory,
        resource_name: str,
        port: int,
        timeout: int = 300,
    ):
        self.request = request
        self.port = port
        self.timeout = timeout
        self.resource_name = resource_name
        self._server: Optional[ManagedProcess] = None
        self._owns_process = False

        root_tmp = Path(tempfile.gettempdir()) / "pytest_ref_counting"
        root_tmp.mkdir(parents=True, exist_ok=True)

        self.ref_file = root_tmp / f"pytest_{resource_name}_{port}_ref_count"
        self.lock_file = str(self.ref_file) + ".lock"

    def _create_server(self) -> ManagedProcess:
        """Create the underlying server instance. Must be implemented by subclasses."""
        raise NotImplementedError

    def _read_ref_count(self) -> int:
        """Read current reference count."""
        if self.ref_file.exists():
            try:
                return int(self.ref_file.read_text().strip())
            except (ValueError, IOError):
                return 0
        return 0

    def _write_ref_count(self, count: int):
        """Write reference count atomically."""
        self.ref_file.write_text(str(count))

    def _increment_ref_count(self) -> int:
        """Increment reference count and return new count."""
        count = self._read_ref_count()
        count += 1
        self._write_ref_count(count)
        return count

    def _decrement_ref_count(self) -> int:
        """Decrement reference count and return new count."""
        count = self._read_ref_count()
        count = max(0, count - 1)
        self._write_ref_count(count)
        return count

    def __enter__(self):
        with FileLock(self.lock_file):
            ref_count = self._increment_ref_count()
            if ref_count == 1:
                # First reference - start the process
                self._server = self._create_server()
                self._server.__enter__()
                self._owns_process = True
                logging.info(f"[{self.resource_name}] Started process (ref_count=1)")
            else:
                # Process already running, just track reference
                self._owns_process = False
                logging.info(
                    f"[{self.resource_name}] Reusing existing process (ref_count={ref_count})"
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with FileLock(self.lock_file):
            ref_count = self._decrement_ref_count()
            if ref_count == 0 and self._owns_process:
                # Last reference - stop the process
                if self._server:
                    self._server.__exit__(exc_type, exc_val, exc_tb)
                logging.info(f"[{self.resource_name}] Stopped process (ref_count=0)")
            elif ref_count == 0:
                # Last reference but we don't own it - shouldn't happen, but clean up ref file
                if self.ref_file.exists():
                    self.ref_file.unlink()
                logging.warning(
                    f"[{self.resource_name}] Ref count reached 0 but we don't own process"
                )
            else:
                logging.info(
                    f"[{self.resource_name}] Released reference (ref_count={ref_count})"
                )


class SharedEtcdServer(SharedManagedProcess):
    """EtcdServer with file-based reference counting for multi-process sharing."""

    def __init__(self, request, tmp_path_factory, port=2379, timeout=300):
        super().__init__(request, tmp_path_factory, "etcd", port, timeout)
        # Create a log directory for session-scoped servers
        self._log_dir = tempfile.mkdtemp(prefix=f"pytest_{self.resource_name}_logs_")

    def _create_server(self) -> ManagedProcess:
        """Create EtcdServer instance."""
        server = EtcdServer(self.request, port=self.port, timeout=self.timeout)
        # Override log_dir since request.node.name is empty in session scope
        server.log_dir = self._log_dir
        return server


class SharedNatsServer(SharedManagedProcess):
    """NatsServer with file-based reference counting for multi-process sharing."""

    def __init__(self, request, tmp_path_factory, port=4222, timeout=300):
        super().__init__(request, tmp_path_factory, "nats", port, timeout)
        # Create a log directory for session-scoped servers
        self._log_dir = tempfile.mkdtemp(prefix=f"pytest_{self.resource_name}_logs_")

    def _create_server(self) -> ManagedProcess:
        """Create NatsServer instance."""
        server = NatsServer(self.request, port=self.port, timeout=self.timeout)
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
def runtime_services_dynamic_ports(request, store_kv, request_plane):
    """Provide NATS and Etcd servers with truly dynamic ports per test.

    This fixture actually allocates dynamic ports by passing port=0 to the servers.
    It also sets the NATS_SERVER and ETCD_ENDPOINTS environment variables so that
    Dynamo processes can find the services on the dynamic ports.

    - If store_kv != "etcd", etcd is not started (returns None)
    - If request_plane != "nats", NATS is not started (returns None)

    Returns a tuple of (nats_process, etcd_process) where each has a .port attribute.
    """
    import os

    # Port cleanup is now handled in NatsServer and EtcdServer __exit__ methods
    if request_plane == "nats" and store_kv == "etcd":
        with NatsServer(request, port=0) as nats_process:
            with EtcdServer(request, port=0) as etcd_process:
                # Set environment variables for Rust/Python runtime to use. Note that xdist (parallel execution)
                # will launch isolated tests in a new process, so no need to worry about environment pollution.
                os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
                os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

                yield nats_process, etcd_process

                # No test should rely on these variables after the test, but clean up just in case.
                os.environ.pop("NATS_SERVER", None)
                os.environ.pop("ETCD_ENDPOINTS", None)
    elif request_plane == "nats":
        with NatsServer(request, port=0) as nats_process:
            os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
            yield nats_process, None
            os.environ.pop("NATS_SERVER", None)
    elif store_kv == "etcd":
        with EtcdServer(request, port=0) as etcd_process:
            os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"
            yield None, etcd_process
            os.environ.pop("ETCD_ENDPOINTS", None)
    else:
        yield None, None


@pytest.fixture(scope="session")
def runtime_services_session(request, tmp_path_factory):
    """Session-scoped fixture that provides shared NATS and etcd instances for all tests.

    Uses file-based reference counting to coordinate between pytest-xdist worker processes.
    Only the first worker starts services, and only the last worker tears them down.

    Test isolation is achieved through unique namespaces (test-namespace-{random-suffix}).
    """
    with SharedNatsServer(request, tmp_path_factory) as nats:
        with SharedEtcdServer(request, tmp_path_factory) as etcd:
            yield nats, etcd


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
