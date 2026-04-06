# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Optional

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
from tests.utils.test_output import resolve_test_output_path

_logger = logging.getLogger(__name__)

# Typed stash keys for GPU-parallel config (avoids setting unknown attrs on Config)
_gpu_parallel_gpus_key: pytest.StashKey[list[dict]] = pytest.StashKey()
_gpu_indices_key: pytest.StashKey[list[int] | None] = pytest.StashKey()
_gpu_slots_key: pytest.StashKey[int | None] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add shared command-line options for all tests.

    Shared options that apply across multiple test suites are defined here.
    Suite-specific options (e.g., deploy, fault-tolerance) are defined in
    their respective subdirectory conftest.py files.
    """
    # -------------------------------------------------------------------------
    # Shared Deployment Options (used by multiple test suites)
    # -------------------------------------------------------------------------
    parser.addoption(
        "--image",
        type=str,
        default=None,
        help="Container image to use for deployment (overrides YAML default)",
    )
    parser.addoption(
        "--namespace",
        type=str,
        default=None,  # No default here - subdirectories provide their own
        help="Kubernetes namespace for deployment",
    )
    parser.addoption(
        "--skip-service-restart",
        action="store_true",
        default=None,  # None = use fixture's default behavior
        help="Skip restarting NATS and etcd services before deployment. "
        "Default: deploy tests skip (for speed), fault-tolerance tests restart (for clean state).",
    )
    parser.addoption(
        "--max-vram-gib",
        type=float,
        default=None,
        help="Only run tests with @pytest.mark.profiled_vram_gib(N) that fit in N GiB. "
        "Without -n: runs tests sequentially. "
        "With -n N: runs N tests concurrently as subprocesses with VRAM-aware scheduling. "
        "With -n auto: calculates max concurrent slots from GPU VRAM / max_vram_gib.",
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show which tests would run vs skip based on --max-vram-gib, then exit.",
    )


LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,  # ISO 8601 UTC format
)


# ---------------------------------------------------------------------------
# GPU-serial and GPU-parallel: VRAM-aware test scheduling
#
# Activated only when both --max-vram-gib and -n auto are passed:
#   pytest --max-vram-gib=48 -n auto -m "gpu_1 and sglang" tests/serve/
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Detect GPUs for --max-vram-gib planning and parallel execution."""
    vram_limit = config.getoption("max_vram_gib", default=None)
    if vram_limit is None:
        return
    # Delayed: vram_utils requires pynvml, otherwise conftest fails to load
    # on CPU-only CI runners (e.g. ARM deploy tests) that lack nvidia-ml-py.
    from tests.utils.pytest_parallel_gpu import _parse_cuda_visible
    from tests.utils.vram_utils import auto_worker_count, detect_gpus

    gpus = detect_gpus()
    if gpus:
        config.stash[_gpu_parallel_gpus_key] = gpus

    # Honour CUDA_VISIBLE_DEVICES to restrict which GPUs the scheduler uses.
    # NVML always sees all physical GPUs, so we filter here.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        config.stash[_gpu_indices_key] = _parse_cuda_visible(cvd, gpus)
        selected_gpus = [
            g for g in gpus if g["index"] in config.stash[_gpu_indices_key]
        ]
    else:
        config.stash[_gpu_indices_key] = None  # all GPUs
        selected_gpus = gpus

    # If -n is set with --max-vram-gib, save the slot count and disable xdist
    # so our subprocess orchestrator handles parallelism instead.
    # xdist's pytest_configure(trylast=True) checks _is_distribution_mode()
    # which reads dist/tx (not numprocesses), so we must also clear dist.
    numproc = config.getoption("numprocesses", default=None)
    if numproc is not None and numproc != 0:
        if isinstance(numproc, str) or numproc == -1:
            config.stash[_gpu_slots_key] = (
                auto_worker_count(selected_gpus, vram_limit) if selected_gpus else 1
            )
        else:
            config.stash[_gpu_slots_key] = int(numproc)
        config.option.numprocesses = 0
        config.option.dist = "no"


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session: pytest.Session) -> bool | None:
    """Intercept the test loop for GPU-parallel execution.

    When --max-vram-gib and -n are both present, run tests as independent
    subprocesses via the GPU orchestrator instead of the normal pytest loop.
    Must run before the default pytest loop (tryfirst) so we can return True
    to prevent the default sequential execution.
    """
    config = session.config
    num_slots = config.stash.get(_gpu_slots_key, None)
    vram_limit = config.getoption("max_vram_gib", default=None)

    if num_slots is None or vram_limit is None:
        return None  # serial execution: let normal pytest handle it

    # Imports related to parallel execution must be delayed. See vram_utils pynvml note in pytest_configure for the full reasons
    from tests.utils.pytest_parallel_gpu import run_parallel
    from tests.utils.vram_utils import load_test_meta

    # Collect test IDs from the already-filtered session items
    test_ids = [item.nodeid for item in session.items]
    if not test_ids:
        return True

    meta = load_test_meta()
    is_stream = config.getoption("capture", default="fd") == "no"
    gpu_indices = config.stash.get(_gpu_indices_key, None)

    # Forward original CLI args to child pytest subprocesses so they
    # inherit options like -s, -v, --tb, --durations, --image, etc.
    extra_args: list[str] = []
    if is_stream:
        extra_args.append("-s")
    verbose = config.getoption("verbose", default=0)
    if verbose >= 2:
        extra_args.append("-vv")
    elif verbose >= 1:
        extra_args.append("-v")
    tb_style = config.getoption("tbstyle", default="short")
    if tb_style and tb_style != "short":
        extra_args.append(f"--tb={tb_style}")
    durations = config.getoption("durations", default=None)
    if durations is not None:
        extra_args.append(f"--durations={durations}")
    durations_min = config.getoption("durations_min", default=None)
    if durations_min is not None:
        extra_args.append(f"--durations-min={durations_min}")
    for opt_name, cli_flag in [
        ("image", "--image"),
        ("namespace", "--namespace"),
        ("framework", "--framework"),
        ("profile", "--profile"),
    ]:
        val = config.getoption(opt_name, default=None)
        if val is not None:
            extra_args.extend([cli_flag, str(val)])
    if config.getoption("skip_service_restart", default=None):
        extra_args.append("--skip-service-restart")

    rc = run_parallel(
        test_ids=test_ids,
        meta=meta,
        max_vram_gib=vram_limit,
        num_slots=num_slots,
        gpu_indices=gpu_indices,
        extra_pytest_args=extra_args or None,
        stream=is_stream,
    )

    if rc != 0:
        session.testsfailed = 1
    return True  # we handled the test loop


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
    #
    # Also exclude gdr_copy transport to prevent GDRCopy driver initialization
    # failures (driverInitFileInfo result=11) that can abort the process when
    # the gdrdrv kernel module is not loaded.
    mp.setenv("UCX_TLS", "^mm,gdr_copy")
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
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
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
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to pre-download models for tests"
        ) from exc

    failures = []
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

        except Exception as exc:
            logging.error(f"Failed to pre-download {model_id}: {exc}")
            failures.append(f"{model_id}: {exc}")

    if failures:
        raise RuntimeError(
            "Failed to pre-download required Hugging Face models:\n"
            + "\n".join(failures)
        )


def _enable_offline_with_mistral_patch():
    """Set HF_HUB_OFFLINE=1 and work around a transformers 4.57.3 regression.

    transformers 4.57.3 (PR #42389) introduced _patch_mistral_regex which calls
    huggingface_hub.model_info() unconditionally for every tokenizer load — even
    non-Mistral models with fully cached weights. This API call fails when
    HF_HUB_OFFLINE=1.

    Since tests launch TRT-LLM workers as subprocesses that inherit env vars but
    not in-process monkey-patches, we inject the fix via a sitecustomize.py on
    PYTHONPATH so every subprocess auto-applies it at startup.

    Upstream bug: https://github.com/huggingface/transformers/issues/44843

    TODO: Remove this workaround once transformers ships a fix and TRT-LLM (or
    any other dependency) upgrades to that fixed version.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Apply the patch in this process
    try:
        from huggingface_hub.errors import OfflineModeIsEnabled
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        original = PreTrainedTokenizerBase._patch_mistral_regex

        @classmethod  # type: ignore[misc]
        def _safe_patch(cls, tokenizer, *args, **kwargs):
            try:
                return original.__func__(cls, tokenizer, *args, **kwargs)
            except OfflineModeIsEnabled:
                return tokenizer

        PreTrainedTokenizerBase._patch_mistral_regex = _safe_patch
    except (ImportError, AttributeError):
        return  # transformers version without _patch_mistral_regex — nothing to do

    # Write a sitecustomize.py so subprocesses also get the patch.
    # Use a per-worker dir under xdist to avoid write races.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    patch_dir = os.path.join(tempfile.gettempdir(), f"dynamo_test_hf_patch_{worker_id}")
    os.makedirs(patch_dir, exist_ok=True)
    with open(os.path.join(patch_dir, "sitecustomize.py"), "w") as f:
        f.write(
            "import os\n"
            "if os.environ.get('HF_HUB_OFFLINE') == '1':\n"
            "    try:\n"
            "        from transformers.tokenization_utils_base import"
            " PreTrainedTokenizerBase as _T\n"
            "        from huggingface_hub.errors import"
            " OfflineModeIsEnabled as _E\n"
            "        _orig = _T._patch_mistral_regex\n"
            "        @classmethod\n"
            "        def _safe(cls, tokenizer, *a, **kw):\n"
            "            try:\n"
            "                return _orig.__func__(cls, tokenizer, *a, **kw)\n"
            "            except _E:\n"
            "                return tokenizer\n"
            "        _T._patch_mistral_regex = _safe\n"
            "    except (ImportError, AttributeError):\n"
            "        pass\n"
        )
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{patch_dir}:{pythonpath}" if pythonpath else patch_dir
    logging.info(
        "Enabled HF_HUB_OFFLINE with _patch_mistral_regex workaround "
        "(see https://github.com/huggingface/transformers/issues/44843)"
    )


def _disable_offline_with_mistral_patch():
    """Undo _enable_offline_with_mistral_patch."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    patch_dir = os.path.join(tempfile.gettempdir(), f"dynamo_test_hf_patch_{worker_id}")
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = pythonpath.replace(f"{patch_dir}:", "").replace(
        patch_dir, ""
    )


_download_lock_path = os.path.join(tempfile.gettempdir(), "pytest_model_download.lock")


@pytest.fixture(scope="session")
def predownload_models(pytestconfig):
    """Fixture wrapper around download_models for models used in collected tests.

    Uses a file lock so that under xdist, only one worker downloads at a time
    and the rest reuse the HuggingFace cache.
    """
    models = getattr(pytestconfig, "models_to_download", None)
    with FileLock(_download_lock_path):
        if models:
            logging.info(
                f"Downloading {len(models)} models needed for collected tests\nModels: {models}"
            )
            download_models(model_list=list(models))
        else:
            download_models()

    _enable_offline_with_mistral_patch()
    yield
    _disable_offline_with_mistral_patch()


@pytest.fixture(scope="session")
def predownload_tokenizers(pytestconfig):
    """Fixture wrapper around download_models for tokenizers used in collected tests.

    Uses a file lock so that under xdist, only one worker downloads at a time.
    """
    models = getattr(pytestconfig, "models_to_download", None)
    with FileLock(_download_lock_path):
        if models:
            logging.info(
                f"Downloading tokenizers for {len(models)} models needed for collected tests\nModels: {models}"
            )
            download_models(model_list=list(models), ignore_weights=True)
        else:
            download_models(ignore_weights=True)

    _enable_offline_with_mistral_patch()
    yield
    _disable_offline_with_mistral_patch()


@pytest.fixture(autouse=True)
def logger(request):
    log_dir = resolve_test_output_path(request.node.name)
    log_path = os.path.join(log_dir, "test.log.txt")
    logger = logging.getLogger()
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    yield
    handler.close()
    logger.removeHandler(handler)


def _item_has_marker(item, marker_name):
    """Check if a test item has a marker, including module-level pytestmark."""
    if item.get_closest_marker(marker_name):
        return True
    module = getattr(item, "module", None)
    if module is not None:
        marks = getattr(module, "pytestmark", [])
        if not isinstance(marks, list):
            marks = [marks]
        if any(getattr(m, "name", "") == marker_name for m in marks):
            return True
    return False


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """
    This function is called to modify the list of tests to run.
    """
    # Auto-skip tests marked with a framework marker when the framework is not installed
    framework_markers = {
        "trtllm": "tensorrt_llm",
        "vllm": "vllm",
        "sglang": "sglang",
        "kvbm": "kvbm",
        "lmcache": "lmcache",
    }
    for marker_name, module_name in framework_markers.items():
        if importlib.util.find_spec(module_name) is None:
            skip = pytest.mark.skip(reason=f"{module_name} is not installed")
            for item in items:
                if _item_has_marker(item, marker_name):
                    item.add_marker(skip)

    # Deselect tests based on --max-vram-gib:
    #   - Tests whose profiled VRAM exceeds the limit are removed
    #   - Tests WITHOUT a VRAM marker are also removed (unknown VRAM = unsafe)
    # Using deselect (not skip) so they never reach the xdist scheduler.
    vram_limit = config.getoption("--max-vram-gib", default=None)
    if vram_limit is not None:
        keep = []
        deselected = []
        for item in items:
            vram_mark = item.get_closest_marker("profiled_vram_gib")
            if vram_mark and vram_mark.args and vram_mark.args[0] <= vram_limit:
                keep.append(item)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = keep

    # Write test metadata for the GPU orchestrator to read.
    if vram_limit is not None:
        # Delayed: see vram_utils pynvml note in pytest_configure
        from tests.utils.vram_utils import print_gpu_plan, write_test_meta

        write_test_meta(items)

    # --dry-run: print run/skip breakdown and exit without executing tests.
    # At this point, items only contains tests that passed --max-vram-gib
    # filtering (deselected items were already removed above).
    if config.getoption("--dry-run", default=False):
        would_run = []
        would_skip = []
        for item in items:
            vram_mark = item.get_closest_marker("profiled_vram_gib")
            vram_val = vram_mark.args[0] if vram_mark and vram_mark.args else None
            name = item.nodeid

            skip_reasons = []
            for marker in item.iter_markers("skip"):
                reason = marker.kwargs.get("reason", "")
                if not reason and marker.args:
                    reason = marker.args[0]
                skip_reasons.append(reason or "no reason given")

            if skip_reasons:
                would_skip.append((name, vram_val, skip_reasons))
            else:
                would_run.append((name, vram_val))

        print(f"\n{'=' * 60}")
        print(f"--max-vram-gib={vram_limit or 'not set'}  |  {len(items)} tests")
        print(f"{'=' * 60}")
        if would_run:
            print(f"\nWould RUN ({len(would_run)}):")
            for name, gib in would_run:
                gib_str = f"  ({gib} GiB)" if gib is not None else ""
                print(f"  {name}{gib_str}")
        if would_skip:
            print(f"\nWould SKIP ({len(would_skip)}):")
            for name, vram_val, reasons in would_skip:
                vram_str = f"  ({vram_val} GiB)" if vram_val is not None else ""
                print(f"  {name}{vram_str}  -- {'; '.join(reasons)}")

        gpus = config.stash.get(_gpu_parallel_gpus_key, None)
        gpu_indices = config.stash.get(_gpu_indices_key, None)
        if gpus and vram_limit is not None:
            visible = (
                [g for g in gpus if g["index"] in gpu_indices]
                if gpu_indices is not None
                else gpus
            )
            print_gpu_plan(visible, vram_limit, would_run)
        print()
        items.clear()
        return

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
        self._stop_started_processes()

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
def discovery_backend(request):
    """
    Discovery backend for runtime. Defaults to "etcd".

    To iterate over multiple backends in a test:
        @pytest.mark.parametrize("discovery_backend", ["file", "etcd"], indirect=True)
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
def durable_kv_events(request):
    """
    Whether to use durable KV events via JetStream. Defaults to False (NATS Core mode).

    When False (default):
    - NATS server starts without JetStream (-js flag omitted) for faster startup
    - Workers use local indexer mode (NATS Core / fire-and-forget events)

    When True:
    - NATS server starts with JetStream for durable KV event distribution
    - Workers use --durable-kv-events flag to publish to JetStream

    To use JetStream mode:
        @pytest.mark.parametrize("durable_kv_events", [True], indirect=True)
        def test_example(runtime_services_dynamic_ports):
            ...
    """
    return getattr(request, "param", False)


@pytest.fixture()
def runtime_services(request, discovery_backend, request_plane):
    """
    Start runtime services (NATS and/or etcd) based on discovery_backend and request_plane.

    - If discovery_backend != "etcd", etcd is not started (returns None)
    - If request_plane != "nats", NATS is not started (returns None)

    Returns a tuple of (nats_process, etcd_process) where each has a .port attribute.
    """
    # Port cleanup is now handled in NatsServer and EtcdServer __exit__ methods
    if request_plane == "nats" and discovery_backend == "etcd":
        with NatsServer(request) as nats_process:
            with EtcdServer(request) as etcd_process:
                yield nats_process, etcd_process
    elif request_plane == "nats":
        with NatsServer(request) as nats_process:
            yield nats_process, None
    elif discovery_backend == "etcd":
        with EtcdServer(request) as etcd_process:
            yield None, etcd_process
    else:
        yield None, None


@pytest.fixture()
def runtime_services_dynamic_ports(
    request, discovery_backend, request_plane, durable_kv_events
):
    """Provide NATS and Etcd servers with truly dynamic ports per test.

    This fixture actually allocates dynamic ports by passing port=0 to the servers.
    It also sets the NATS_SERVER and ETCD_ENDPOINTS environment variables so that
    Dynamo processes can find the services on the dynamic ports.

    xdist/parallel safety:
    - Function-scoped: each test gets its own NATS/etcd instances and ports.
    - Each pytest-xdist worker runs tests in a separate process, so env vars do not
      leak across workers.

    - If discovery_backend != "etcd", etcd is not started (returns None)
    - NATS is always started when etcd is used, because KV events require NATS
      regardless of the request_plane (tcp/nats only affects request transport)
    - NATS Core mode (no JetStream) is the default; JetStream is enabled when durable_kv_events=True

    Returns a tuple of (nats_process, etcd_process) where each has a .port attribute.
    """
    import os

    # Port cleanup is now handled in NatsServer and EtcdServer __exit__ methods
    # Always start NATS when etcd is used - KV events require NATS regardless of request_plane
    # When durable_kv_events=False (default), disable JetStream for faster startup
    if discovery_backend == "etcd":
        with NatsServer(
            request, port=0, disable_jetstream=not durable_kv_events
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
            request, port=0, disable_jetstream=not durable_kv_events
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
    - kv_event_port: ZMQ port for vLLM KV event publishing (avoids collisions under xdist)
    """
    frontend_port = allocate_port(DefaultPort.FRONTEND.value)
    system_port_list = allocate_ports(num_system_ports, DefaultPort.SYSTEM1.value)
    kv_event_port = allocate_port(DefaultPort.SYSTEM1.value)
    all_ports = [frontend_port, *system_port_list, kv_event_port]
    try:
        yield ServicePorts(
            frontend_port=frontend_port,
            system_ports=system_port_list,
            kv_event_port=kv_event_port,
        )
    finally:
        deallocate_ports(all_ports)
