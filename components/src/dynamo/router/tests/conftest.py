# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import tempfile

import pytest

from dynamo.router.tests.support.constants import TEST_MODELS
from dynamo.router.tests.support.managed_process import ManagedProcess
from dynamo.router.tests.support.port_utils import (
    allocate_port,
    allocate_ports,
    deallocate_port,
    deallocate_ports,
)

_logger = logging.getLogger(__name__)


def download_models(model_list=None, ignore_weights=False):
    """Download models - can be called directly or via fixture."""
    if model_list is None:
        model_list = TEST_MODELS

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
                weight_patterns = [
                    "*.bin",
                    "*.safetensors",
                    "*.h5",
                    "*.msgpack",
                    "*.ckpt.index",
                ]
                snapshot_download(
                    repo_id=model_id,
                    token=hf_token,
                    ignore_patterns=weight_patterns,
                )
            else:
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
    """Set HF_HUB_OFFLINE=1 and work around tokenizer startup API calls."""
    os.environ["HF_HUB_OFFLINE"] = "1"

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
        return

    patch_dir = os.path.join(tempfile.gettempdir(), "dynamo_test_hf_patch")
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
            "        def _safe_patch(cls, tokenizer, *args, **kwargs):\n"
            "            try:\n"
            "                return _orig.__func__(cls, tokenizer, *args, **kwargs)\n"
            "            except _E:\n"
            "                return tokenizer\n"
            "        _T._patch_mistral_regex = _safe_patch\n"
            "    except (ImportError, AttributeError):\n"
            "        pass\n"
        )

    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{patch_dir}:{pythonpath}" if pythonpath else patch_dir


def _disable_offline_with_mistral_patch():
    """Undo _enable_offline_with_mistral_patch."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    patch_dir = os.path.join(tempfile.gettempdir(), "dynamo_test_hf_patch")
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = pythonpath.replace(f"{patch_dir}:", "").replace(
        patch_dir, ""
    )


@pytest.fixture(scope="session")
def predownload_tokenizers(pytestconfig):
    """Fixture wrapper around download_models for tokenizers used in collected tests."""
    models = getattr(pytestconfig, "models_to_download", None)
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


@pytest.fixture
def discovery_backend(request):
    return getattr(request, "param", "etcd")


@pytest.fixture
def request_plane(request):
    return getattr(request, "param", "nats")


@pytest.fixture
def durable_kv_events(request):
    return getattr(request, "param", False)


class EtcdServer(ManagedProcess):
    def __init__(self, request, port=2379, timeout=300):
        use_random_port = port == 0
        if use_random_port:
            port, peer_port = allocate_ports(2, 2380)
        else:
            peer_port = None

        self.port = port
        self.peer_port = peer_port
        self.use_random_port = use_random_port
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

        command.extend(["--data-dir", data_dir])
        super().__init__(
            env=etcd_env,
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_all_matching_process_names=not use_random_port,
            health_check_ports=[port],
            data_dir=data_dir,
            log_dir=request.node.name,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
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
        use_random_port = port == 0
        if use_random_port:
            port = allocate_port(4223)

        self.port = port
        self.use_random_port = use_random_port
        self._request = request
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
            terminate_all_matching_process_names=not use_random_port,
            data_dir=data_dir,
            health_check_ports=[port],
            health_check_funcs=[self._nats_ready],
            log_dir=request.node.name,
        )

    def _nats_ready(self, timeout: float = 5) -> bool:
        import asyncio
        import concurrent.futures

        import nats

        async def check():
            try:
                nc = await nats.connect(
                    f"nats://localhost:{self.port}",
                    connect_timeout=min(timeout, 2),
                )
                try:
                    if not self._disable_jetstream:
                        js = nc.jetstream()
                        await js.account_info()
                    return True
                finally:
                    await nc.close()
            except Exception:
                return False

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, check()).result(timeout=timeout)
        except RuntimeError:
            return asyncio.run(check())

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.use_random_port:
                deallocate_port(self.port)
        except Exception as e:
            logging.warning(f"Failed to release NatsServer port: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)

    def stop(self):
        _logger.info(f"Stopping NATS server on port {self.port}")
        self._stop_started_processes()

    def start(self):
        _logger.info(f"Starting NATS server on port {self.port} with fresh state")
        if not self._disable_jetstream:
            old_data_dir = self.data_dir  # type: ignore[has-type]
            if old_data_dir is not None:
                shutil.rmtree(old_data_dir, ignore_errors=True)
            self.data_dir = tempfile.mkdtemp(prefix="nats_")

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


@pytest.fixture()
def runtime_services_dynamic_ports(
    request, discovery_backend, request_plane, durable_kv_events
):
    """Provide NATS and Etcd servers with truly dynamic ports per test."""
    if discovery_backend == "etcd":
        with NatsServer(
            request, port=0, disable_jetstream=not durable_kv_events
        ) as nats_process:
            with EtcdServer(request, port=0) as etcd_process:
                orig_nats = os.environ.get("NATS_SERVER")
                orig_etcd = os.environ.get("ETCD_ENDPOINTS")

                os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
                os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

                yield nats_process, etcd_process

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
