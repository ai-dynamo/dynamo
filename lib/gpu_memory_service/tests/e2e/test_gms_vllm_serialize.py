# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process E2E test: GMS serialize/deserialize with a vLLM engine.

This variant avoids subprocesses and `nvidia-smi` polling entirely:

1. Start a GMS server in a background thread.
2. Instantiate an in-process vLLM `LLM` with `load_format="gms"`.
3. Verify generation works and the loader logged GMS write mode.
4. Save GMS state to disk via `GMSStorageClient.save()`.
5. Tear down the engine and GMS server, then start a fresh GMS server.
6. Restore the saved state via `GMSStorageClient.load_to_gms()`.
7. Instantiate a second in-process vLLM `LLM` in read-only GMS mode.
8. Verify generation works and the loader logged GMS read mode.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import shutil
import threading
from pathlib import Path
from typing import Generator, Optional

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
uvloop = pytest.importorskip("uvloop")

from gpu_memory_service.client.gms_storage_client import GMSStorageClient  # noqa: E402
from gpu_memory_service.common.utils import get_socket_path  # noqa: E402
from gpu_memory_service.integrations.vllm import (  # noqa: E402, F401
    worker as _gms_vllm_worker,
)
from gpu_memory_service.server import GMSRPCServer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

logger = logging.getLogger(__name__)


def _run_server(
    socket_path: str,
    ready_event: threading.Event,
    stop_event: threading.Event,
) -> None:
    async def _serve() -> None:
        server = GMSRPCServer(socket_path, device=0)
        await server.start()
        ready_event.set()
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.05)
        finally:
            await server.stop()

    uvloop.install()
    asyncio.run(_serve())


class ThreadedGMSServer:
    """Thread wrapper for an in-process GMS server."""

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self.socket_path = str(get_socket_path(device))
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=_run_server,
            args=(self.socket_path, self._ready, self._stop),
            daemon=True,
        )

    def start(self) -> None:
        sock = Path(self.socket_path)
        if sock.exists():
            sock.unlink()
        self._thread.start()
        self._started = True
        assert self._ready.wait(timeout=10), "GMS server did not start in time"

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            logger.warning("GMS server thread did not stop within 10 s")
        sock = Path(self.socket_path)
        if sock.exists():
            sock.unlink()


def _reset_gms_allocator_singleton() -> None:
    """Clear the process-global GMS allocator state between vLLM engine lifecycles."""
    from gpu_memory_service.client.torch import allocator as gms_allocator

    manager = gms_allocator.get_gms_client_memory_manager()
    if manager is not None:
        manager.close()

    gms_allocator._manager = None
    gms_allocator._mem_pool = None
    gms_allocator._tag = "weights"


def _destroy_llm(llm: Optional[LLM]) -> None:
    """Best-effort cleanup for an in-process vLLM engine."""
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    if llm is not None:
        try:
            sleep = getattr(llm, "sleep", None)
            if callable(sleep):
                sleep()
        except Exception:
            pass

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    cleanup_dist_env_and_memory()

    _reset_gms_allocator_singleton()


def _build_llm(model: str, *, read_only: bool) -> LLM:
    extra = {"gms_read_only": True} if read_only else {}
    return LLM(
        model=model,
        load_format="gms",
        worker_cls="gpu_memory_service.integrations.vllm.worker.GMSWorker",
        enforce_eager=True,
        max_model_len=512,
        trust_remote_code=False,
        disable_log_stats=True,
        gpu_memory_utilization=0.1,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        model_loader_extra_config=extra,
    )


def _generate_text(llm: LLM, prompt: str = "The capital of France is") -> str:
    outputs = llm.generate(
        prompt,
        SamplingParams(max_tokens=10, temperature=0.0),
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text


@pytest.fixture(scope="module")
def tmp_save_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    d = tmp_path_factory.mktemp("gms_save")
    yield d
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.e2e
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGMSSerializeDeserialize:
    """Serialize GMS state to disk, restore it, and confirm vLLM reuses weights."""

    def test_full_lifecycle(
        self,
        request: pytest.FixtureRequest,
        tmp_save_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        model: str = request.config.getoption("--model")
        restore_workers: int = request.config.getoption("--restore-workers")

        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        gms1 = ThreadedGMSServer(device=0)
        gms2 = ThreadedGMSServer(device=0)
        llm1: Optional[LLM] = None
        llm2: Optional[LLM] = None

        caplog.set_level(logging.INFO)

        try:
            logger.info(
                "=== Phase 1: start threaded GMS + in-process vLLM (RW mode) ==="
            )
            gms1.start()

            caplog.clear()
            llm1 = _build_llm(model, read_only=False)
            generated_text = _generate_text(llm1)
            assert generated_text, "Phase 1 baseline generation returned empty text."
            assert "Write mode" in caplog.text or "write mode" in caplog.text, (
                "Expected vLLM to publish weights in GMS write mode.\n"
                f"Captured logs:\n{caplog.text}"
            )

            logger.info("=== Phase 2: save GMS state to disk ===")
            client = GMSStorageClient(
                str(tmp_save_dir),
                socket_path=gms1.socket_path,
                device=0,
            )
            manifest = client.save()
            assert manifest.allocations, "Save produced 0 allocations."

            logger.info("=== Phase 3: tear down engine and GMS ===")
            _destroy_llm(llm1)
            llm1 = None
            gms1.stop()

            logger.info("=== Phase 4: start fresh threaded GMS + restore ===")
            gms2.start()
            restore_client = GMSStorageClient(
                socket_path=gms2.socket_path,
                device=0,
            )
            id_map = restore_client.load_to_gms(
                str(tmp_save_dir),
                max_workers=restore_workers,
            )
            assert len(id_map) == len(manifest.allocations)

            logger.info("=== Phase 5: in-process vLLM attaches in RO mode ===")
            caplog.clear()
            llm2 = _build_llm(model, read_only=True)
            restored_text = _generate_text(llm2)
            assert restored_text, "Generation after GMS restore returned empty text."
            assert "Read mode" in caplog.text or "read mode" in caplog.text, (
                "Expected vLLM to import weights in GMS read mode.\n"
                f"Captured logs:\n{caplog.text}"
            )
        finally:
            _destroy_llm(llm2)
            _destroy_llm(llm1)
            gms2.stop()
            gms1.stop()
