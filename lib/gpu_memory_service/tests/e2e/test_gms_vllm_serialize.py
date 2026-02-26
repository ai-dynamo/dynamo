# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: GMS serialize/deserialize with a real vLLM inference engine.

Test scenario
-------------
1. Start GPU Memory Service server (device 0).
2. Start vLLM (load_format=gms, GMSWorker) — the first process to connect gets
   the **RW lock** and loads model weights from disk into GMS, then commits.
3. Verify that a basic inference request succeeds (baseline).
4. Dump the live GMS state to a temporary directory on disk via
   ``gms-storage-client save``.
5. Kill the vLLM server and the GMS server.  Destroy everything.
6. Start a **fresh, empty** GMS server.
7. Restore the saved state into it via ``gms-storage-client load``, which
   replays all GPU allocations and calls commit — the server now has the
   weights without ever loading from HuggingFace / disk cache.
8. Start vLLM again with the same flags — this time it connects in **RO mode**
   (weights already committed) and attaches to the existing GPU allocations
   instead of loading from disk.
9. Confirm that an inference request succeeds and that vLLM logs show it used
   GMS read (RO) mode.

Running
-------
    # From lib/gpu_memory_service, with the gpu-memory-service package installed:
    pytest tests/e2e/test_gms_vllm_serialize.py -v -s

    # Choose a different model (must be vLLM-compatible):
    pytest tests/e2e/test_gms_vllm_serialize.py -v -s --model Qwen/Qwen2.5-0.5B

    # Override the vLLM API port (default: 18_700):
    pytest tests/e2e/test_gms_vllm_serialize.py -v -s --vllm-port 8100

Prerequisites
-------------
* CUDA GPU with enough VRAM for the chosen model.
* ``vllm`` installed in the active Python environment.
* ``gpu-memory-service`` installed (editable or otherwise) in the same env.
* The environment must provide ``ldconfig`` on PATH (required by Triton's
  CUDA discovery).  On NixOS use the provided ``shell.nix`` to enter a
  suitable FHS environment before running.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator, Optional

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


class ManagedProcess:
    """Subprocess wrapper that terminates the entire process group on close."""

    def __init__(self, cmd: list[str], log_path: Path, env: Optional[dict] = None):
        self.cmd = cmd
        self.log_path = log_path
        self._log_fh = log_path.open("w")
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env or os.environ.copy(),
            preexec_fn=os.setsid,
        )
        logger.info("Started pid=%d: %s", self._proc.pid, " ".join(cmd))
        logger.info("  log → %s", log_path)

    @property
    def pid(self) -> int:
        return self._proc.pid

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def terminate(self, name: str = "process") -> None:
        if not self.is_running():
            return
        logger.info("Terminating %s (pid=%d)…", name, self._proc.pid)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("SIGTERM timed out; sending SIGKILL to %s", name)
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                self._proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        logger.info("%s stopped.", name)

    def close(self) -> None:
        self._log_fh.close()

    def tail(self, n: int = 40) -> str:
        try:
            lines = self.log_path.read_text().splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return "(log unavailable)"


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _wait_gpu_free(exclude_pids: set[int], timeout: int = 120) -> bool:
    """Poll nvidia-smi until no compute apps remain (other than *exclude_pids*).

    Returns True when the GPU is clear, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = {
            int(p.strip())
            for p in result.stdout.strip().split("\n")
            if p.strip().isdigit()
        }
        remaining = pids - exclude_pids
        if not remaining:
            return True
        logger.info("Waiting for GPU drain; active pids: %s", sorted(remaining))
        time.sleep(3)
    return False


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _wait_for_http(url: str, timeout: int) -> bool:
    """Poll *url* until it returns HTTP 200 or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_inference(
    model: str, port: int, prompt: str = "The capital of France is"
) -> str:
    """Send a single completion request; return the generated text."""
    resp = _post_json(
        f"http://localhost:{port}/v1/completions",
        {"model": model, "prompt": prompt, "max_tokens": 10, "temperature": 0.0},
    )
    return resp["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Sub-command helpers
# ---------------------------------------------------------------------------


def _python() -> str:
    return sys.executable


def _gms_server_cmd(device: int = 0) -> list[str]:
    return [_python(), "-m", "gpu_memory_service", "--device", str(device)]


def _vllm_serve_cmd(model: str, port: int, device: int = 0) -> list[str]:
    return [
        _python(),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--load-format",
        "gms",
        "--worker-cls",
        "gpu_memory_service.integrations.vllm.worker.GMSWorker",
        "--port",
        str(port),
        "--max-model-len",
        "512",
        "--disable-log-stats",
    ]


def _gms_save_cmd(output_dir: Path, device: int = 0) -> list[str]:
    return [
        _python(),
        "-m",
        "gpu_memory_service.cli.storage_runner",
        "save",
        "--output-dir",
        str(output_dir),
        "--device",
        str(device),
        "--verbose",
    ]


def _gms_load_cmd(
    input_dir: Path, device: int = 0, restore_workers: int = 4
) -> list[str]:
    return [
        _python(),
        "-m",
        "gpu_memory_service.cli.storage_runner",
        "load",
        "--input-dir",
        str(input_dir),
        "--device",
        str(device),
        "--workers",
        str(restore_workers),
        "--verbose",
    ]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def _gms_socket_path(device: int = 0) -> Path:
    from gpu_memory_service.common.utils import get_socket_path

    return Path(get_socket_path(device))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_save_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    d = tmp_path_factory.mktemp("gms_save")
    yield d
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("gms_logs")


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestGMSSerializeDeserialize:
    """Serialize GMS state to disk, restore it, and confirm vLLM reuses weights."""

    def test_full_lifecycle(
        self,
        request: pytest.FixtureRequest,
        tmp_save_dir: Path,
        log_dir: Path,
    ) -> None:
        model: str = request.config.getoption("--model")
        port: int = request.config.getoption("--vllm-port")
        startup_timeout: int = request.config.getoption("--vllm-startup-timeout")
        restore_workers: int = request.config.getoption("--restore-workers")
        device: int = 0

        gms1: Optional[ManagedProcess] = None
        vllm1: Optional[ManagedProcess] = None
        gms2: Optional[ManagedProcess] = None
        vllm2: Optional[ManagedProcess] = None

        try:
            # ----------------------------------------------------------------
            # Phase 1 – load weights from disk into GMS (RW mode)
            # ----------------------------------------------------------------
            logger.info("=== Phase 1: start GMS + vLLM (RW mode) ===")

            gms1 = ManagedProcess(
                _gms_server_cmd(device),
                log_dir / "gms_phase1.log",
            )
            time.sleep(2)
            assert gms1.is_running(), f"GMS server exited unexpectedly.\n{gms1.tail()}"

            vllm1 = ManagedProcess(
                _vllm_serve_cmd(model, port, device),
                log_dir / "vllm_phase1.log",
            )
            assert _wait_for_http(f"http://localhost:{port}/health", startup_timeout), (
                f"vLLM (phase 1) did not become healthy within {startup_timeout}s.\n"
                f"vLLM log tail:\n{vllm1.tail()}\n"
                f"GMS log tail:\n{gms1.tail()}"
            )
            logger.info("vLLM (phase 1) is healthy.")

            # Baseline inference — confirm weights were loaded correctly.
            generated_text = run_inference(model, port)
            logger.info("Phase 1 inference OK. Generated: %r", generated_text)
            assert generated_text, "Phase 1 baseline inference returned empty text."

            # Confirm that vLLM loaded in RW (write) mode, not RO.
            vllm1_log = vllm1.log_path.read_text()
            assert "Write mode" in vllm1_log or "write mode" in vllm1_log, (
                "Expected vLLM to load weights in GMS write mode (RW), "
                "but could not find 'Write mode' in the log.\n"
                f"vLLM log tail:\n{vllm1.tail()}"
            )

            # ----------------------------------------------------------------
            # Phase 2 – dump GMS state to disk
            # ----------------------------------------------------------------
            logger.info("=== Phase 2: save GMS state to disk ===")

            _run(_gms_save_cmd(tmp_save_dir, device))

            manifest_path = tmp_save_dir / "manifest.json"
            assert (
                manifest_path.exists()
            ), "manifest.json not written by gms-storage-client save."
            manifest = json.loads(manifest_path.read_text())
            n_allocs = len(manifest.get("allocations", []))
            assert n_allocs > 0, "Save produced 0 allocations — nothing was saved."
            logger.info("Saved %d allocations to %s.", n_allocs, tmp_save_dir)

            # ----------------------------------------------------------------
            # Phase 3 – destroy vLLM and GMS
            # ----------------------------------------------------------------
            logger.info("=== Phase 3: tear down vLLM and GMS ===")

            vllm1.terminate("vLLM phase-1")
            vllm1.close()
            vllm1 = None

            gms1.terminate("GMS phase-1")
            gms1.close()
            gms1 = None

            # Wait for all vLLM GPU processes (including EngineCore subprocesses) to
            # release their CUDA VMM handles before starting a fresh GMS server.
            # The main vLLM API-server exits quickly but the EngineCore worker keeps
            # GPU memory alive until its CUDA context is torn down.  If we don't wait,
            # GMS2 will fail with cuMemCreate: out of memory.
            own_pid = os.getpid()
            if not _wait_gpu_free(exclude_pids={own_pid}, timeout=120):
                logger.warning(
                    "GPU did not fully drain within 120 s; proceeding anyway."
                )

            # Remove stale socket so the fresh GMS server can bind.
            sock = _gms_socket_path(device)
            if sock.exists():
                sock.unlink()
                logger.info("Removed stale socket %s.", sock)

            # ----------------------------------------------------------------
            # Phase 4 – start fresh GMS and restore state from disk
            # ----------------------------------------------------------------
            logger.info("=== Phase 4: fresh GMS server + restore from disk ===")

            gms2 = ManagedProcess(
                _gms_server_cmd(device),
                log_dir / "gms_phase2.log",
            )
            time.sleep(2)
            assert (
                gms2.is_running()
            ), f"GMS server (phase 2) exited unexpectedly.\n{gms2.tail()}"

            restore_total_bytes = sum(
                a["aligned_size"]
                for a in json.loads((tmp_save_dir / "manifest.json").read_text()).get(
                    "allocations", []
                )
            )
            restore_t0 = time.monotonic()
            _run(_gms_load_cmd(tmp_save_dir, device, restore_workers=restore_workers))
            restore_elapsed = time.monotonic() - restore_t0
            restore_throughput = restore_total_bytes / (restore_elapsed * 1024**3)
            logger.info(
                "GMS state restored and committed: %.3f GiB in %.3f s  (%.2f GiB/s, workers=%d)",
                restore_total_bytes / 1024**3,
                restore_elapsed,
                restore_throughput,
                restore_workers,
            )

            # ----------------------------------------------------------------
            # Phase 5 – attach vLLM in RO mode and run inference
            # ----------------------------------------------------------------
            logger.info("=== Phase 5: start vLLM (RO mode) + inference ===")

            vllm2 = ManagedProcess(
                _vllm_serve_cmd(model, port, device),
                log_dir / "vllm_phase2.log",
            )
            assert _wait_for_http(f"http://localhost:{port}/health", startup_timeout), (
                f"vLLM (phase 2) did not become healthy within {startup_timeout}s.\n"
                f"vLLM log tail:\n{vllm2.tail()}\n"
                f"GMS log tail:\n{gms2.tail()}"
            )
            logger.info("vLLM (phase 2) is healthy.")

            # The critical assertion: vLLM must have attached in RO mode.
            vllm2_log = vllm2.log_path.read_text()
            assert "Read mode" in vllm2_log or "read mode" in vllm2_log, (
                "Expected vLLM (phase 2) to import weights in GMS read (RO) mode, "
                "but could not find 'Read mode' in the log.  "
                "This suggests vLLM reloaded from disk instead of reusing saved GMS state.\n"
                f"vLLM log tail:\n{vllm2.tail()}"
            )

            # Inference must succeed with the restored weights.
            restored_text = run_inference(model, port)
            logger.info("Phase 2 inference OK. Generated: %r", restored_text)
            assert restored_text, "Inference after GMS restore returned empty text."

            logger.info(
                "SUCCESS: weights serialized to disk, GMS destroyed and recreated, "
                "vLLM re-attached in RO mode, inference succeeded.\n"
                "  Phase-1 output: %r\n  Phase-2 output: %r",
                generated_text,
                restored_text,
            )

        finally:
            for proc, name in [
                (vllm2, "vLLM phase-2"),
                (gms2, "GMS phase-2"),
                (vllm1, "vLLM phase-1"),
                (gms1, "GMS phase-1"),
            ]:
                if proc is not None:
                    proc.terminate(name)
                    proc.close()
