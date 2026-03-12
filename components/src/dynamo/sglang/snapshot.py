# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Snapshot integration for SGLang workers.

Handles the checkpoint job pod lifecycle:
1. Early exit if a checkpoint already exists (idempotency)
2. Sleep model for CRIU-friendly GPU state
3. Signal readiness for DaemonSet to begin checkpoint
4. Wait for watcher signals from the DaemonSet
5. Wake model after restore

SGLang does not have a native sleep/wake API like vLLM.  Instead we use
release_memory_occupation / resume_memory_occupation through the
SGLangCheckpointAdapter, which presents the same sleep()/wake_up()
interface that CheckpointConfig.run_lifecycle expects.

Environment variables:
- DYN_READY_FOR_CHECKPOINT_FILE: Path where this worker writes readiness marker
- DYN_CHECKPOINT_STORAGE_TYPE: Storage backend (pvc, s3, oci) (optional, defaults to pvc)
- DYN_CHECKPOINT_LOCATION: Full checkpoint path (optional when PATH+HASH are provided)
- DYN_CHECKPOINT_PATH + DYN_CHECKPOINT_HASH: PVC base path + hash (used to derive location)

Signals handled in checkpoint mode:
- SIGUSR1: Checkpoint completed, exit process
- SIGCONT: Restore completed, wake model and continue
- SIGKILL (from watcher on failure): Process is terminated immediately (unhandleable)
"""

import asyncio
import ctypes
import gc
import logging
import os
import signal
import time
from typing import Optional

import sglang as sgl

logger = logging.getLogger(__name__)


def _log_memory_diagnostics(label: str) -> None:
    """Log detailed memory breakdown for the current process.

    Parses /proc/self/smaps to separate anonymous (heap, mmap'd buffers) from
    file-backed (mmap'd files like safetensors shards) resident pages.  Large
    anonymous regions are the main contributor to pages-*.img bloat in CRIU
    checkpoints.
    """
    pid = os.getpid()

    # Quick summary from /proc/self/status
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(
                    ("VmRSS:", "VmSize:", "RssAnon:", "RssFile:", "RssShmem:")
                ):
                    logger.info("[MemDiag:%s] %s", label, line.strip())
    except Exception as e:
        logger.warning("[MemDiag:%s] Failed to read /proc status: %s", label, e)

    # Detailed breakdown from /proc/self/smaps
    try:
        anon_rss_total = 0
        file_rss_total = 0
        large_regions: list[tuple[int, str]] = []
        current_header = ""
        current_rss = 0
        current_is_anon = False

        with open(f"/proc/{pid}/smaps") as f:
            for line in f:
                if line[0] in "0123456789abcdef":
                    # Flush previous region
                    if current_header and current_rss > 0:
                        if current_is_anon:
                            anon_rss_total += current_rss
                        else:
                            file_rss_total += current_rss
                        if current_rss >= 10 * 1024:  # >= 10 MiB
                            large_regions.append((current_rss, current_header.strip()))
                    current_header = line
                    current_rss = 0
                    # Anonymous = no file path, or [heap], [stack], [anon:*]
                    parts = line.split()
                    pathname = parts[5].strip() if len(parts) > 5 else ""
                    current_is_anon = (
                        pathname == ""
                        or pathname.startswith("[heap]")
                        or pathname.startswith("[stack")
                        or pathname.startswith("[anon:")
                    )
                elif line.startswith("Rss:"):
                    current_rss = int(line.split()[1])  # in kB

        # Flush last region
        if current_header and current_rss > 0:
            if current_is_anon:
                anon_rss_total += current_rss
            else:
                file_rss_total += current_rss
            if current_rss >= 10 * 1024:
                large_regions.append((current_rss, current_header.strip()))

        logger.info(
            "[MemDiag:%s] RSS breakdown: anonymous=%.2f MiB, file-backed=%.2f MiB, total=%.2f MiB",
            label,
            anon_rss_total / 1024,
            file_rss_total / 1024,
            (anon_rss_total + file_rss_total) / 1024,
        )

        # Log the largest regions (these are prime suspects for pages-*.img bloat)
        large_regions.sort(reverse=True)
        for rss_kb, header in large_regions[:15]:
            logger.info(
                "[MemDiag:%s] Large region: %.2f MiB  %s",
                label,
                rss_kb / 1024,
                header[:200],
            )
    except Exception as e:
        logger.warning("[MemDiag:%s] Failed to parse smaps: %s", label, e)


def _try_release_memory(label: str) -> None:
    """Force Python GC and glibc malloc_trim to return freed memory to the OS.

    Logs RSS before/after so you can see how much memory was actually reclaimable.
    """
    pid = os.getpid()

    def _get_rss_kb() -> int:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    rss_before = _get_rss_kb()

    # Force Python garbage collection (breaks reference cycles)
    collected = gc.collect()
    rss_after_gc = _get_rss_kb()

    # Ask glibc to return freed heap pages to the OS
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception as e:
        logger.debug("[MemRelease:%s] malloc_trim failed: %s", label, e)

    rss_after_trim = _get_rss_kb()

    logger.info(
        "[MemRelease:%s] gc.collect freed %d objects, "
        "RSS: %.2f MiB -> %.2f MiB (gc) -> %.2f MiB (malloc_trim), "
        "reclaimed=%.2f MiB",
        label,
        collected,
        rss_before / 1024,
        rss_after_gc / 1024,
        rss_after_trim / 1024,
        (rss_before - rss_after_trim) / 1024,
    )


_SLEEP_MODE_LEVEL = 1

# Memory tags to release/resume for CRIU checkpoint/restore.
# All GPU resources must be released so CRIU can snapshot the process cleanly.
_MEMORY_TAGS = ["kv_cache", "weights", "cuda_graph"]


class SGLangCheckpointAdapter:
    """Adapts an sgl.Engine to the sleep/wake_up interface expected by
    CheckpointConfig.run_lifecycle (matching vLLM's AsyncLLM API).

    sleep():   pause generation -> release GPU memory
    wake_up(): resume GPU memory -> continue generation
    """

    def __init__(self, engine: sgl.Engine):
        self._engine = engine

    async def sleep(self, level: int = 1) -> None:
        logger.info("Sleeping engine")
        from sglang.srt.managers.io_struct import (
            PauseGenerationReqInput,
            ReleaseMemoryOccupationReqInput,
        )

        # Drain in-flight requests before touching GPU memory
        await self._engine.tokenizer_manager.pause_generation(PauseGenerationReqInput())
        await self._engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=_MEMORY_TAGS), None
        )

    async def wake_up(self) -> None:
        from sglang.srt.managers.io_struct import (
            ContinueGenerationReqInput,
            ResumeMemoryOccupationReqInput,
        )

        # Synchronize the CUDA context in this (tokenizer-manager) process so
        # any pending ops complete before we message the scheduler.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning("CUDA sync before resume_memory_occupation failed: %s", e)

        await self._engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=_MEMORY_TAGS), None
        )
        await self._engine.tokenizer_manager.continue_generation(
            ContinueGenerationReqInput()
        )


class CheckpointConfig:
    """Parsed and validated checkpoint configuration from environment variables."""

    def __init__(self):
        self.ready_file = os.environ["DYN_READY_FOR_CHECKPOINT_FILE"]
        self.storage_type = os.environ.get("DYN_CHECKPOINT_STORAGE_TYPE", "pvc")
        self.location = os.environ.get("DYN_CHECKPOINT_LOCATION", "")
        if not self.location:
            checkpoint_path = os.environ.get("DYN_CHECKPOINT_PATH", "").rstrip("/")
            checkpoint_hash = os.environ.get("DYN_CHECKPOINT_HASH", "")
            if checkpoint_path and checkpoint_hash:
                self.location = f"{checkpoint_path}/{checkpoint_hash}"
        self.is_checkpoint_job = bool(self.location)
        self._checkpoint_done = asyncio.Event()
        self._restore_done = asyncio.Event()

    def checkpoint_exists(self) -> bool:
        """Check if a completed checkpoint already exists (idempotency).

        A checkpoint is complete when its directory exists at the base path root
        (not under the tmp/ staging area). Directory presence = done.
        """
        if self.storage_type != "pvc":
            return False

        if os.path.isdir(self.location):
            logger.info(f"Existing checkpoint found at {self.location}, skipping")
            return True

        logger.info(f"No checkpoint at {self.location}, creating new one")
        return False

    async def run_lifecycle(self, engine_client, sleep_level: int) -> bool:
        """Run the full checkpoint lifecycle after the engine is loaded.

        1. Put model to sleep (CRIU-friendly GPU state)
        2. Write ready file (triggers DaemonSet checkpoint via readiness probe)
        3. Wait for watcher signal (checkpoint complete, restore complete, or failure)
        4. If restored: wake model and return True (caller proceeds with registration)
        5. If checkpoint done: return False (caller should exit)
        """
        # Sleep model for checkpoint
        logger.info(f"Putting model to sleep (level={sleep_level})")
        await engine_client.sleep(level=sleep_level)

        _log_memory_diagnostics("after_sleep")
        _try_release_memory("after_sleep")
        _log_memory_diagnostics("pre_checkpoint")

        # Install signal handlers before writing the ready file so there is no
        # window where the DaemonSet can send SIGUSR1/SIGCONT while the default
        # signal disposition (terminate) is still in effect.
        self._install_signal_handlers()

        # Signal readiness
        with open(self.ready_file, "w") as f:
            f.write("ready")
        logger.info(
            "Ready for checkpoint. Waiting for watcher signal "
            "(SIGUSR1=checkpoint complete, SIGCONT=restore complete)"
        )

        try:
            event = await self._wait_for_watcher_signal()
            if event == "restore":
                logger.info("Restore signal detected (SIGCONT)")
                logger.info("Waking up model after restore")
                await engine_client.wake_up()
                return True

            # SIGUSR1: checkpoint complete
            logger.info("Checkpoint completion signal detected (SIGUSR1)")
            return False
        finally:
            self._remove_signal_handlers()
            # Remove the ready file so that a restarting pod does not leave a
            # stale marker that could trick the DaemonSet into acting on it.
            try:
                os.unlink(self.ready_file)
            except OSError:
                pass

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, self._checkpoint_done.set)
        # SIGCONT is used as the restore-complete signal. The snapshot DaemonSet
        # watcher is the only sender, so there is no conflict with POSIX
        # job-control semantics in practice.
        loop.add_signal_handler(signal.SIGCONT, self._restore_done.set)
        # No handler for checkpoint failure: the watcher sends SIGKILL, which
        # terminates the process immediately (cannot be caught).

    def _remove_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        loop.remove_signal_handler(signal.SIGUSR1)
        loop.remove_signal_handler(signal.SIGCONT)

    async def _wait_for_watcher_signal(self) -> str:
        waiters = {
            asyncio.create_task(self._checkpoint_done.wait()): "checkpoint",
            asyncio.create_task(self._restore_done.wait()): "restore",
        }
        try:
            done, pending = await asyncio.wait(
                waiters.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            winner = done.pop()
            await winner
            return waiters[winner]
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()


async def handle_checkpoint_mode(server_args) -> tuple[bool, Optional[sgl.Engine]]:
    """Single entry point for Dynamo Snapshot integration.

    Must be called BEFORE runtime creation so the engine can be checkpointed
    without active NATS/etcd connections.

    Returns:
        (should_exit, engine) where:
        - (True, None): caller should return immediately (checkpoint already
          exists, or checkpoint completed successfully).
        - (False, None): not in checkpoint mode — cold-start normally.
        - (False, engine): restore completed — caller should use this engine.
    """
    if "DYN_READY_FOR_CHECKPOINT_FILE" not in os.environ:
        return False, None

    # Validate: either a full location or path + hash must be set.
    if not os.environ.get("DYN_CHECKPOINT_LOCATION"):
        path = os.environ.get("DYN_CHECKPOINT_PATH", "")
        hash_ = os.environ.get("DYN_CHECKPOINT_HASH", "")
        if not path or not hash_:
            raise EnvironmentError(
                "Checkpoint mode requires either DYN_CHECKPOINT_LOCATION or both "
                "DYN_CHECKPOINT_PATH and DYN_CHECKPOINT_HASH"
            )

    cfg = CheckpointConfig()
    checkpoint_exists = cfg.checkpoint_exists()

    if cfg.is_checkpoint_job and checkpoint_exists:
        return True, None

    if not cfg.is_checkpoint_job and not checkpoint_exists:
        return False, None

    logger.info("Checkpoint mode enabled (watcher-driven signals)")

    # Enable memory_saver so GPU memory can be released for CRIU.
    # When using GMS, weights use VA-stable unmap/remap (no CPU backup); GMS
    # forbids enable_weights_cpu_backup. Otherwise use CPU backup for weights.
    server_args.enable_memory_saver = True
    _using_gms = getattr(server_args, "load_format", None) == "gms" or (
        isinstance(getattr(server_args, "load_format", None), type)
        and getattr(server_args.load_format, "__name__", "") == "GMSModelLoader"
    )
    if not _using_gms:
        server_args.enable_weights_cpu_backup = True

    # NOTE: The GMS static-state patch (no-op _export/_import_static_state)
    # is applied inside the scheduler child process via
    # gpu_memory_service.integrations.sglang.patches.patch_static_state_for_gms,
    # triggered when the child unpickles server_args.load_format=GMSModelLoader.
    # Patching here in the parent would have no effect because SGLang uses
    # multiprocessing spawn, not fork.

    start_time = time.time()
    engine = sgl.Engine(server_args=server_args)
    logger.info(
        f"SGLang engine loaded in {time.time() - start_time:.2f}s (checkpoint mode)"
    )

    _log_memory_diagnostics("after_engine_load")
    _try_release_memory("after_engine_load")
    _log_memory_diagnostics("after_engine_load_cleanup")

    adapter = SGLangCheckpointAdapter(engine)
    if not await cfg.run_lifecycle(adapter, _SLEEP_MODE_LEVEL):
        return True, None

    return False, engine
