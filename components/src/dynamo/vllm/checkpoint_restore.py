# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint/restore (chrek) integration for vLLM workers.

Handles the checkpoint job pod lifecycle:
1. Early exit if a checkpoint already exists (idempotency)
2. Sleep model for CRIU-friendly GPU state
3. Signal readiness for DaemonSet to begin checkpoint
4. Wait for watcher signals from the DaemonSet
5. Wake model after restore

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
import logging
import os
import signal
from typing import Optional

logger = logging.getLogger(__name__)


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
        # SIGCONT is used as the restore-complete signal. The chrek DaemonSet
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


def get_checkpoint_config() -> tuple[bool, Optional[CheckpointConfig]]:
    """Resolve checkpoint configuration, handling early-exit and cold-start cases.

    Checkpoint mode is detected by DYN_READY_FOR_CHECKPOINT_FILE being set.

    Returns:
        (early_exit, config) where:
        - early_exit=True, config=None: checkpoint job re-run, checkpoint already
          exists — caller should return immediately.
        - early_exit=False, config=None: not in checkpoint mode, or regular worker
          with no checkpoint available yet — cold-start normally.
        - early_exit=False, config=CheckpointConfig: checkpoint lifecycle should run.
    """
    if "DYN_READY_FOR_CHECKPOINT_FILE" not in os.environ:
        return False, None

    # Validate checkpoint location: either a full location or path + hash must be set.
    # Check the value (not just presence) so an empty string is treated as unset.
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
        # Idempotent checkpoint job re-run: checkpoint already exists.
        return True, None

    if not cfg.is_checkpoint_job and not checkpoint_exists:
        # Regular worker with no checkpoint available yet: cold-start normally.
        return False, None

    return False, cfg
