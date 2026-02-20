# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint/restore (chrek) integration for vLLM workers.

Handles the checkpoint job pod lifecycle:
1. Early exit if a checkpoint already exists (idempotency)
2. Sleep model for CRIU-friendly GPU state
3. Signal readiness for DaemonSet to begin checkpoint
4. Poll for checkpoint completion or CRIU restore detection
5. Wake model after restore

Environment variables (all required in checkpoint mode, no fallbacks):
- DYN_CHECKPOINT_SIGNAL_FILE: Path where DaemonSet writes completion signal
- DYN_READY_FOR_CHECKPOINT_FILE: Path where this worker writes readiness marker
- DYN_CHECKPOINT_STORAGE_TYPE: Storage backend (pvc, s3, oci)
- DYN_CHECKPOINT_LOCATION: Full checkpoint path (for idempotency check)
- DYN_RESTORE_MARKER_FILE: Path written by restore-entrypoint before CRIU restore
"""

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = [
    "DYN_CHECKPOINT_SIGNAL_FILE",
    "DYN_READY_FOR_CHECKPOINT_FILE",
    "DYN_CHECKPOINT_STORAGE_TYPE",
    "DYN_CHECKPOINT_LOCATION",
    "DYN_RESTORE_MARKER_FILE",
]


class CheckpointConfig:
    """Parsed and validated checkpoint configuration from environment variables."""

    def __init__(self):
        self.signal_file = os.environ["DYN_CHECKPOINT_SIGNAL_FILE"]
        self.ready_file = os.environ["DYN_READY_FOR_CHECKPOINT_FILE"]
        self.storage_type = os.environ["DYN_CHECKPOINT_STORAGE_TYPE"]
        self.location = os.environ["DYN_CHECKPOINT_LOCATION"]
        self.restore_marker = os.environ["DYN_RESTORE_MARKER_FILE"]

    def _read_status_file(self, path: str) -> dict:
        with open(path) as f:
            status = json.load(f)

        success = status.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"missing or invalid success field in {path}")
        return status

    def checkpoint_exists(self) -> bool:
        """Check if a completed checkpoint already exists (idempotency).

        For PVC storage, checks for checkpoint.done marker at the location.
        Returns True if the job should exit without loading the model.
        """
        assert (
            self.storage_type == "pvc"
        ), "Checkpoint existence check is only implemented for PVC storage"
        if self.storage_type == "pvc" and self.location:
            done_marker = f"{self.location}/checkpoint.done"
            if os.path.exists(done_marker):
                try:
                    status = self._read_status_file(done_marker)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        f"Invalid checkpoint.done marker at {done_marker}, ignoring stale checkpoint: {exc}"
                    )
                    return False

                if status["success"]:
                    logger.info(
                        f"Existing successful checkpoint found at {self.location}, skipping"
                    )
                    return True

                logger.warning(
                    f"Existing checkpoint marker reports failure at {self.location}: "
                    f"{status.get('error', 'unknown error')}"
                )
                return False

            logger.info(f"No checkpoint at {self.location}, creating new one")
        return False

    async def run_lifecycle(self, engine_client, sleep_level: int) -> bool:
        """Run the full checkpoint lifecycle after the engine is loaded.

        1. Put model to sleep (CRIU-friendly GPU state)
        2. Write ready file (triggers DaemonSet checkpoint via readiness probe)
        3. Poll for signal file (checkpoint done) or restore marker (CRIU restored us)
        4. If restored: wake model and return True (caller proceeds with registration)
        5. If checkpoint done: return False (caller should exit)
        """
        # Sleep model for checkpoint
        logger.info(f"Putting model to sleep (level={sleep_level})")
        await engine_client.sleep(level=sleep_level)

        # Signal readiness
        with open(self.ready_file, "w") as f:
            f.write("ready")
        logger.info(
            f"Ready for checkpoint. Waiting for signal: {self.signal_file} "
            f"or restore marker: {self.restore_marker}"
        )

        # Poll for signal or restore
        while True:
            if os.path.exists(self.restore_marker):
                logger.info(f"Restore detected (marker: {self.restore_marker})")
                logger.info("Waking up model after restore")
                await engine_client.wake_up()
                return True

            if os.path.exists(self.signal_file):
                try:
                    signal = self._read_status_file(self.signal_file)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Invalid checkpoint signal file {self.signal_file}: {exc}"
                    ) from exc

                if signal["success"]:
                    logger.info(f"Checkpoint complete (signal: {self.signal_file})")
                    return False

                raise RuntimeError(
                    f"Checkpoint failed (signal: {self.signal_file}): "
                    f"{signal.get('error', 'unknown error')}"
                )

            await asyncio.sleep(1)


def get_checkpoint_config() -> Optional[CheckpointConfig]:
    """Returns CheckpointConfig if in checkpoint mode, None otherwise.

    Checkpoint mode is detected by DYN_CHECKPOINT_SIGNAL_FILE being set.
    If in checkpoint mode, all required env vars must be present — raises
    EnvironmentError if any are missing.
    """
    if "DYN_CHECKPOINT_SIGNAL_FILE" not in os.environ:
        return None

    missing = [v for v in _REQUIRED_ENV_VARS if v not in os.environ]
    if missing:
        raise EnvironmentError(
            f"Checkpoint mode requires these environment variables: {', '.join(missing)}"
        )
    return CheckpointConfig()
