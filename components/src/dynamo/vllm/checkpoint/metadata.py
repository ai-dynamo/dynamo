# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint metadata management for CRIU checkpoint/restore."""

import json
import os
from typing import Optional

import logging
logger = logging.getLogger(__name__)


class CheckpointMetadata:
    """Metadata saved alongside CRIU checkpoint for restore."""

    def __init__(self):
        self.tty_rdev: Optional[str] = None
        self.tty_dev: Optional[str] = None
        self.tree_pid: Optional[int] = None
        self.zmq_port: Optional[int] = None
        self.cuda_pids: list[int] = []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tty_rdev": self.tty_rdev,
            "tty_dev": self.tty_dev,
            "tree_pid": self.tree_pid,
            "zmq_port": self.zmq_port,
            "cuda_pids": self.cuda_pids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        """Create from dictionary loaded from JSON."""
        meta = cls()
        meta.tty_rdev = data.get("tty_rdev")
        meta.tty_dev = data.get("tty_dev")
        meta.tree_pid = data.get("tree_pid")
        meta.zmq_port = data.get("zmq_port")
        meta.cuda_pids = data.get("cuda_pids", [])
        return meta

    def save(self, checkpoint_dir: str) -> None:
        """Save metadata to JSON file in checkpoint directory."""
        try:
            path = os.path.join(checkpoint_dir, "vllm_checkpoint_metadata.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("Saved checkpoint metadata to %s", path)
        except Exception as e:
            logger.warning("Failed to save checkpoint metadata: %s", e)

    @classmethod
    def load(cls, checkpoint_dir: str) -> Optional["CheckpointMetadata"]:
        """Load metadata from JSON file in checkpoint directory."""
        try:
            path = os.path.join(checkpoint_dir, "vllm_checkpoint_metadata.json")
            if not os.path.exists(path):
                logger.warning("Checkpoint metadata file not found: %s", path)
                return None
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load checkpoint metadata: %s", e)
            return None

    @property
    def tty_external(self) -> Optional[str]:
        """Format TTY info for CRIU external option."""
        if self.tty_rdev and self.tty_dev:
            return f"tty[{self.tty_rdev}:{self.tty_dev}]"
        return None
