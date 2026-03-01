# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from utils import Trajectory

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Replay buffer storing trajectories with weight version tracking.

    Key features:
    - Weight version tracking for importance sampling correction
    - Age-based filtering to prevent stale trajectories
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        logger.debug(f"[Buffer] Initializing replay buffer (max_size={max_size})")
        self.max_size = max_size
        self.trajectories: deque[Trajectory] = deque(maxlen=max_size)
        self.last_target_weight_already_generated: int = -1
        self._lock = threading.Lock()

        # Stats
        self.total_pushed = 0
        self.total_sampled = 0
        self.total_expired = 0

    def push(
        self,
        trajectory: Trajectory,
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """
        Add a trajectory with weight version metadata.

        Args:
            trajectory: The trajectory to add
            weight_version: Model weights used to generate this trajectory
            target_weight_version: Training step this trajectory is intended for

        Returns:
            "success" if added, "full" if buffer is at capacity
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            trajectory.generation_weight_version = weight_version
            trajectory.target_weight_version = target_weight_version
            trajectory.timestamp = time.time()

            self.trajectories.append(trajectory)
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated, target_weight_version
            )
            self.total_pushed += 1

            return "success"

    def sample(
        self,
        num_trajectories: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[Tuple[List[Trajectory], float]]:
        """
        Sample trajectories for the current training step.

        Args:
            num_trajectories: Number of trajectories to sample
            current_weight_version: Current training step
            max_age_steps: Maximum age for trajectories in the buffer

        Returns:
            Tuple of (trajectories, avg_age) or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            # Remove expired trajectories
            min_valid_version = max(0, current_weight_version - max_age_steps)
            expired_count = 0
            while (
                self.trajectories
                and self.trajectories[0].generation_weight_version < min_valid_version
            ):
                self.trajectories.popleft()
                expired_count += 1
                self.total_expired += 1

            if expired_count > 0:
                logger.debug(f"[Buffer] Expired {expired_count} stale trajectories")

            # Find trajectories intended for current step
            intended_indices = [
                i
                for i, t in enumerate(self.trajectories)
                if t.target_weight_version == current_weight_version
                and min_valid_version
                <= t.generation_weight_version
                <= current_weight_version
            ]

            if len(intended_indices) < num_trajectories:
                return None

            # Select trajectories
            selected_indices = intended_indices[:num_trajectories]
            selected = [self.trajectories[i] for i in selected_indices]

            # Compute average age
            selected_versions = [
                self.trajectories[i].generation_weight_version for i in selected_indices
            ]
            avg_age = current_weight_version - sum(selected_versions) / len(
                selected_versions
            )

            # Remove selected from buffer
            for idx in sorted(selected_indices, reverse=True):
                del self.trajectories[idx]

            self.total_sampled += len(selected)

            return selected, avg_age

    def size(self) -> int:
        with self._lock:
            return len(self.trajectories)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_size": len(self.trajectories),
                "max_size": self.max_size,
                "total_pushed": self.total_pushed,
                "total_sampled": self.total_sampled,
                "total_expired": self.total_expired,
                "last_target_generated": self.last_target_weight_already_generated,
            }

    def get_existing_target_weights(self) -> set:
        """Get set of target weight versions that have generated trajectories."""
        with self._lock:
            return set(t.target_weight_version for t in self.trajectories)
