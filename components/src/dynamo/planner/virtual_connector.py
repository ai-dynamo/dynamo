# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time
from typing import Optional

from dynamo.planner.defaults import WORKER_COMPONENT_NAMES
from dynamo.planner.planner_connector import PlannerConnector
from dynamo.runtime import DistributedRuntime, EtcdKvCache
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Constants for scaling readiness check and waiting
SCALING_CHECK_INTERVAL = int(
    os.environ.get("SCALING_CHECK_INTERVAL", 10)
)  # Check every 10 seconds
SCALING_MAX_WAIT_TIME = int(
    os.environ.get("SCALING_MAX_WAIT_TIME", 1800)
)  # Maximum wait time: 30 minutes (1800 seconds)
SCALING_MAX_RETRIES = SCALING_MAX_WAIT_TIME // SCALING_CHECK_INTERVAL  # 180 retries


class VirtualConnector(PlannerConnector):
    """
    This is a virtual connector for planner to output scaling decisions to non-native environments
    This virtual connector does not actually scale the deployment, instead, it communicates with the non-native environment through ETCD
    The deployment environment needs to read from ETCD to receive the scaling decisions and update ETCD to report scaling status
    The prefix for the ETCD key is /{dynamo_namespace}/planner/
    To output the scaling decisions, planner write to three keys:
       - num_prefill_workers: an integer (stored as string), specifying how many prefill workers the deployment should have in the last scaling decision
       - num_decode_workers: an integer (stored as string), specifying how many decode workers the deployment should have in the last scaling decision
       - decision_id: an integer (stored as string), specifying an incremental id for the last scaling decision, if there's no scaling decision, the value would be -1
    To receive the status of the scaling decisions, the deployment environment needs to write to this key:
       - scaled_decision_id: an integer (stored as string), specifying if the newest decision_id that has been scaled
    """

    def __init__(
        self, runtime: DistributedRuntime, dynamo_namespace: str, backend: str
    ):
        etcd_client = runtime.do_not_use_etcd_client()
        if etcd_client is None:
            raise RuntimeError("ETCD client is not initialized")

        self.backend = backend
        self.dynamo_namespace = dynamo_namespace
        self.worker_component_names = WORKER_COMPONENT_NAMES[backend]

        # Initialize current worker counts
        self.num_prefill_workers = 0
        self.num_decode_workers = 0
        self.decision_id = -1

        # Track when we first started skipping scaling due to unready state
        self.first_skip_timestamp: Optional[float] = None

        # Store etcd_client for async initialization
        self._etcd_client = etcd_client
        self._etcd_kv_cache = None

    async def _async_init(self):
        """Async initialization that must be called after __init__"""
        if self._etcd_kv_cache is not None:
            return  # Already initialized

        # Create EtcdKvCache with initial values
        initial_values = {
            "num_prefill_workers": str(self.num_prefill_workers).encode("utf-8"),
            "num_decode_workers": str(self.num_decode_workers).encode("utf-8"),
            "decision_id": str(self.decision_id).encode("utf-8"),
        }

        self._etcd_kv_cache = await EtcdKvCache.create(
            self._etcd_client,
            f"/{self.dynamo_namespace}/planner/",
            initial_values,
        )

        # Load current values from ETCD if they exist
        await self._load_current_state()

    @property
    def etcd_kv_cache(self):
        """Get the etcd_kv_cache, ensuring async initialization is complete"""
        if self._etcd_kv_cache is None:
            raise RuntimeError(
                "VirtualConnector not properly initialized. Call _async_init() first."
            )
        return self._etcd_kv_cache

    async def _load_current_state(self):
        """Load current state from ETCD"""
        # Get all current values
        all_values = await self.etcd_kv_cache.get_all()

        # Parse num_prefill_workers
        if "num_prefill_workers" in all_values:
            try:
                self.num_prefill_workers = int(
                    all_values["num_prefill_workers"].decode("utf-8")
                )
            except (ValueError, AttributeError):
                logger.warning(
                    "Failed to parse num_prefill_workers from ETCD, using default 0"
                )

        # Parse num_decode_workers
        if "num_decode_workers" in all_values:
            try:
                self.num_decode_workers = int(
                    all_values["num_decode_workers"].decode("utf-8")
                )
            except (ValueError, AttributeError):
                logger.warning(
                    "Failed to parse num_decode_workers from ETCD, using default 0"
                )

        # Parse decision_id
        if "decision_id" in all_values:
            try:
                self.decision_id = int(all_values["decision_id"].decode("utf-8"))
            except (ValueError, AttributeError):
                logger.warning(
                    "Failed to parse decision_id from ETCD, using default -1"
                )

    def _component_to_worker_type(self, component_name: str) -> Optional[str]:
        """Map component name to worker type (prefill or decode)"""
        if component_name == self.worker_component_names.prefill_worker_k8s_name:
            return "prefill"
        elif component_name == self.worker_component_names.decode_worker_k8s_name:
            return "decode"
        else:
            return None

    async def _is_scaling_ready(self) -> bool:
        """Check if the previous scaling decision has been completed"""
        # If this is the first decision, it's always ready
        if self.decision_id == -1:
            return True

        # Check if scaled_decision_id matches current decision_id
        scaled_decision_id_bytes = await self.etcd_kv_cache.get("scaled_decision_id")
        if scaled_decision_id_bytes:
            try:
                scaled_decision_id = int(scaled_decision_id_bytes.decode("utf-8"))
                return scaled_decision_id >= self.decision_id
            except (ValueError, AttributeError):
                logger.warning("Failed to parse scaled_decision_id from ETCD")

        # If no scaled_decision_id exists, assume not ready
        return False

    async def _update_scaling_decision(
        self, num_prefill: Optional[int] = None, num_decode: Optional[int] = None
    ):
        """Update scaling decision in ETCD"""
        # Check if there's actually a change
        prefill_changed = (
            num_prefill is not None and num_prefill != self.num_prefill_workers
        )
        decode_changed = (
            num_decode is not None and num_decode != self.num_decode_workers
        )

        if not prefill_changed and not decode_changed:
            logger.info(
                f"No scaling needed (prefill={self.num_prefill_workers}, decode={self.num_decode_workers}), skipping ETCD update"
            )
            return

        # Check if previous scaling is ready
        is_ready = await self._is_scaling_ready()

        if not is_ready:
            current_time = time.time()

            # If this is the first time we're skipping, record the timestamp
            if self.first_skip_timestamp is None:
                self.first_skip_timestamp = current_time
                logger.info(
                    f"Previous scaling decision #{self.decision_id} not ready, starting to track skip time"
                )

            # Check if we've been waiting too long
            if self.first_skip_timestamp is not None:
                time_waited = current_time - self.first_skip_timestamp
            else:
                # This should not happen since we just set it above, but for type safety
                time_waited = 0.0
            if time_waited < SCALING_MAX_WAIT_TIME:
                logger.warning(
                    f"Previous scaling decision #{self.decision_id} not ready, "
                    f"skipping new decision (waited {time_waited:.1f}s / {SCALING_MAX_WAIT_TIME}s)"
                )
                return
            else:
                logger.warning(
                    f"Previous scaling decision #{self.decision_id} not ready after {SCALING_MAX_WAIT_TIME}s, "
                    f"proceeding with new decision anyway"
                )

        # Reset the skip timestamp since we're making a decision
        self.first_skip_timestamp = None

        # Update internal state
        if num_prefill is not None:
            self.num_prefill_workers = num_prefill
        if num_decode is not None:
            self.num_decode_workers = num_decode

        # Increment decision_id
        self.decision_id += 1

        # Write to ETCD
        await self.etcd_kv_cache.put(
            "num_prefill_workers", str(self.num_prefill_workers).encode("utf-8")
        )
        await self.etcd_kv_cache.put(
            "num_decode_workers", str(self.num_decode_workers).encode("utf-8")
        )
        await self.etcd_kv_cache.put(
            "decision_id", str(self.decision_id).encode("utf-8")
        )

        logger.info(
            f"Updated scaling decision #{self.decision_id}: prefill={self.num_prefill_workers}, decode={self.num_decode_workers}"
        )

    async def _wait_for_scaling_completion(self):
        """Wait for the deployment environment to report that scaling is complete"""
        for _ in range(SCALING_MAX_RETRIES):
            scaled_decision_id_bytes = await self.etcd_kv_cache.get(
                "scaled_decision_id"
            )
            if scaled_decision_id_bytes:
                try:
                    scaled_decision_id = int(scaled_decision_id_bytes.decode("utf-8"))
                    if scaled_decision_id >= self.decision_id:
                        logger.info(f"Scaling decision #{self.decision_id} completed")
                        return
                except (ValueError, AttributeError):
                    logger.warning("Failed to parse scaled_decision_id from ETCD")

            await asyncio.sleep(SCALING_CHECK_INTERVAL)

        logger.warning(
            f"Timeout waiting for scaling decision #{self.decision_id} to complete after {SCALING_MAX_WAIT_TIME}s"
        )

    async def add_component(self, component_name: str, blocking: bool = True):
        """Add a component by increasing its replica count by 1"""
        worker_type = self._component_to_worker_type(component_name)
        if worker_type is None:
            logger.warning(f"Unknown component name: {component_name}, skipping")
            return

        if worker_type == "prefill":
            await self._update_scaling_decision(
                num_prefill=self.num_prefill_workers + 1
            )
        elif worker_type == "decode":
            await self._update_scaling_decision(num_decode=self.num_decode_workers + 1)

        if blocking:
            await self._wait_for_scaling_completion()

    async def remove_component(self, component_name: str, blocking: bool = True):
        """Remove a component by decreasing its replica count by 1"""
        worker_type = self._component_to_worker_type(component_name)
        if worker_type is None:
            logger.warning(f"Unknown component name: {component_name}, skipping")
            return

        if worker_type == "prefill":
            new_count = max(0, self.num_prefill_workers - 1)
            await self._update_scaling_decision(num_prefill=new_count)
        elif worker_type == "decode":
            new_count = max(0, self.num_decode_workers - 1)
            await self._update_scaling_decision(num_decode=new_count)

        if blocking:
            await self._wait_for_scaling_completion()

    async def set_component_replicas(
        self, target_replicas: dict[str, int], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise ValueError("target_replicas cannot be empty")

        num_prefill = None
        num_decode = None

        for component_name, replicas in target_replicas.items():
            worker_type = self._component_to_worker_type(component_name)
            if worker_type is None:
                logger.warning(f"Unknown component name: {component_name}, skipping")
                continue

            if worker_type == "prefill":
                num_prefill = replicas
            elif worker_type == "decode":
                num_decode = replicas

        # Update scaling decision if there are any changes
        await self._update_scaling_decision(
            num_prefill=num_prefill, num_decode=num_decode
        )

        if blocking:
            await self._wait_for_scaling_completion()
