# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import filelock

from dynamo.planner.circusd import CircusController
from dynamo.planner.planner_connector import PlannerConnector
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class LocalConnector(PlannerConnector):
    def __init__(self, namespace: str, runtime: DistributedRuntime):
        """
        Initialize LocalConnector and connect to CircusController.

        Args:
            namespace: The Dynamo namespace
            runtime: Optional DistributedRuntime instance
        """
        self.namespace = namespace
        self.runtime = runtime
        self.state_file = Path.home() / ".dynamo" / "state" / f"{namespace}.json"
        self.circus = CircusController.from_state_file(namespace)
        self._lockfile = self.state_file.with_suffix(".lock")
        self._file_lock = filelock.FileLock(self._lockfile)
        self.worker_client: Any | None = None
        self.prefill_client: Any | None = None
        self.etcd_client: Any | None = None

    async def _load_state(self) -> Dict[str, Any]:
        """Load state from state file.

        Returns:
            State dictionary
        """
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file not found: {self.state_file}")

        with self._file_lock:
            with open(self.state_file, "r") as f:
                return json.load(f)

    async def _save_state(self, state: Dict[str, Any]) -> bool:
        """Save state to state file.

        Args:
            state: State dictionary to save

        Returns:
            True if successful
        """
        try:
            with self._file_lock:
                with open(self.state_file, "w") as f:
                    json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    async def _get_available_gpus(self) -> List[str]:
        """Get list of unallocated GPU IDs.

        Returns:
            List of available GPU IDs
        """
        state = await self._load_state()
        system_resources = state.get("environment", {}).get("SYSTEM_RESOURCES", {})
        all_gpus = set(str(gpu) for gpu in system_resources.get("gpu_info", []))

        allocated_gpus: set[str] = set()
        for component_info in state.get("components", {}).values():
            resources = component_info.get("resources", {})
            gpu_list = resources.get("allocated_gpus", [])
            allocated_gpus.update(str(gpu) for gpu in gpu_list)

        logger.info(f"Allocated GPUs: {allocated_gpus}")
        available = sorted(list(all_gpus - allocated_gpus))
        logger.info(f"Available GPUs: {available}")
        return available

    async def add_component(self, component_name: str, blocking: bool = True) -> bool:
        """
        Add a component. The steps are as follows:

        1. Load state
        2. Find max suffix to create unique watcher name
        3. Built environment and command for watcher
        4. Block until component is running

        Args:
            component_name: Name of the component

        Returns:
            True if successful
        """
        state = await self._load_state()
        # Find max suffix
        max_suffix = 0
        for watcher_name in state["components"].keys():
            if watcher_name.startswith(f"{self.namespace}_{component_name}_"):
                suffix = int(
                    watcher_name.replace(f"{self.namespace}_{component_name}_", "")
                )
                max_suffix = max(max_suffix, suffix)

        watcher_name = f"{self.namespace}_{component_name}_{max_suffix + 1}"

        if component_name not in [
            c.replace(f"{self.namespace}_", "") for c in state["components"]
        ]:
            raise ValueError(
                f"Component {component_name} not found in state configuration"
            )

        # Get base command and config
        component_info = state["components"][f"{self.namespace}_{component_name}"]
        base_cmd = component_info["cmd"].split("--worker-env")[0].strip()
        service_config = state["environment"].get("DYNAMO_SERVICE_CONFIG")

        # Build environment
        watcher_env = os.environ.copy()
        if component_name in ["VllmWorker", "PrefillWorker"]:
            available_gpus = await self._get_available_gpus()
            if not available_gpus:
                raise ValueError("No GPUs available for allocation")
            gpu_id = available_gpus[0]
            watcher_env["CUDA_VISIBLE_DEVICES"] = gpu_id

        watcher_env["DYNAMO_SERVICE_CONFIG"] = service_config

        # Build worker env list and command
        worker_env_list = [watcher_env]
        worker_env_arg = json.dumps(worker_env_list)
        # We add a custom component name to ensure that the lease is attatched to this specific watcher
        full_cmd = f"{base_cmd} --worker-env '{worker_env_arg}' --custom-component-name '{watcher_name}'"

        pre_add_endpoint_ids = await self._get_endpoint_ids(component_name)
        logger.info(f"Pre-add endpoint IDs: {pre_add_endpoint_ids}")

        logger.info(f"Adding watcher {watcher_name}")
        success = await self.circus.add_watcher(
            name=watcher_name, cmd=full_cmd, env=watcher_env, singleton=True
        )

        if success:
            resources = {}
            if component_name in ["VllmWorker", "PrefillWorker"]:
                resources["allocated_gpus"] = [gpu_id]

            state["components"][watcher_name] = {
                "watcher_name": watcher_name,
                "cmd": full_cmd,
                "resources": resources,
            }
            await self._save_state(state)
            logger.info(
                f"Succesfully created {watcher_name}. Waiting for worker to start..."
            )

        if blocking:
            required_endpoint_ids = pre_add_endpoint_ids + 1
            while True:
                current_endpoint_ids = await self._get_endpoint_ids(component_name)
                if current_endpoint_ids == required_endpoint_ids:
                    break
                logger.info(
                    f"Waiting for {component_name} to start. Current endpoint IDs: {current_endpoint_ids}, Required endpoint IDs: {required_endpoint_ids}"
                )
                await asyncio.sleep(5)

        return success

    async def remove_component(
        self, component_name: str, blocking: bool = True
    ) -> bool:
        """
        Remove a component. The initial components are not numbered so we simply remove their resources
        and lease but keep the entry in order to use the cmd. This allows us to re-add the component
        without having to re-specify the cmd. For components that have been added, we remove their entry
        entry

        Args:
            component_name: Name of the component

        Returns:
            True if successful
        """
        logger.info(f"Attempting to remove component {component_name}")
        state = await self._load_state()
        matching_components = {}

        base_name = f"{self.namespace}_{component_name}"
        base_name_with_underscore = f"{base_name}_"

        for watcher_name in state["components"].keys():
            if watcher_name == base_name:
                matching_components[0] = watcher_name
            elif watcher_name.startswith(base_name_with_underscore):
                suffix = int(watcher_name.replace(base_name_with_underscore, ""))
                matching_components[suffix] = watcher_name

        if not matching_components:
            logger.error(f"No matching components found for {component_name}")
            return False

        highest_suffix = max(matching_components.keys())
        target_watcher = matching_components[highest_suffix]
        logger.info(f"Removing watcher {target_watcher}")

        pre_remove_endpoint_ids = await self._get_endpoint_ids(component_name)

        if component_name == "VllmWorker" or component_name == "PrefillWorker":
            lease_id = state["components"][target_watcher]["lease"]
            await self._revoke_lease(lease_id)

            # Poll endpoint to ensure that worker has shut down gracefully and then remove the watcher
            if blocking:
                required_endpoint_ids = pre_remove_endpoint_ids - 1
                while True:
                    current_endpoint_ids = await self._get_endpoint_ids(component_name)
                    if current_endpoint_ids == required_endpoint_ids:
                        break
                    logger.info(
                        f"Waiting for {component_name} to shutdown. Current endpoint IDs: {current_endpoint_ids}, Required endpoint IDs: {required_endpoint_ids}"
                    )
                    await asyncio.sleep(5)

        success = await self.circus.remove_watcher(name=target_watcher)
        logger.info(
            f"Circus remove_watcher for {target_watcher} {'succeeded' if success else 'failed'}"
        )

        if success:
            if highest_suffix > 0:  # Numbered watcher - remove entire entry
                if target_watcher in state["components"]:
                    del state["components"][target_watcher]
            else:  # Base watcher - just clear resources and lease
                if target_watcher in state["components"]:
                    state["components"][target_watcher]["resources"] = {}
                    state["components"][target_watcher]["lease"] = None
            await self._save_state(state)

        return success

    async def _get_endpoint_ids(self, component_name: str) -> int:
        """
        Get the endpoint IDs for a component.

        Args:
            component_name: Name of the component

        Returns:
            Number of endpoint IDs for a component
        """
        if component_name == "VllmWorker":
            if self.worker_client is None:
                self.worker_client = (
                    await self.runtime.namespace(self.namespace)
                    .component(component_name)
                    .endpoint("generate")
                    .client()
                )
            worker_ids = self.worker_client.endpoint_ids()
            return len(worker_ids)
        elif component_name == "PrefillWorker":
            if self.prefill_client is None:
                self.prefill_client = (
                    await self.runtime.namespace(self.namespace)
                    .component(component_name)
                    .endpoint("mock")
                    .client()
                )
            prefill_ids = self.prefill_client.endpoint_ids()
            return len(prefill_ids)
        else:
            raise ValueError(f"Component {component_name} not supported")

    async def _revoke_lease(self, lease_id: int) -> bool:
        """
        Wrapper function around the etcd client to revoke a lease

        Args:
            lease_id: Lease ID to revoke

        Returns:
            True if successful
        """
        if self.etcd_client is None:
            self.etcd_client = self.runtime.etcd_client()  # type: ignore
        try:
            await self.etcd_client.revoke_lease(lease_id)
            logger.info(f"Revoked lease {lease_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke lease {lease_id}: {e}")
            return False

    def __del__(self):
        """Cleanup circus controller connection on deletion."""
        if hasattr(self, "circus"):
            self.circus.close()
