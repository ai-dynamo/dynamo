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
from typing import Any, Dict, List, Optional

from circus.client import CircusClient
from circus.exc import CallError

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class CircusController:
    """A circus client implementation for Dynamo"""

    def __init__(self, endpoint: str):
        """Initialize connection to arbiter.

        Args:
            endpoint: The circus endpoint (e.g., tcp://127.0.0.1:54927)
        """
        self.endpoint = endpoint
        self.client = CircusClient(endpoint=endpoint, timeout=15.0)

    @classmethod
    def from_state_file(cls, namespace: str) -> "CircusController":
        """
        Create a CircusController from a Dynamo state file.

        Args:
            namespace: The Dynamo namespace

        Returns:
            CircusController instance

        Raises:
            FileNotFoundError: If state file doesn't exist
            ValueError: If no endpoint found in state file
        """
        state_file = (
            Path(
                os.environ.get("DYN_LOCAL_STATE_DIR", Path.home() / ".dynamo" / "state")
            )
            / f"{namespace}.json"
        )
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        with open(state_file, "r") as f:
            state = json.load(f)

        endpoint = state.get("circus_endpoint")
        if not endpoint:
            raise ValueError(f"No endpoint found in state file: {state_file}")

        return cls(endpoint)

    async def add_watcher(
        self,
        name: str,
        cmd: str,
        env: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **options: Any,
    ) -> bool:
        """
        Add a new watcher to circus

        Args:
            name: Name of the watcher
            cmd: Command to run
            env: Environment variables
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            **options: Additional watcher options

        Returns:
            True if successful, False otherwise
        """
        watcher_options: dict[str, Any] = {
            "copy_env": True,
            "stop_children": True,
            "graceful_timeout": 86400,
            "respawn": False,
        }
        if env:
            watcher_options["env"] = env
        watcher_options.update(options)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2**attempt)
                    logger.info(
                        f"Retrying add_watcher for {name} (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)

                response = self.client.send_message(
                    "add",
                    name=name,
                    cmd=cmd,
                    args=[],
                    options=watcher_options,
                    start=True,
                )

                if response.get("status") == "ok":
                    logger.info(
                        f"Successfully added watcher {name} on attempt {attempt + 1}"
                    )
                    return True

                logger.error(
                    f"Failed to add watcher {name}: {response.get('reason', 'unknown error')}"
                )
                return False
            except Exception as e:
                if "arbiter is already running" in str(e):
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to remove watcher {name} after {max_retries} attempts: arbiter busy"
                        )
                        return False
                    logger.warning(
                        f"Arbiter busy with manage_watchers command, will retry removing watcher {name}"
                    )
                    continue

                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to add watcher {name} after {max_retries} attempts: {e}"
                    )
                    return False
                logger.warning(f"Error adding watcher {name}: {e}")

        return False

    async def remove_watcher(
        self,
        name: str,
        nostop: bool = False,
        waiting: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 600,  # 10 minutes
    ) -> bool:
        """
        Terminate processes and remove a watcher

        Args:
            name: The name of the watcher to remove
            nostop: Whether to stop the processes or not
            waiting: Whether to wait for completion
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if successful, False otherwise
        """
        exited = await self._wait_for_process_graceful_exit(name, timeout)
        if not exited:
            logger.error(
                f"Process for {name} did not exit gracefully. Proceeding with forced removal."
            )

        logger.info(f"Removing watcher {name}")
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = retry_delay * (2**attempt)
                    logger.info(
                        f"Retrying remove_watcher for {name} (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)

                response = self.client.send_message(
                    "rm",
                    name=name,
                    nostop=nostop,
                    waiting=waiting,
                )

                if response.get("status") == "ok":
                    logger.info(
                        f"Successfully removed watcher {name} on attempt {attempt + 1}"
                    )
                    break

                logger.error(f"Failed to remove watcher {name}: {response}")
                return False
            except Exception as e:
                if "arbiter is already running" in str(e):
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to remove watcher {name} after {max_retries} attempts: arbiter busy"
                        )
                        return False
                    logger.warning(
                        f"Arbiter busy with manage_watchers command, will retry removing watcher {name}"
                    )
                    continue

                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to remove watcher {name} after {max_retries} attempts: {e}"
                    )
                    return False

        # Verify the watcher is actually gone
        removed = await self._verify_watcher_removal(name)
        if not removed:
            logger.error(f"Watcher {name} still exists after {max_retries} attempts")
            return False

        return True

    async def _wait_for_process_graceful_exit(
        self, name: str, timeout: int = 600
    ) -> bool:
        """
        Wait for a watcher's process to exit gracefully. This is usually called after
        we've revoked the lease which triggers a graceful exit.

        Args:
            name: The name of the watcher
            timeout: The timeout for the wait

        Returns:
            True if the process exited gracefully, False otherwise
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"Timeout ({timeout}s) reached waiting for {name} to exit gracefully. "
                    f"Proceeding with forced removal."
                )
                return False

            num_processes = await self._get_watcher_processes(name)
            if num_processes is None:
                logger.error(f"Failed to get process count for {name}")
                return False
            if num_processes == 0:
                logger.info(f"Processes for {name} have exited gracefully")
                return True

            logger.info(
                f"Currently {num_processes} alive, waiting for it to exit gracefully "
                f"({int(elapsed)}s/{timeout}s elapsed)"
            )
            await asyncio.sleep(1)

    async def _verify_watcher_removal(
        self, name: str, max_attempts: int = 10, delay: float = 1.0
    ) -> bool:
        """
        Verify that a watcher has been removed. This is usually called after a forced removal.

        Args:
            name: The name of the watcher
            max_attempts: The maximum number of attempts to verify the watcher removal
            delay: The delay between attempts in seconds

        Returns:
            True if the watcher has been removed, False otherwise
        """
        for attempt in range(max_attempts):
            watchers = await self._list_watchers()
            if watchers is None:
                logger.error("Failed to list watchers")
                return False

            if name not in watchers:
                logger.info(f"Verified watcher {name} has been removed")
                return True

            logger.info(
                f"Waiting for watcher {name} to be fully removed (attempt {attempt + 1}/{max_attempts})"
            )
            await asyncio.sleep(delay)

        logger.error(
            f"Watcher {name} still exists after {max_attempts} verification attempts"
        )
        return False

    async def _get_watcher_processes(self, name: str) -> Optional[int]:
        """
        Get number of processes for a watcher.

        Args:
            name: The name of the watcher

        Returns:
            Number of processes for the watcher. Returns None operation fails.
        """
        try:
            response = self.client.send_message("numprocesses", name=name)
            return int(response.get("numprocesses", 0))
        except (CallError, Exception) as e:
            logger.error(f"Failed to get process count for {name}: {e}")
            return None

    async def _list_watchers(self) -> Optional[List[str]]:
        """
        List all watchers managed by circus.

        Returns:
            List of watcher names. Returns None if the list operation fails.
        """
        try:
            response = self.client.send_message("list")
            return response.get("watchers", [])
        except (CallError, Exception) as e:
            logger.error(f"Failed to list watchers: {e}")
            return None

    def close(self) -> None:
        """Close the connection to the arbiter."""
        if hasattr(self, "client"):
            self.client.stop()
