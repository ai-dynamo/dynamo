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

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from circus.client import CircusClient
from circus.exc import CallError

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

    async def _list_watchers(self) -> List[str]:
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
            return []

    def close(self) -> None:
        """Close the connection to the arbiter."""
        if hasattr(self, "client"):
            self.client.stop()
