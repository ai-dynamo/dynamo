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
import sys
from typing import Literal

import pytest

from dynamo.planner import LocalConnector
from dynamo.runtime import DistributedRuntime, dynamo_worker

pytestmark = pytest.mark.skip("This is not a test file")

ComponentType = Literal["VllmWorker", "PrefillWorker"]
VALID_COMPONENTS = ["VllmWorker", "PrefillWorker"]


async def test_state_management(connector: LocalConnector) -> bool:
    """Test state file operations."""
    print("\n=== Testing State Management ===")
    try:
        # Test load state
        state = await connector._load_state()
        print("✓ Load state successful")

        # Test save state (with a copy)
        success = await connector._save_state(state)
        print(
            f"{'✓' if success else '✗'} Save state {'successful' if success else 'failed'}"
        )

        return True
    except Exception as e:
        print(f"✗ State management test failed: {e}")
        return False


async def test_add_component(
    connector: LocalConnector, component: ComponentType
) -> bool:
    """Test adding a component."""
    print(f"\n=== Testing Add Component: {component} ===")
    try:
        success = await connector.add_component(component)
        print(
            f"{'✓' if success else '✗'} Add {component} {'successful' if success else 'failed'}"
        )
        return success
    except Exception as e:
        print(f"✗ Add {component} test failed: {e}")
        return False


async def test_remove_component(
    connector: LocalConnector, component: ComponentType
) -> bool:
    """Test removing a component."""
    print(f"\n=== Testing Remove Component: {component} ===")
    try:
        state = await connector._load_state()
        base_name = f"{connector.namespace}_{component}_"

        # Find all components with numbered suffixes
        matching_components = []
        for watcher_name in state["components"].keys():
            if watcher_name.startswith(base_name):
                try:
                    suffix = int(watcher_name.replace(base_name, ""))
                    matching_components.append((suffix, watcher_name))
                except ValueError:
                    continue

        if not matching_components:
            base_component = f"{connector.namespace}_{component}"
            if base_component in state["components"]:
                success = await connector.remove_component(component)
                print(
                    f"{'✓' if success else '✗'} Remove {component} {'successful' if success else 'failed'}"
                )
                return success
            else:
                print(f"✗ No {component} components found to remove")
                return False

        # Remember which watcher we're removing
        highest_suffix = max(suffix for suffix, _ in matching_components)
        target_component = f"{base_name}{highest_suffix}"

        success = await connector.remove_component(component)

        # New verification logic that handles both numbered and base watchers
        if success:
            new_state = await connector._load_state()

            # For numbered watchers (with suffix > 0)
            if highest_suffix > 0:
                # Success if the component is completely removed
                if target_component not in new_state["components"]:
                    print(f"✓ Successfully removed {target_component}")
                    return True
                else:
                    print(f"✗ Failed to remove {target_component} from state")
                    return False
            # For base watchers (no suffix)
            else:
                base_component = f"{connector.namespace}_{component}"
                if base_component in new_state["components"]:
                    resources = new_state["components"][base_component].get(
                        "resources", {}
                    )
                    if not resources.get("allocated_gpus"):
                        print(f"✓ Successfully cleared resources for {base_component}")
                        return True
                    else:
                        print(f"✗ Failed to clear resources for {base_component}")
                        return False

            # If we get here, neither condition was met
            print(f"✗ Unexpected state after removing {component}")
            return False

        print(f"✗ Failed to remove {component}")
        return False

    except Exception as e:
        print(f"✗ Remove {component} test failed: {e}")
        return False


@dynamo_worker()
async def main(runtime: DistributedRuntime):
    connector = LocalConnector("dynamo", runtime)

    await connector.add_component("PrefillWorker")
    await connector.add_component("VllmWorker")
    await connector.remove_component("VllmWorker")
    await connector.remove_component("PrefillWorker")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
