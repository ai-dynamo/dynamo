# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo backend tests.

Handles conditional test collection to prevent import errors when backend
frameworks are not installed in the current container.
"""

import importlib.util


def pytest_ignore_collect(collection_path, config):
    """Skip collecting backend test files if their framework isn't installed.

    Checks for backend test directories and corresponding installed packages.
    """
    path_str = str(collection_path)

    # Map backend test paths to required modules
    backend_requirements = {
        "/vllm/tests/": "vllm",
        "/sglang/tests/": "sglang",
        "/trtllm/tests/": "tensorrt_llm",
    }

    for path_pattern, required_module in backend_requirements.items():
        if path_pattern in path_str:
            print(f"[DEBUG] Found backend test path: {path_str}")
            print(f"[DEBUG] Looking for module: {required_module}")

            spec = importlib.util.find_spec(required_module)
            print(f"[DEBUG] find_spec('{required_module}') = {spec}")

            if spec is not None:
                print(f"[DEBUG] Module origin: {spec.origin}")
                print(
                    f"[DEBUG] Module submodule_search_locations: {spec.submodule_search_locations}"
                )

                # Check if this is dynamo's wrapper or the real package
                if spec.origin and "/dynamo/" in spec.origin:
                    print(
                        f"[DEBUG] WARNING: Found dynamo's wrapper package at {spec.origin}"
                    )
                    print("[DEBUG] This may cause import errors!")

            if spec is None:
                print("[DEBUG] Module not found - SKIPPING collection")
                return True  # Module not available, skip this file
            else:
                print(
                    "[DEBUG] Module found - COLLECTING (but may fail if it's the wrapper)"
                )

    return None  # Not a backend test or module available
