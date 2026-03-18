# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Skip collection of tests in this directory - they require dependencies
# not available in the main dynamo test environment.
collect_ignore = [
    "test_synthesizer.py",
    "test_sampler.py",
    "test_roundtrip_hashes.py",
    "mock_server.py",  # Not a test, just a utility
]
