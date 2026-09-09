# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The tokenizer round-trip suite downloads external data and is run explicitly.
# CPU-only synthesis and sampling regressions are collected normally.
collect_ignore = [
    "test_roundtrip_hashes.py",
    "mock_server.py",  # Not a test, just a utility
]
