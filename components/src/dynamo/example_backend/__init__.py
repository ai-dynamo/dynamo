# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example backend: a sample backend implementation for Dynamo.

This backend returns a fixed "Hello World!" reply streamed token by token.
It exists as a minimal reference for writing new backends.
"""

__all__ = ["ExampleBackend", "ExampleHandler"]
