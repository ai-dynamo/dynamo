# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example backend: a sample backend implementation for Dynamo.

This backend returns a fixed "Hello World!" reply streamed token by token.
It exists as a minimal reference for writing new backends.
"""

from dynamo.example_backend.args import ExampleBackendConfig
from dynamo.example_backend.backend import ExampleBackend
from dynamo.example_backend.handlers import ExampleHandler

__all__ = ["ExampleBackend", "ExampleBackendConfig", "ExampleHandler"]
