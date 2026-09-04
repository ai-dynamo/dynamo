# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform providers: the small set of things a substrate must be able to do.

Ordered by what the existing suite actually uses, not by what is easiest to
write: local subprocesses are the substrate of most tests in ``tests/``,
Kubernetes is next, and Docker is used by roughly one.
"""

from .local import LocalProvider, LocalRole

__all__ = ["LocalProvider", "LocalRole"]
