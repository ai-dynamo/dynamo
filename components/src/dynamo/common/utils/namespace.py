# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional


def get_worker_namespace(namespace: Optional[str] = None) -> str:
    """Get the Dynamo namespace for a worker.

    Uses the provided namespace, or falls back to the DYN_NAMESPACE environment
    variable (defaulting to "dynamo"). If DYN_NAMESPACE_WORKER_SUFFIX is set,
    it is appended as "{namespace}-{suffix}" to support multiple sets of workers
    for the same model.
    """
    if not namespace:
        namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace
