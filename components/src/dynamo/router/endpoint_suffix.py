# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def apply_worker_suffix_to_endpoint(endpoint_path: str) -> str:
    """Apply DYN_NAMESPACE_WORKER_SUFFIX to endpoint namespace when present."""
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if not suffix:
        return endpoint_path
    if "." in suffix:
        return endpoint_path
    parts = endpoint_path.split(".")
    if len(parts) != 3:
        return endpoint_path
    namespace, component, endpoint = parts
    return f"{namespace}-{suffix}.{component}.{endpoint}"
