# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional


def get_spec_decode_runtime_data(server_args: Any) -> Optional[dict[str, Any]]:
    try:
        nextn = int(getattr(server_args, "speculative_num_steps", 0) or 0)
    except (TypeError, ValueError):
        return None
    if nextn <= 0:
        return None

    data: dict[str, Any] = {"nextn": nextn, "source": "backend_config"}
    method = getattr(server_args, "speculative_algorithm", None)
    if method:
        data["method"] = str(method)
    return data
