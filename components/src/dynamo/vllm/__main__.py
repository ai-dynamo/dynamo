# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re


def _isolate_failover_compile_cache() -> str | None:
    """Give each failover engine an independent writable compiler cache."""
    if "VLLM_CACHE_ROOT" in os.environ:
        return None
    shadow_mode = os.environ.get("DYN_GMS_FAILOVER_SHADOW_MODE", "0")
    if shadow_mode.strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    engine_id = os.environ.get("ENGINE_ID")
    if not engine_id:
        return None
    safe_engine_id = re.sub(r"[^A-Za-z0-9_.-]", "_", engine_id)
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_root = os.path.join(cache_home, "vllm-gms-failover", safe_engine_id)
    os.environ["VLLM_CACHE_ROOT"] = cache_root
    return cache_root


if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if __name__ == "__main__":
    from dynamo.common.gms_failover import configure_failover_nccl_environment

    configure_failover_nccl_environment()
    _isolate_failover_compile_cache()

    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    # Check before importing dynamo.vllm.main: restore standby mode must capture
    # env and hold without importing vLLM or constructing backend/runtime state.
    maybe_run_restore_standby_mode()

    from dynamo.vllm.main import main

    main()
