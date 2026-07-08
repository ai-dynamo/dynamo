# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gpu_memory_service.integrations.vllm.install_kv_leases import (
    install_gms_engine_core_sleep,
)


def test_sleep_utility_is_visible_on_spawned_engine_core_proc():
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    original = EngineCore.__dict__.get("gms_sleep_no_clear")
    if original is not None:
        delattr(EngineCore, "gms_sleep_no_clear")
    try:
        assert install_gms_engine_core_sleep()
        assert "gms_sleep_no_clear" in EngineCore.__dict__
        assert hasattr(EngineCoreProc, "gms_sleep_no_clear")
        assert not install_gms_engine_core_sleep()
    finally:
        if hasattr(EngineCore, "gms_sleep_no_clear"):
            delattr(EngineCore, "gms_sleep_no_clear")
        if original is not None:
            EngineCore.gms_sleep_no_clear = original
