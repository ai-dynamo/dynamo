#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

try:
    from dynamo._core import MockEngineArgs as MockEngineArgs
    from dynamo._core import PlannerReplayBridge as PlannerReplayBridge
    from dynamo._core import ReasoningConfig as ReasoningConfig
    from dynamo._core import SglangArgs as SglangArgs
    from dynamo._core import TrtllmArgs as TrtllmArgs
    from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
except ImportError:
    # The Rust extension is provided by ai-dynamo-runtime. Keep importing the
    # package itself cheap in static tooling environments where _core is absent.
    pass
else:
    __all__.extend(
        [
            "MockEngineArgs",
            "PlannerReplayBridge",
            "ReasoningConfig",
            "SglangArgs",
            "TrtllmArgs",
            "run_mocker_trace_replay",
        ]
    )

    def run_mocker_trace_replay(
        trace_file,
        extra_engine_args=None,
        router_config=None,
        num_workers=1,
        replay_concurrency=None,
        router_mode="round_robin",
        arrival_speedup_ratio=1.0,
        trace_block_size=512,
        trace_format="mooncake",
        trace_shared_prefix_ratio=0.0,
        trace_num_prefix_groups=0,
        sla_ttft_ms=None,
        sla_itl_ms=None,
        sla_e2e_ms=None,
    ):
        return _run_mocker_trace_replay(
            trace_file,
            extra_engine_args=extra_engine_args,
            router_config=router_config,
            num_workers=num_workers,
            replay_concurrency=replay_concurrency,
            replay_mode="offline",
            router_mode=router_mode,
            arrival_speedup_ratio=arrival_speedup_ratio,
            trace_block_size=trace_block_size,
            trace_format=trace_format,
            trace_shared_prefix_ratio=trace_shared_prefix_ratio,
            trace_num_prefix_groups=trace_num_prefix_groups,
            # Goodput SLA (offline only): report carries goodput_* when set.
            sla_ttft_ms=sla_ttft_ms,
            sla_itl_ms=sla_itl_ms,
            sla_e2e_ms=sla_e2e_ms,
        )


try:
    from dynamo._core import AicEngineConfig as AicEngineConfig
    from dynamo._core import EngineCapacity as EngineCapacity
    from dynamo._core import EngineCapacityRequest as EngineCapacityRequest
    from dynamo._core import EnginePerfLimits as EnginePerfLimits
    from dynamo._core import OptimizationTarget as OptimizationTarget
    from dynamo._core import RustEnginePerfModel as RustEnginePerfModel
    from dynamo._core import RustEnginePerfOptions as RustEnginePerfOptions
except ImportError:
    # These classes are available only when the Python extension is built with
    # the optional aic-forward-pass Cargo feature.
    pass
else:
    __all__.extend(
        [
            "AicEngineConfig",
            "EngineCapacity",
            "EngineCapacityRequest",
            "EnginePerfLimits",
            "OptimizationTarget",
            "RustEnginePerfModel",
            "RustEnginePerfOptions",
        ]
    )
