# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import Any


def _load_aiconfigurator_modules() -> tuple[Any, Any, Any]:
    try:
        common = importlib.import_module("aiconfigurator.sdk.common")
        task = importlib.import_module("aiconfigurator.sdk.task")
        utils = importlib.import_module("aiconfigurator.sdk.utils")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "aiconfigurator is required to enumerate dense TP candidates for replay optimization"
        ) from exc
    return common, task, utils


def _enumerate_dense_tp_candidates(
    backend: str, system: str
) -> tuple[list[int], list[int]]:
    common, task, utils = _load_aiconfigurator_modules()
    backend_enum = getattr(common.BackendName, backend)
    prefill_cfg, decode_cfg = task.build_disagg_parallel_lists(
        backend_name=backend,
        prefill_system=system,
        decode_system=system,
        is_moe=False,
        should_enable_pp=False,
    )

    prefill_parallel = utils.enumerate_parallel_config(
        num_gpu_list=prefill_cfg["num_gpu_per_worker"],
        tp_list=prefill_cfg["tp_list"],
        pp_list=prefill_cfg["pp_list"],
        dp_list=prefill_cfg["dp_list"],
        moe_tp_list=prefill_cfg["moe_tp_list"],
        moe_ep_list=prefill_cfg["moe_ep_list"],
        is_moe=False,
        backend=backend_enum,
    )
    decode_parallel = utils.enumerate_parallel_config(
        num_gpu_list=decode_cfg["num_gpu_per_worker"],
        tp_list=decode_cfg["tp_list"],
        pp_list=decode_cfg["pp_list"],
        dp_list=decode_cfg["dp_list"],
        moe_tp_list=decode_cfg["moe_tp_list"],
        moe_ep_list=decode_cfg["moe_ep_list"],
        is_moe=False,
        backend=backend_enum,
    )

    def extract_tp(parallel_configs: list[list[int]]) -> list[int]:
        return sorted(
            {
                tp
                for tp, pp, dp, moe_tp, moe_ep in parallel_configs
                if pp == 1 and dp == 1 and moe_tp == 1 and moe_ep == 1
            }
        )

    return extract_tp(prefill_parallel), extract_tp(decode_parallel)
