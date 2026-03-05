# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helpers to build AIC-compatible DataFrames from real-GPU benchmark results.

The picking functions in ``aiconfigurator.sdk.picking`` expect DataFrames
whose columns match the ``ColumnsStatic`` schema.  Only a subset of columns
are actually accessed; this module populates exactly those columns.
"""

from __future__ import annotations

import pandas as pd
from aiconfigurator.sdk import common
from aiconfigurator.sdk.picking import _build_disagg_summary_dict


def make_parallel_label(tp: int, pp: int, dp: int, moe_tp: int, moe_ep: int) -> str:
    """Build the ``parallel`` label string used for dedup in picking."""
    if moe_ep > 1:
        return f"dep{moe_ep}"
    elif moe_tp > 1:
        return f"tep{moe_tp}"
    else:
        return f"tp{tp}"


def build_prefill_row(
    *,
    model: str,
    isl: int,
    osl: int,
    ttft: float,
    tp: int,
    pp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
    backend: str = "",
    system: str = "",
) -> dict:
    """Build a single prefill row dict with the minimal columns needed by AIC picking.

    Only columns actually accessed by ``pick_autoscale`` and
    ``_build_disagg_summary_dict`` are populated.
    """
    num_gpus = tp * pp * dp
    seq_s = 1000.0 / ttft * dp if ttft > 0 else 0.0

    return {
        "ttft": ttft,
        "seq/s": seq_s,
        "seq/s/gpu": seq_s / num_gpus if num_gpus > 0 else 0.0,
        "global_bs": 1 * dp,
        "parallel": make_parallel_label(tp, pp, dp, moe_tp, moe_ep),
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "osl": osl,
        "model": model,
        "isl": isl,
        "bs": 1,
        "moe_tp": moe_tp,
        "moe_ep": moe_ep,
        "prefix": 0,
        "gemm": "",
        "kvcache": "",
        "fmha": "",
        "moe": "",
        "comm": "",
        "memory": "",
        "backend": backend,
        "version": "",
        "system": system,
        "power_w": 0.0,
    }


def build_decode_row(
    *,
    tpot: float,
    thpt_per_gpu: float,
    num_request: int,
    num_gpus: int,
    osl: int,
    tp: int,
    pp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
    backend: str = "",
    system: str = "",
) -> dict:
    """Build a single decode row dict with the minimal columns needed by AIC picking.

    Only columns actually accessed by ``pick_autoscale`` and
    ``_build_disagg_summary_dict`` are populated.
    """
    seq_s = thpt_per_gpu * num_gpus / osl if osl > 0 else 0.0

    return {
        "tpot": tpot,
        "seq/s": seq_s,
        "seq/s/gpu": thpt_per_gpu / osl if osl > 0 else 0.0,
        "global_bs": num_request,
        "parallel": make_parallel_label(tp, pp, dp, moe_tp, moe_ep),
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "concurrency": num_request,
        "bs": num_request // dp if dp > 0 else num_request,
        "tokens/s/user": 1000.0 / tpot if tpot > 0 else 0.0,
        "moe_tp": moe_tp,
        "moe_ep": moe_ep,
        "gemm": "",
        "kvcache": "",
        "fmha": "",
        "moe": "",
        "comm": "",
        "memory": "",
        "backend": backend,
        "version": "",
        "system": system,
        "power_w": 0.0,
    }


def build_disagg_df_from_static(
    prefill_df: pd.DataFrame,
    decode_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-product prefill x decode into a ColumnsDisagg DataFrame.

    Used when calling ``pick_default`` or ``pick_load_match`` from
    THOROUGH-mode benchmark results.
    """
    combos: list[dict] = []
    for _, p_row in prefill_df.iterrows():
        for _, d_row in decode_df.iterrows():
            combo = _build_disagg_summary_dict(
                prefill_summary_dict=p_row.to_dict(),
                prefill_num_worker=1,
                decode_summary_dict=d_row.to_dict(),
                decode_num_worker=1,
            )
            combos.append(combo)
    if not combos:
        return pd.DataFrame(columns=common.ColumnsDisagg)
    return pd.DataFrame(combos, columns=common.ColumnsDisagg)
