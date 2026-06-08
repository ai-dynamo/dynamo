# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared AIC perf-model configuration ArgGroup."""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument

_AIC_PERF_FIELDS: tuple[str, ...] = (
    "aic_backend",
    "aic_system",
    "aic_backend_version",
    "aic_tp_size",
    "aic_model_path",
    "aic_moe_tp_size",
    "aic_moe_ep_size",
    "aic_attention_dp_size",
    "aic_nextn",
    "aic_nextn_accept_rates",
)


class AicPerfConfigBase(ConfigBase):
    aic_backend: Optional[str]
    aic_system: Optional[str]
    aic_backend_version: Optional[str]
    aic_tp_size: int
    aic_model_path: Optional[str]
    aic_moe_tp_size: Optional[int]
    aic_moe_ep_size: Optional[int]
    aic_attention_dp_size: Optional[int]
    aic_nextn: Optional[int]
    aic_nextn_accept_rates: Optional[str]

    def aic_perf_kwargs(self) -> dict:
        return {field: getattr(self, field) for field in _AIC_PERF_FIELDS}


class AicPerfArgGroup(ArgGroup):
    def add_arguments(self, parser) -> None:
        g = parser.add_argument_group("AIC Perf Model Options")

        add_argument(
            g,
            flag_name="--aic-backend",
            env_var="DYN_AIC_BACKEND",
            default=None,
            help=(
                "[EXPERIMENTAL] AIC backend family to model "
                "(for example: vllm or sglang)."
            ),
        )
        add_argument(
            g,
            flag_name="--aic-system",
            env_var="DYN_AIC_SYSTEM",
            default=None,
            help=(
                "[EXPERIMENTAL] AIC hardware/system identifier "
                "(for example: h200_sxm)."
            ),
        )
        add_argument(
            g,
            flag_name="--aic-backend-version",
            env_var="DYN_AIC_BACKEND_VERSION",
            default=None,
            help="[EXPERIMENTAL] Pinned backend version for AIC database lookup.",
        )
        add_argument(
            g,
            flag_name="--aic-tp-size",
            env_var="DYN_AIC_TP_SIZE",
            default=1,
            help="[EXPERIMENTAL] Tensor parallel size to model in AIC.",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--aic-model-path",
            env_var="DYN_AIC_MODEL_PATH",
            default=None,
            help=(
                "[EXPERIMENTAL] Model path or model identifier to use for "
                "AIC perf lookup."
            ),
        )
        add_argument(
            g,
            flag_name="--aic-moe-tp-size",
            env_var="DYN_AIC_MOE_TP_SIZE",
            default=None,
            help=(
                "[EXPERIMENTAL] MoE tensor-parallel size to model in AIC. "
                "Required by some MoE models."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--aic-moe-ep-size",
            env_var="DYN_AIC_MOE_EP_SIZE",
            default=None,
            help=(
                "[EXPERIMENTAL] MoE expert-parallel size to model in AIC. "
                "Required by some MoE models."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--aic-attention-dp-size",
            env_var="DYN_AIC_ATTENTION_DP_SIZE",
            default=None,
            help="[EXPERIMENTAL] Attention data-parallel size to model in AIC.",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--aic-nextn",
            env_var="DYN_AIC_NEXTN",
            default=None,
            help=(
                "[EXPERIMENTAL] MTP/Eagle speculative-decoding draft-token count "
                "for AIC latency modeling (max 5). Omit to disable spec dec."
            ),
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--aic-nextn-accept-rates",
            env_var="DYN_AIC_NEXTN_ACCEPT_RATES",
            default=None,
            help=(
                "[EXPERIMENTAL] Comma-separated per-position accept rates for MTP "
                "draft tokens (e.g. '0.85,0.3,0,0,0'). Padded to length 5."
            ),
        )
