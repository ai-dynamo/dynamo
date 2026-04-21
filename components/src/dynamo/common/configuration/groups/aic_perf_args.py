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
)


class AicPerfConfigBase(ConfigBase):
    aic_backend: Optional[str]
    aic_system: Optional[str]
    aic_backend_version: Optional[str]
    aic_tp_size: int
    aic_model_path: Optional[str]

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
