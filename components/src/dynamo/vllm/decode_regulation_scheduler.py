# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DecodeRegulationScheduler -- InstrumentedScheduler with per-step decode regulation.

The policy and the skip mechanism live in ``decode_regulation`` (no vLLM import, unit
tested on a fake scheduler); this class binds them to vLLM's scheduler. Forward
pass metrics keep working: skipped requests are simply not in a step's
``SchedulerOutput``, and ``decode_regulation`` log lines report what was skipped.

Selected automatically when ``DYN_DECODE_REGULATION_MODE`` is not ``off``
(see ``args.py``), or via:
    --scheduler-cls "dynamo.vllm.decode_regulation_scheduler.DecodeRegulationScheduler"
"""

from __future__ import annotations

import logging

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from dynamo.vllm.decode_regulation import (
    DecodeRegulationConfig,
    DecodeRegulationMixin,
    vllm_skip_unsupported,
)
from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler

logger = logging.getLogger(__name__)


class DecodeRegulationScheduler(DecodeRegulationMixin, InstrumentedScheduler):
    def __init__(self, vllm_config, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        unsupported = []
        skip_unsupported = vllm_skip_unsupported(Scheduler, Request)
        if skip_unsupported:
            unsupported.append(skip_unsupported)
        if getattr(self, "use_pp", False):
            unsupported.append(
                "pipeline parallelism (placeholders may be non-zero on rejoin)"
            )
        if getattr(vllm_config, "speculative_config", None) is not None:
            unsupported.append(
                "speculative decoding (skipped requests keep draft placeholders)"
            )
        model = getattr(
            getattr(vllm_config, "model_config", None), "served_model_name", ""
        )
        self._reg_init(
            DecodeRegulationConfig.from_env(),
            unsupported="; ".join(unsupported),
            model_name=model if isinstance(model, str) else str(model),
            # InstrumentedScheduler resolves the rank vLLM zeroes under external DP.
            dp_rank=self._fpm_dp_rank,
        )
