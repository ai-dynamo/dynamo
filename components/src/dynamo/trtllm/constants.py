# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class DisaggregationMode(Enum):
    AGGREGATED = "prefill_and_decode"
    PREFILL = "prefill"
    DECODE = "decode"
    ENCODE = "encode"
