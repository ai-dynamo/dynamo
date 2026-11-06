<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TokenSpeed Disaggregated Serving

This directory holds the launcher, checkpoint preparation script, and validator
for experimental LongCat-Flash prefill/decode disaggregation with KV-aware
routing on TokenSpeed.

For prerequisites, deployment steps, and configuration, see
[Disaggregated Serving with TokenSpeed](../../../docs/fern/pages/cli/disaggregated-serving/tokenspeed.md).

## Contents

| Path | Description |
|---|---|
| `launch_disagg.sh` | Starts the frontend, a prefill worker, or a decode worker |
| `prepare_longcat.py` | Adds the missing `model_type` to the LongCat-Flash checkpoint config |
| `validate_disagg.py` | Verifies generation, KV transfer, and cache-aware prefill routing |
| `tests/` | Unit tests for the validator |
