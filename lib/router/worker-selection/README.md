<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-worker-selection

`dynamo-worker-selection` picks one worker rank for a request from a table of candidates. A policy is a chain of filters and scorers followed by a picker, each of which a plugin can supply; the crate also holds the candidate rows and views those plugins read, conditional disaggregation (whether a request needs a separate prefill worker), and the LoRA filter. Plugins reach it through `dynamo_kv_router::plugins::prelude`.
