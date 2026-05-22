# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost-eval decision service.

Hosts Planner regression models behind a NATS request/reply endpoint that the
KV router's RegressionConditionalPrefillPolicy queries on its slow path. See
``components/src/dynamo/cost_eval/service.py`` for the moving parts and the
top-level ``regressionpolicy_implementation.md`` doc for the design context.
"""
