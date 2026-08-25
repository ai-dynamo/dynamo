# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in DGDR v1beta2 integration surface.

Per the Spica/AI-Simulate-Sweeper design: "V2 should land as an opt-in
implementation... V2 should not change the existing V1 profiler defaults or
k8s controller/operator path." Code under this package reuses the v1
CONFIG_MODIFIERS/DGD-template machinery in components/src/dynamo/profiler
rather than duplicating it, but is otherwise additive and independent of the
v1 rapid.py/thorough.py entry points.
"""
