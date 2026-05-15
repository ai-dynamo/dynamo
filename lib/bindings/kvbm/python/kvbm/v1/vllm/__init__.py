# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical v1 vLLM façade namespace.

The connector entry point lives at ``kvbm.v1.vllm.connector`` and mirrors
``kvbm.v2.vllm.connector`` with a single-character path difference. The
implementation substrate is still ``kvbm.v1.vllm_integration.connector``;
this façade is a lazy re-export, identical in spirit to the legacy
``kvbm.vllm_integration.connector`` shim from phase 1.
"""
