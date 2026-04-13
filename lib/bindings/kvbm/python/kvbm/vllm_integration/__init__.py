# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Back-compat shim — `kvbm.vllm_integration.*` resolves to the v1 implementation
# under `kvbm.v1.vllm_integration.*` (Option A, see ACTIVE_PLAN.md phase 1).
# Kept minimal so pytest collection on a vllm-free host doesn't transitively
# import vllm via the v1 connector module.
