# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-side handle + per-engine adapters (SGLang, vLLM)."""

from gms_kv_ring.engines.handle import GMSKvRing

__all__ = ["GMSKvRing"]
