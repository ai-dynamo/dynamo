# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned-routing campaign harness (offline replay evaluation, tuning, reporting).

``HARNESS_VERSION`` is part of every result-cache key. Bump it whenever the replay invocation,
any metric definition in :mod:`learned_routing.goodput`, or the E0 method
(:data:`learned_routing.e0.METHOD`) changes, so stale cached records are never reused.

- ``lrh-4``: E0 decode context ``ISL + j + 2`` (``ais-chunked-estimator-v2``) and a 1e-6 relative
  tolerance on the E2E clause (build audit goodput r2 F1).
"""

HARNESS_VERSION = "lrh-4"
