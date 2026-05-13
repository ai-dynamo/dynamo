# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiler-side shim for AIConfiguratorPerfEstimator.

The real implementation has moved to
``dynamo.planner.monitoring.aic_estimator`` so the planner can run AIC
interpolation in-process at bootstrap time. This module re-exports the
estimator for back-compat with existing profiler callers.
"""

from dynamo.planner.monitoring.aic_estimator import AIConfiguratorPerfEstimator

__all__ = ["AIConfiguratorPerfEstimator"]
