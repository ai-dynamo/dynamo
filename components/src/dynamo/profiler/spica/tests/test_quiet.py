# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``import dynamo.profiler.spica`` must quiet the jax/vizier import noise (see dynamo.profiler.spica._quiet)."""

import logging
import os

import dynamo.profiler.spica as spica


def test_import_spica_quiets_jax():
    # jax pinned to CPU (the GP-bandit is CPU-only) -> no GPU-probe warning
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert spica.__name__ == "dynamo.profiler.spica"
    # the noisy jax/absl loggers are raised to ERROR so a sweep doesn't spam warnings
    assert logging.getLogger("jax._src.xla_bridge").level == logging.ERROR
    assert logging.getLogger("absl").level == logging.ERROR
