# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Silence the jax / equinox / jaxopt import noise that Vizier's GP-bandit pulls in.

Vizier's GP optimizer runs on JAX. The default ``ai-dynamo[spica]`` extra is CPU-only; users can
install the matching JAX CUDA plugin separately for large sweeps. So we:

- pin ``JAX_PLATFORMS=cpu`` only when no CUDA plugin is installed, skipping jax's noisy
  *"An NVIDIA GPU may be present ... but a CUDA-enabled jaxlib is not installed. Falling back
  to cpu."* warning for the normal CPU install;
- leave the platform unset when ``jax_cuda12_plugin`` is present, allowing JAX to select GPU;
- quiet the ``jax`` / ``absl`` loggers (e.g. absl's *"Python 3.8+ is required"*);
- drop the jax / jaxlib / equinox / jaxopt ``DeprecationWarning``s and the
  *"JAXopt is no longer maintained"* notice.

Imported **first** from :mod:`dynamo.profiler.spica` so it runs before anything (lazily) imports vizier/jax.
An explicit ``JAX_PLATFORMS`` always wins.
"""

from __future__ import annotations

import logging
import os
import warnings
from importlib.util import find_spec

# Must precede the first ``import jax`` (Vizier imports it lazily during a sweep).
if "JAX_PLATFORMS" not in os.environ and find_spec("jax_cuda12_plugin") is None:
    os.environ["JAX_PLATFORMS"] = "cpu"

for _logger in ("jax", "jax._src.xla_bridge", "absl"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

for _module in (r"jax(\..*)?", r"jaxlib(\..*)?", r"equinox(\..*)?", r"jaxopt(\..*)?"):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=_module)
warnings.filterwarnings("ignore", message=".*JAXopt is no longer maintained.*")
