# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure the JAX/Vizier runtime when a Spica sampler is constructed.

Vizier's GP optimizer runs on JAX. The default ``ai-dynamo[spica]`` extra is CPU-only; users can
install the matching JAX CUDA plugin separately for large sweeps. So we:

- pin ``JAX_PLATFORMS=cpu`` only when no CUDA plugin is installed, skipping jax's noisy
  *"An NVIDIA GPU may be present ... but a CUDA-enabled jaxlib is not installed. Falling back
  to cpu."* warning for the normal CPU install;
- leave the platform unset when ``jax_cuda12_plugin`` is present, allowing JAX to select GPU;
- quiet the ``jax`` / ``absl`` loggers (e.g. absl's *"Python 3.8+ is required"*);
- drop the jax / jaxlib / equinox / jaxopt ``DeprecationWarning``s and the
  *"JAXopt is no longer maintained"* notice.

The package root deliberately does not import or execute this module: importing
``dynamo.profiler.spica`` must not mutate process-wide environment, logging, or
warning state. :class:`~dynamo.profiler.spica.sampler.VizierBranchSampler` calls
:func:`configure_vizier_runtime` immediately before importing Vizier. An explicit
``JAX_PLATFORMS`` always wins.
"""

from __future__ import annotations

import logging
import os
import warnings
from importlib.util import find_spec


def configure_vizier_runtime() -> None:
    """Apply Spica's opt-in JAX/Vizier process configuration.

    Warning filters are installed on every sampler construction because test runners
    and embedding applications may reset the warning filter list between sweeps.
    """

    # Must precede the first ``import jax`` (Vizier imports it lazily during a sweep).
    if "JAX_PLATFORMS" not in os.environ and find_spec("jax_cuda12_plugin") is None:
        os.environ["JAX_PLATFORMS"] = "cpu"

    for logger_name in ("jax", "jax._src.xla_bridge", "absl"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    for module in (
        r"jax(\..*)?",
        r"jaxlib(\..*)?",
        r"equinox(\..*)?",
        r"jaxopt(\..*)?",
    ):
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=module)
    warnings.filterwarnings("ignore", message=".*JAXopt is no longer maintained.*")
    # google-vizier 0.1.21 calls protobuf's Python 3.12-deprecated UTC helper.
    warnings.filterwarnings(
        "ignore",
        message=r"datetime\.datetime\.utcnow\(\) is deprecated.*",
        category=DeprecationWarning,
        module=r"google\.protobuf\.internal\.well_known_types",
    )
