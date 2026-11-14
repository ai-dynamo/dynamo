# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIS perf model inputs that the tests of several components share.

The frontend, the routers and squeeze_evolve take the same AIS arguments; the
router section of a simulation config and the replay keywords take the same
configs. Importing this module does not import AISimulate, so a test module
that is collected without AISimulate can import it.
"""

import contextlib
import importlib
import re
import warnings
from collections.abc import Iterator
from typing import Any

ENV_VAR = "DYN_AIS_PERF_MODEL"
MODEL = "Qwen/Qwen3-0.6B"
IDENTITY = {"model": MODEL, "system": "h200_sxm", "backend": "vllm"}
PREFILL_ENTRY = {**IDENTITY, "worker_type": "prefill", "backend_version": "0.11.0"}
AGGREGATED_ENTRY = {**IDENTITY, "worker_type": "aggregated"}

AIS = ["--router-prefill-load-model", "ais"]
NOT_AIS = ["--router-prefill-load-model", "none"]
OLD_FLAT = [
    "--ais-backend",
    "vllm",
    "--ais-system",
    "h200_sxm",
    "--ais-model-path",
    MODEL,
]


def normalized(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``config`` as a router's perf config holds it.

    ``config`` is written out by the test. The defaults come from the
    AISimulate config class, not from the code under test.
    """
    # Loaded on first use, as dynamo.common.ais_perf_model loads it: test
    # modules that are collected without AISimulate import this module.
    sdk = importlib.import_module("aisimulate_core.sdk")
    return sdk.ForwardPassPerfModelConfig(**config).to_dict()


def expected(**keys: Any) -> dict[str, Any]:
    """Return what ``ais_perf_kwargs()`` gives for ``IDENTITY`` and ``keys``."""
    return {"config": normalized({**IDENTITY, "worker_type": "aggregated", **keys})}


def exactly(text: str) -> str:
    """Return a pattern that matches ``text`` and nothing more."""
    return "^" + re.escape(text) + "$"


@contextlib.contextmanager
def no_warning() -> Iterator[None]:
    """Turn a warning inside the block into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
