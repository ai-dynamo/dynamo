# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIS perf model inputs that the tests of the router family share.

The frontend, the routers and squeeze_evolve take the same AIS inputs from one
argument group. This is a plain module, not a conftest: a test file imports
what it needs, the fixture included.
"""

import contextlib
import os
import re
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

MODEL = "Qwen/Qwen3-0.6B"
IDENTITY = {"model": MODEL, "system": "h200_sxm", "backend": "vllm"}
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
NOTHING_GIVEN = (
    "--router-prefill-load-model=ais requires an AIS perf model: "
    "set --ais-perf-model or DYN_AIS_PERF_MODEL"
)

_ENV_PREFIXES = ("DYN_AIS_", "DYN_AIC_", "DYN_ROUTER_PREFILL_LOAD_")


@pytest.fixture
def no_ais_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep AIS inputs of the shell that runs the tests out of a test."""
    for name in list(os.environ):
        # A squeeze_evolve tier carries ais_perf_model.
        if name.startswith(_ENV_PREFIXES) or name == "DYN_SQUEEZE_TIERS":
            monkeypatch.delenv(name)


def expected(**keys: Any) -> dict[str, Any]:
    """Return what ``ais_perf_kwargs()`` gives for ``IDENTITY`` and ``keys``."""
    # Imported here: a test file that takes only the fixture from this module
    # has tests that run without AISimulate.
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    config = {**IDENTITY, "worker_type": "aggregated", **keys}
    return {"config": ForwardPassPerfModelConfig(**config).to_dict()}


def exactly(text: str) -> str:
    """Return a pattern that matches ``text`` and nothing more."""
    return "^" + re.escape(text) + "$"


@contextlib.contextmanager
def no_warning() -> Iterator[None]:
    """Turn a warning inside the block into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
