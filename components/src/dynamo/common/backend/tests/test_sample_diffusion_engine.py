# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conformance tests for the raw-media (DiffusionEngine) pipeline, driven by
the CPU-only ``SampleDiffusionEngine``. Mirrors ``test_sample_engine.py`` for
the token path: it pins the structural guarantees a ``RawEngine`` must hold
(``EngineConfig.llm is None``, raw dict-in/dict-out, ``RawEngine`` identity for
adapter routing) without a GPU or model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo.common.backend.engine import RawEngine  # noqa: E402
from dynamo.common.backend.health_check import is_probe  # noqa: E402
from dynamo.common.backend.sample_diffusion_engine import (  # noqa: E402
    SampleDiffusionEngine,
)
from dynamo.llm import ModelInput  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _ctx() -> MagicMock:
    """Minimal Context stand-in. The sample diffusion engine ignores the
    context (single short step, no cancellation)."""
    return MagicMock()


async def _collect(engine: SampleDiffusionEngine, request: dict) -> list[dict]:
    return [chunk async for chunk in engine.generate(request, _ctx())]


async def test_start_returns_engine_config_with_no_llm_registration():
    """The defining structural guarantee for a RawEngine: ``start`` returns an
    ``EngineConfig`` whose ``llm`` sub-record is ``None`` (no KV/DP metadata to
    advertise). This is the raw-side analog of the token engines'
    registration-metadata check."""
    engine = SampleDiffusionEngine(delay=0.0)
    cfg = await engine.start(0)
    assert cfg.llm is None
    assert cfg.model == "sample-diffusion-model"
    assert cfg.served_model_name == "sample-diffusion-model"


async def test_engine_is_a_rawengine_so_routing_uses_raw_adapter():
    """Worker routes to the raw JSON adapter via ``isinstance(engine,
    RawEngine)``; a DiffusionEngine must satisfy that check."""
    assert isinstance(SampleDiffusionEngine(delay=0.0), RawEngine)


async def test_generate_yields_single_terminal_response():
    engine = SampleDiffusionEngine(delay=0.0)
    chunks = await _collect(engine, {"prompt": "a cat", "response_format": "b64_json"})
    assert len(chunks) == 1
    assert "data" in chunks[0] and len(chunks[0]["data"]) == 1


async def test_generate_honors_n():
    engine = SampleDiffusionEngine(delay=0.0)
    chunks = await _collect(
        engine, {"prompt": "x", "n": 3, "response_format": "b64_json"}
    )
    assert len(chunks[0]["data"]) == 3


async def test_generate_defaults_missing_n_to_one():
    engine = SampleDiffusionEngine(delay=0.0)
    chunks = await _collect(engine, {"prompt": "x", "response_format": "b64_json"})
    assert len(chunks[0]["data"]) == 1


async def test_generate_rejects_out_of_range_n():
    """``n`` is validated to the OpenAI 1..10 range — out-of-range values
    (including a falsy ``0``) raise rather than silently coercing to 1."""
    engine = SampleDiffusionEngine(delay=0.0)
    for bad_n in (0, 11):
        with pytest.raises(ValueError, match="between 1 and 10"):
            await _collect(
                engine, {"prompt": "x", "n": bad_n, "response_format": "b64_json"}
            )


async def test_response_format_b64_json_returns_base64_data():
    engine = SampleDiffusionEngine(delay=0.0)
    chunks = await _collect(engine, {"prompt": "x", "response_format": "b64_json"})
    assert "b64_json" in chunks[0]["data"][0]


async def test_response_format_defaults_to_self_contained_url():
    engine = SampleDiffusionEngine(delay=0.0)
    chunks = await _collect(engine, {"prompt": "x"})
    url = chunks[0]["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")


async def test_health_check_payload_is_a_probe_that_generate_completes():
    """The canary payload must be detectable by ``is_probe`` (so generate can
    bypass coordination) and must drive ``generate`` to a terminal response."""
    engine = SampleDiffusionEngine(delay=0.0)
    payload = await engine.health_check_payload()
    assert is_probe(payload)
    chunks = await _collect(engine, payload)
    assert len(chunks) == 1 and "data" in chunks[0]


async def test_from_args_configures_a_raw_media_worker():
    engine, worker_config = await SampleDiffusionEngine.from_args([])
    assert isinstance(engine, RawEngine)
    # Raw media: forwarded verbatim (no tokenizer stage) and no KV routing.
    assert worker_config.model_input == ModelInput.Text
    assert worker_config.enable_kv_routing is False


async def test_cleanup_is_noop_and_idempotent():
    engine = SampleDiffusionEngine(delay=0.0)
    await engine.start(0)
    await engine.cleanup()
    await engine.cleanup()  # second call must be safe
