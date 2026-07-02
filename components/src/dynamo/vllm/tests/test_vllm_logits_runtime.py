# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM slice of the DYN_ENABLE_TEST_LOGITS_PROCESSOR
hook: the engine-loaded `DynamoVllmLogitsProcessor`, the per-request
activation that stashes serialized spec entries onto `SamplingParams`, and
the startup registration helper.

The shared spec-entry policy (generation-stage gating, serialization) is
tested in `dynamo.common.backend.tests.test_engine`. These tests exercise
the vLLM realizer in isolation with mocked `SamplingParams`, so they need
`vllm` + `torch` importable but no GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Skip cleanly (not a collection error) when real vLLM is absent: pytest's
# collection puts components/src/dynamo on sys.path, so a bare `import vllm`
# resolves to the dynamo.vllm shadow (no `v1.sample`). Probe the exact
# submodule the adapter needs so the shadow / missing-backend case skips.
torch = pytest.importorskip("torch")
pytest.importorskip("vllm.v1.sample.logits_processor")

from dynamo.common.backend.engine import (  # noqa: E402
    ForcedTokenSequenceSpec,
    serialize_logits_processor_entries,
)
from dynamo.vllm.logits_processing.adapter import (  # noqa: E402
    DYNAMO_VLLM_LOGITS_PROCESSOR_PATH,
    DynamoVllmLogitsProcessor,
    activate_logits_processors,
    register_dynamo_logits_processor,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _new_processor() -> DynamoVllmLogitsProcessor:
    """Build the adapter without running `AdapterLogitsProcessor.__init__`
    (which wants a real `vllm_config` / device). The methods under test
    (`is_argmax_invariant`, `new_req_logits_processor`) hold no instance
    state, so bypassing the base ctor keeps the test free of a GPU engine."""
    return DynamoVllmLogitsProcessor.__new__(DynamoVllmLogitsProcessor)


# ---------------------------------------------------------------------------
# Startup registration
# ---------------------------------------------------------------------------


def test_register_appends_adapter_path():
    engine_args = SimpleNamespace(logits_processors=None)
    register_dynamo_logits_processor(engine_args)
    assert engine_args.logits_processors == [DYNAMO_VLLM_LOGITS_PROCESSOR_PATH]


def test_register_is_idempotent_and_preserves_existing():
    engine_args = SimpleNamespace(logits_processors=["some.other:Processor"])
    register_dynamo_logits_processor(engine_args)
    register_dynamo_logits_processor(engine_args)
    assert engine_args.logits_processors == [
        "some.other:Processor",
        DYNAMO_VLLM_LOGITS_PROCESSOR_PATH,
    ]


# ---------------------------------------------------------------------------
# Per-request activation
# ---------------------------------------------------------------------------


def test_activate_no_op_on_empty_entries():
    sampling_params = SimpleNamespace(extra_args=None)
    activate_logits_processors(sampling_params, [])
    assert sampling_params.extra_args is None


def test_activate_stashes_serialized_entries():
    sampling_params = SimpleNamespace(extra_args=None)
    entries = [ForcedTokenSequenceSpec(token_ids=(5, 6), eos_token_id=2)]
    activate_logits_processors(sampling_params, entries)
    assert sampling_params.extra_args["dynamo_logits"] == (
        serialize_logits_processor_entries(entries)
    )


def test_activate_clears_stale_payload_on_empty_entries():
    """A no-activation request clears any pre-existing dynamo_logits (e.g. a
    client-supplied vllm_xargs) so the engine-loaded adapter is not triggered,
    while leaving other extra_args untouched."""
    sampling_params = SimpleNamespace(
        extra_args={
            "dynamo_logits": [
                {"kind": "forced_sequence", "token_ids": [1], "eos_token_id": 2}
            ],
            "kv_transfer_params": {"x": 1},
        }
    )
    activate_logits_processors(sampling_params, [])
    assert "dynamo_logits" not in sampling_params.extra_args
    assert sampling_params.extra_args["kv_transfer_params"] == {"x": 1}


def test_activate_preserves_existing_extra_args():
    sampling_params = SimpleNamespace(extra_args={"kv_transfer_params": {"x": 1}})
    activate_logits_processors(
        sampling_params, [ForcedTokenSequenceSpec(token_ids=(5,), eos_token_id=2)]
    )
    assert sampling_params.extra_args["kv_transfer_params"] == {"x": 1}
    assert "dynamo_logits" in sampling_params.extra_args


# ---------------------------------------------------------------------------
# Engine-loaded adapter
# ---------------------------------------------------------------------------


def test_is_not_argmax_invariant():
    # Forced-sequence masking changes the argmax, so vLLM must apply it
    # even on greedy requests.
    assert _new_processor().is_argmax_invariant() is False


def test_new_req_logits_processor_returns_none_without_activation():
    proc = _new_processor()
    assert proc.new_req_logits_processor(SimpleNamespace(extra_args=None)) is None
    assert proc.new_req_logits_processor(SimpleNamespace(extra_args={})) is None


def test_new_req_logits_processor_applies_forced_sequence_with_state():
    """The returned request callable forces token_ids[0], then token_ids[1],
    then EOS — advancing per-request state across decode steps and returning
    the mutated 1-D logits tensor (vLLM's request-callable contract)."""
    proc = _new_processor()
    entries = [ForcedTokenSequenceSpec(token_ids=(5, 6), eos_token_id=2)]
    sampling_params = SimpleNamespace(extra_args=None)
    activate_logits_processors(sampling_params, entries)

    req_proc = proc.new_req_logits_processor(sampling_params)
    assert req_proc is not None

    vocab = 8
    # Step 0 → force token 5.
    logits = torch.zeros(vocab)
    out = req_proc([], logits)
    assert out is logits  # mutated in place, returned
    assert torch.argmax(logits).item() == 5
    # Step 1 → force token 6.
    logits = torch.zeros(vocab)
    req_proc([5], logits)
    assert torch.argmax(logits).item() == 6
    # Step 2 (sequence exhausted) → force EOS (2).
    logits = torch.zeros(vocab)
    req_proc([5, 6], logits)
    assert torch.argmax(logits).item() == 2


def test_distinct_requests_get_independent_state():
    """Each `new_req_logits_processor` call realizes fresh processors, so
    two concurrent requests never share the forced-sequence counter."""
    proc = _new_processor()
    sp = SimpleNamespace(extra_args=None)
    activate_logits_processors(
        sp, [ForcedTokenSequenceSpec(token_ids=(5, 6), eos_token_id=2)]
    )
    req_a = proc.new_req_logits_processor(sp)
    req_b = proc.new_req_logits_processor(sp)

    # Advance A by one step; B must still start at token 5.
    req_a([], torch.zeros(8))
    logits_b = torch.zeros(8)
    req_b([], logits_b)
    assert torch.argmax(logits_b).item() == 5
