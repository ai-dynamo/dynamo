# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang slice of the DYN_ENABLE_TEST_LOGITS_PROCESSOR
hook: the scheduler-side `DynamoSglangLogitProcessor` (batch-row application
+ per-request state keyed by UID) and the per-request `activate` helper that
wires entries into `custom_params` / the `custom_logit_processor` kwarg.

The shared spec-entry policy (gating, serialization) is tested in
`dynamo.common.backend.tests.test_engine`. These tests exercise the SGLang
realizer with CPU tensors, so they need `sglang` + `torch` importable but no
GPU."""

from __future__ import annotations

import pytest

# Skip cleanly (not a collection error) when real SGLang is absent: pytest's
# collection puts components/src/dynamo on sys.path, so a bare `import sglang`
# resolves to the dynamo.sglang shadow (no `srt.sampling`). Probe the exact
# submodule the adapter needs so the shadow / missing-backend case skips.
torch = pytest.importorskip("torch")
pytest.importorskip("sglang.srt.sampling.custom_logit_processor")

from dynamo.common.backend.engine import (  # noqa: E402
    ForcedTokenSequenceSpec,
    serialize_logits_processor_entries,
)
from dynamo.sglang.logits_processing.adapter import (  # noqa: E402
    DynamoSglangLogitProcessor,
    activate_logits_processors,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _payload(token_ids, eos, uid):
    return {
        "dynamo_logits": serialize_logits_processor_entries(
            [ForcedTokenSequenceSpec(token_ids=token_ids, eos_token_id=eos)]
        ),
        "dynamo_uid": uid,
    }


# ---------------------------------------------------------------------------
# Per-request activation
# ---------------------------------------------------------------------------


def test_activate_no_op_on_empty_entries():
    sampling_params: dict = {"temperature": 0.0}
    kwargs = activate_logits_processors(sampling_params, [], request_uid="r1")
    assert kwargs == {}
    assert "custom_params" not in sampling_params


def test_activate_sets_custom_params_and_returns_processor_kwarg():
    sampling_params: dict = {}
    entries = [ForcedTokenSequenceSpec(token_ids=(5, 6), eos_token_id=2)]
    kwargs = activate_logits_processors(sampling_params, entries, request_uid="r1")

    # Top-level generate kwarg carries the serialized processor class.
    assert "custom_logit_processor" in kwargs
    assert isinstance(kwargs["custom_logit_processor"], str)
    # Per-request activation rides in sampling_params["custom_params"].
    custom = sampling_params["custom_params"]
    assert custom["dynamo_uid"] == "r1"
    assert custom["dynamo_logits"] == serialize_logits_processor_entries(entries)


def test_activate_preserves_existing_custom_params():
    """A caller-set custom_params entry must survive activation (merge, not
    replace), so the hook doesn't silently drop other request params."""
    sampling_params: dict = {"custom_params": {"other": 1}}
    entries = [ForcedTokenSequenceSpec(token_ids=(5,), eos_token_id=2)]
    activate_logits_processors(sampling_params, entries, request_uid="r1")
    custom = sampling_params["custom_params"]
    assert custom["other"] == 1
    assert custom["dynamo_uid"] == "r1"
    assert custom["dynamo_logits"] == serialize_logits_processor_entries(entries)


@pytest.mark.parametrize("requested_n", [None, 1, 3])
def test_activate_forces_n_to_one(requested_n):
    """SGLang expands n>1 into batch rows sharing the request's custom_params,
    which would collide the per-request processor state — so the hook pins n=1."""
    sampling_params: dict = {}
    if requested_n is not None:
        sampling_params["n"] = requested_n
    entries = [ForcedTokenSequenceSpec(token_ids=(5,), eos_token_id=2)]
    activate_logits_processors(sampling_params, entries, request_uid="r1")
    assert sampling_params["n"] == 1


@pytest.mark.parametrize("bad_uid", [None, ""])
def test_activate_rejects_missing_uid(bad_uid):
    """A missing/empty uid would collide per-request state across requests
    (the engine must pass context.id(), not the optional trace_id), so it is
    rejected loudly rather than silently mis-tracked."""
    sampling_params: dict = {}
    entries = [ForcedTokenSequenceSpec(token_ids=(5,), eos_token_id=2)]
    with pytest.raises(ValueError):
        activate_logits_processors(sampling_params, entries, request_uid=bad_uid)
    # sampling_params is left untouched on rejection.
    assert "custom_params" not in sampling_params


# ---------------------------------------------------------------------------
# Scheduler-side batch processor
# ---------------------------------------------------------------------------


def test_call_no_op_without_custom_params():
    proc = DynamoSglangLogitProcessor()
    logits = torch.zeros((1, 8))
    out = proc(logits, None)
    assert torch.equal(out, torch.zeros((1, 8)))


def test_call_applies_only_to_rows_with_payload():
    """Row 0 activates a forced sequence; row 1 has no payload and is left
    untouched. Confirms batch rows map 1:1 to custom_param_list."""
    proc = DynamoSglangLogitProcessor()
    logits = torch.zeros((2, 8))
    custom_param_list = [_payload((5,), 2, "row0"), {}]

    proc(logits, custom_param_list)

    assert torch.argmax(logits[0]).item() == 5  # forced to token 5
    assert torch.equal(logits[1], torch.zeros(8))  # untouched


def test_call_advances_per_request_state_by_uid():
    """The scheduler caches one processor instance, so per-UID state must
    advance across successive batches: token 5, then 6, then EOS (2)."""
    proc = DynamoSglangLogitProcessor()
    payload = _payload((5, 6), 2, "req-a")

    logits = torch.zeros((1, 8))
    proc(logits, [payload])
    assert torch.argmax(logits[0]).item() == 5

    logits = torch.zeros((1, 8))
    proc(logits, [payload])
    assert torch.argmax(logits[0]).item() == 6

    logits = torch.zeros((1, 8))
    proc(logits, [payload])
    assert torch.argmax(logits[0]).item() == 2  # sequence exhausted → EOS


def test_call_tracks_distinct_uids_independently():
    """Two requests in the same batch keep independent counters."""
    proc = DynamoSglangLogitProcessor()
    pa = _payload((5, 6), 2, "a")
    pb = _payload((5, 6), 2, "b")

    # Advance only request "a" once.
    proc(torch.zeros((1, 8)), [pa])

    # Next batch has both; "a" is on its 2nd step (token 6), "b" on its 1st (5).
    logits = torch.zeros((2, 8))
    proc(logits, [pa, pb])
    assert torch.argmax(logits[0]).item() == 6
    assert torch.argmax(logits[1]).item() == 5


def test_call_reused_uid_shares_state():
    """Pins WHY the uid must be request-unique: a *reused* uid (e.g. if the
    engine keyed on a shared/None trace_id instead of context.id()) makes a
    second request inherit the first's advanced counter — the forced sequence
    would start mid-stream. This documents the contract activate enforces."""
    proc = DynamoSglangLogitProcessor()
    payload = _payload((5, 6), 2, "shared")

    proc(torch.zeros((1, 8)), [payload])  # request A, step 0 → token 5
    # A "different" request reusing the same uid resumes A's state at token 6,
    # NOT a fresh token 5 — corruption. Hence request_uid must be unique.
    logits = torch.zeros((1, 8))
    proc(logits, [payload])
    assert torch.argmax(logits[0]).item() == 6
