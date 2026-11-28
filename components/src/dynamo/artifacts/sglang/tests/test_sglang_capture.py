# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("msgspec")
pytest.importorskip("zstandard")

from dynamo.artifacts.sglang.capture import (  # noqa: E402 -- optional dependency guards
    LIST_BYTES_PER_VALUE,
    SglangArtifactSession,
    hidden_state_profile_from_sglang,
    sglang_hidden_rows,
)
from dynamo.artifacts.vllm.capture import (  # noqa: E402 -- optional dependency guards
    CaptureBudget,
    CaptureCapacityError,
)
from dynamo.artifacts.vllm.format import (  # noqa: E402 -- optional dependency guards
    HiddenStateChunks,
    HiddenStates,
    assemble_hidden_states,
    decode_generation_artifact,
)
from dynamo.artifacts.vllm.hidden_states import HiddenStateProfile  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

LAYERS = (3, 6)
HIDDEN = 4
WIDTH = len(LAYERS) * HIDDEN


def _server_args(**overrides):
    values = {
        "enable_return_aux_hidden_states": True,
        "speculative_algorithm": "EAGLE3",
        "return_aux_hidden_state_layer_ids": [5, 2],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _model_config(dtype="torch.bfloat16", hidden_size=HIDDEN):
    return SimpleNamespace(dtype=dtype, hidden_size=hidden_size)


def test_profile_reports_layer_inputs_in_capture_order():
    profile = hidden_state_profile_from_sglang(_server_args(), _model_config())
    # SGLang's explicit ids name layer outputs; the artifact names layer inputs.
    assert profile.layer_ids == LAYERS
    assert profile.runtime_data() == {
        "layer_ids": [3, 6],
        "hidden_size": HIDDEN,
        "dtype": "bf16",
    }


def test_profile_is_none_without_capture():
    args = _server_args(enable_return_aux_hidden_states=False)
    assert hidden_state_profile_from_sglang(args, _model_config()) is None


@pytest.mark.parametrize(
    "args, config, match",
    [
        (_server_args(speculative_algorithm="DFLASH"), _model_config(), "EAGLE3"),
        (
            _server_args(return_aux_hidden_state_layer_ids=None),
            _model_config(),
            "--return-aux-hidden-state-layer-ids",
        ),
        (_server_args(), _model_config(dtype="torch.float8_e4m3fn"), "dtype"),
        (_server_args(), _model_config(hidden_size=None), "hidden size"),
    ],
)
def test_misconfigured_capture_fails_at_startup(args, config, match):
    with pytest.raises(ValueError, match=match):
        hidden_state_profile_from_sglang(args, config)


def _bf16_values(rows, seed=0):
    values = np.random.default_rng(seed).standard_normal((rows, WIDTH))
    bits = values.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def test_rows_flatten_prefill_blocks_and_decode_rows():
    values = _bf16_values(4)
    raw = [values[:3].tolist(), values[3].tolist()]
    count, data = sglang_hidden_rows(raw, width=WIDTH, dtype="bf16")
    assert count == 4
    expected = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    assert data == expected


def test_rows_convert_fp16_and_fp32_exactly():
    values = np.random.default_rng(1).standard_normal((2, WIDTH))
    half = values.astype(np.float16)
    count, data = sglang_hidden_rows(
        [half.astype(np.float32).tolist()], width=WIDTH, dtype="fp16"
    )
    assert (count, data) == (2, half.tobytes())
    single = values.astype(np.float32)
    assert sglang_hidden_rows([single.tolist()], width=WIDTH, dtype="fp32") == (
        2,
        single.tobytes(),
    )


@pytest.mark.parametrize(
    "raw, dtype, match",
    [
        ([[0.1] * WIDTH], "bf16", "not bf16"),
        ([[0.1] * WIDTH], "fp16", "not fp16"),
        ([[1.0] * (WIDTH - 1)], "fp32", "geometry"),
        ([[[1.0] * WIDTH, [1.0] * (WIDTH - 1)]], "fp32", "layout"),
        ([[]], "fp32", "layout"),
        ([3.0], "fp32", "layout"),
        (None, "fp32", "no hidden states"),
    ],
)
def test_unexpected_rows_are_rejected(raw, dtype, match):
    with pytest.raises(ValueError, match=match):
        sglang_hidden_rows(raw, width=WIDTH, dtype=dtype)


def _profile(dtype="bf16"):
    return HiddenStateProfile(LAYERS, HIDDEN, dtype, storage_root="")


def _request(mode="inline_base64", destination=None, options=None, contents=None):
    settings = {
        "kind": "generation_artifact",
        "contents": list(contents or ("hidden_states",)),
        "mode": mode,
    }
    if destination is not None:
        settings["destination"] = destination
    if options is not None:
        settings["content_options"] = {"hidden_states": options}
    return {"extra_args": {"nvext": {"metadata_upload": settings}}}


def _session(req=None, budget=None, prompt=3, max_tokens=3, **kwargs):
    return SglangArtifactSession.from_request(
        req or _request(),
        prompt_token_count=prompt,
        max_tokens=max_tokens,
        budget=budget or CaptureBudget(1 << 24),
        hidden_state_profile=kwargs.pop("profile", _profile()),
        **kwargs,
    )


def _run(capture, raw, *, prompt=(10, 11, 12), chunks=((13, 14),), retractions=0):
    for tokens in chunks:
        capture.observe_tokens(list(tokens), list(prompt))
    capture.observe_finish({"hidden_states": raw, "num_retractions": retractions})


def _decoded(result):
    data = base64.b64decode(result["data_base64"])
    return decode_generation_artifact(data).choices[0]


@pytest.mark.asyncio
async def test_inline_capture_matches_the_returned_rows():
    capture = _session()
    values = _bf16_values(4)
    # Prompt of 3, completion of 2: positions 0..3 were forward inputs.
    _run(capture, [values[:3].tolist(), values[3].tolist()])
    result = await capture.finalize()

    choice = _decoded(result)
    assert choice.sequence_token_ids.as_array().tolist() == [10, 11, 12, 13, 14]
    (component,) = choice.components
    assert isinstance(component, HiddenStates)
    assert component.token_start == 0
    assert component.layer_ids == LAYERS
    rows = np.frombuffer(component.hidden_states.data, dtype=np.uint16).reshape(
        4, WIDTH
    )
    assert rows.tobytes() == (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    assert capture.reserved_bytes == 0


@pytest.mark.asyncio
async def test_rows_after_a_cache_hit_still_cover_token_start():
    capture = _session(_request(options={"token_start": 3}))
    values = _bf16_values(4)
    # Positions 0..1 came from the prefix cache; SGLang returned 2..3.
    _run(capture, [values[2:3].tolist(), values[3].tolist()])
    (component,) = _decoded(await capture.finalize()).components
    assert component.token_start == 3
    rows = np.frombuffer(component.hidden_states.data, dtype=np.uint16).reshape(
        1, WIDTH
    )
    assert (
        rows.tobytes() == (values[3:].view(np.uint32) >> 16).astype(np.uint16).tobytes()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows, retractions",
    [
        (2, 0),  # suffix starts at 2, after token_start 0: cache or chunked prefill
        (5, 0),  # more rows than positions
        (4, 1),  # a retracted request may repeat rows
        (4, None),  # retraction count unknown
    ],
)
async def test_rows_that_cannot_be_aligned_fail_the_capture(rows, retractions):
    capture = _session()
    values = _bf16_values(rows)
    _run(capture, [values.tolist()], retractions=retractions)
    result = await capture.finalize()
    assert result["state"] == "failed"
    assert result["error_code"] == "capture_invalid"
    assert capture.reserved_bytes == 0


@pytest.mark.asyncio
async def test_missing_rows_fail_the_capture():
    capture = _session()
    capture.observe_tokens([13, 14], [10, 11, 12])
    capture.observe_finish({"num_retractions": 0})
    assert (await capture.finalize())["error_code"] == "capture_invalid"


def test_other_contents_are_rejected_before_admission():
    budget = CaptureBudget(1 << 24)
    with pytest.raises(ValueError, match="hidden_states only"):
        _session(_request(contents=("hidden_states", "selected_logprobs")), budget)
    assert budget.used == 0


def test_hidden_states_need_a_capturing_worker():
    with pytest.raises(ValueError, match="--enable-return-aux-hidden-states"):
        _session(profile=None)


def test_non_artifact_request_creates_no_session():
    assert _session({"extra_args": {}}) is None


def test_reservation_covers_python_float_lists_and_is_released():
    budget = CaptureBudget(1 << 24)
    capture = _session(budget=budget, prompt=3, max_tokens=3)
    assert budget.used >= 6 * WIDTH * LIST_BYTES_PER_VALUE
    capture.close()
    assert budget.used == 0


def test_list_reservation_failure_releases_the_base_reservation():
    base = _session(budget=CaptureBudget(1 << 24)).reserved_bytes
    lists = 6 * WIDTH * LIST_BYTES_PER_VALUE
    budget = CaptureBudget(base - lists + lists // 2)
    with pytest.raises(CaptureCapacityError):
        _session(budget=budget)
    assert budget.used == 0


@pytest.mark.asyncio
async def test_closed_session_ignores_late_rows():
    capture = _session()
    capture.close()
    capture.observe_finish({"hidden_states": [[1.0] * WIDTH], "num_retractions": 0})
    assert capture.hidden_rows is None


class _Runtime:
    max_artifact_bytes = 1 << 30

    def __init__(self):
        self.submitted = []

    def validate(self, destination, mode):
        pass

    async def submit(self, data, contents, destination, *, mode, parts=None):
        self.submitted.append((data, parts))
        return {"kind": "generation_artifact", "mode": mode, "state": "pending"}


@pytest.mark.asyncio
async def test_chunked_capture_sends_rows_as_parts():
    runtime = _Runtime()
    capture = _session(
        _request(
            mode="object_store",
            destination={
                "kind": "managed_store",
                "profile": "approved",
                "object_key": "runs/r1/sample-0",
                "chunk_prefix": "runs/r1/chunks",
            },
            options={"chunk_tokens": 2},
        ),
        runtime=runtime,
    )
    values = _bf16_values(4)
    _run(capture, [values[:3].tolist(), values[3].tolist()])
    assert (await capture.finalize())["state"] == "pending"

    ((data, parts),) = runtime.submitted
    (component,) = decode_generation_artifact(data).choices[0].components
    assert isinstance(component, HiddenStateChunks)
    assert [chunk.rows for chunk in component.chunks] == [2, 2]
    assembled = assemble_hidden_states(component, parts.__getitem__)
    assert (
        assembled.tobytes()
        == (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    )
