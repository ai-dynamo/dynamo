# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release-pinned adapter boundaries, not full N-2 serving conformance.

Run in each pinned fixture's engine image and the current engine image. Old
reader tests execute exact released function excerpts with real SamplingParams
and released logprob parsers. Guided decoding, RL, discovery, Rust serialization,
engine verification/generation and disaggregated execution are not exercised.
Current-reader tests replay the old writer's extracted field vocabulary; they
do not execute the old Rust writer. No historical Dynamo bindings are loaded.
"""

import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import vllm
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from dynamo.common.legacy_vllm import LegacyVllmRelease
from dynamo.vllm.protocol_extensions import (
    CAPABILITY_KEY,
    ProtocolExtensionError,
    apply_sampling_extensions,
    lower_sampling_extensions,
    protocol_capability,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
]


@pytest.fixture(params=["1.4.0", "1.5.0"])
def release(request):
    path = Path(__file__).parent / "fixtures" / "protocol_releases"
    return json.loads((path / f"dynamo-{request.param}.json").read_text())


def execute_excerpt(record, namespace):
    source = record["source"]
    assert hashlib.sha256(source.encode()).hexdigest() == record["sha256"]
    # Only checked-in, generated test fixtures are executable here. Never accept
    # network/client source. Keep annotations deferred as in the released code.
    exec(  # noqa: S102 -- execute trusted, pinned repository test fixtures only
        compile(
            "from __future__ import annotations\n" + source, "<release-fixture>", "exec"
        ),
        namespace,
    )


@pytest.fixture
def old_builder(release):
    if vllm.__version__ != release["engine_version"]:
        pytest.skip(f"Requires release-pinned vLLM {release['engine_version']}")
    namespace = {
        "SamplingParams": SamplingParams,
        "_DELTA_REQUEST_OUTPUT_KIND": RequestOutputKind.DELTA,
        "logger": logging.getLogger(__name__),
    }
    parsers = {"logger": namespace["logger"]}
    for name in ("_parse_non_negative_int", "parse_logprob_options"):
        execute_excerpt(release["functions"][name], parsers)
    namespace["_shared_logprobs"] = SimpleNamespace(
        parse_logprob_options=parsers["parse_logprob_options"]
    )
    for record in release["constants"].values():
        execute_excerpt(record, namespace)
    execute_excerpt(release["functions"]["build_sampling_params"], namespace)
    return namespace["build_sampling_params"]


def internal_request(extra=None):
    return {
        "token_ids": [1, 2],
        "sampling_options": {"temperature": 0.5},
        "stop_conditions": {"max_tokens": 16},
        "output_options": {"logprobs": 2, "prompt_logprobs": 0},
        "extra_args": extra,
    }


def dual_write(fields):
    # Tests the writer's representation, not identification of a legacy card.
    # Actual old cards lack this capability; target resolution remains separate.
    capability = protocol_capability(SamplingParams(), vllm.__version__)
    return lower_sampling_extensions(fields, {CAPABILITY_KEY: capability})


def test_old_reader_keeps_default_request_operable(old_builder):
    actual = old_builder(internal_request(), {"top_p": 0.8})
    assert actual.temperature == 0.5
    assert actual.top_p == 0.8
    assert actual.max_tokens == 16
    assert actual.logprobs == 2
    assert actual.prompt_logprobs == 0
    assert actual.detokenize is False
    assert actual.output_kind == RequestOutputKind.DELTA


@pytest.mark.parametrize(
    "field", ["allowed_token_ids", "bad_words_token_ids", "logprob_token_ids"]
)
@pytest.mark.parametrize("tokens", [None, [], [0], [1, 2]])
def test_current_dual_write_survives_old_adapter(old_builder, release, field, tokens):
    value = [tokens] if field == "bad_words_token_ids" and tokens else tokens
    request = internal_request(dual_write({field: value}))
    original = deepcopy(request)
    actual = old_builder(request, {})
    attribute = "_bad_words_token_ids" if field == "bad_words_token_ids" else field
    expected = getattr(SamplingParams(), attribute) if value is None else value
    assert getattr(actual, attribute) == expected
    expected_count = (
        None
        if release["dynamo_release"] == "1.5.0"
        and field == "logprob_token_ids"
        and value
        else 2
    )
    assert actual.logprobs == expected_count
    assert request == original


def test_new_envelope_alone_is_ignored_by_old_reader(old_builder):
    extra = dual_write({"allowed_token_ids": [0]})
    del extra["sampling_options"]
    assert old_builder(internal_request(extra), {}).allowed_token_ids is None


@pytest.mark.parametrize(
    "field", ["allowed_token_ids", "bad_words_token_ids", "logprob_token_ids"]
)
@pytest.mark.parametrize("tokens", [None, [], [0], [1, 2]])
def test_explicit_legacy_policy_reaches_pinned_adapter(
    old_builder, release, field, tokens
):
    value = [tokens] if field == "bad_words_token_ids" and tokens else tokens
    target = LegacyVllmRelease(release["dynamo_release"])
    if (
        target is LegacyVllmRelease.DYNAMO_14
        and field == "logprob_token_ids"
        and value is not None
    ):
        with pytest.raises(ProtocolExtensionError, match="legacy vLLM release"):
            lower_sampling_extensions({field: value}, {}, legacy_target=target)
        return
    extra = lower_sampling_extensions({field: value}, {}, legacy_target=target)
    assert "backend_extensions" not in extra
    actual = old_builder(internal_request(extra), {})
    attribute = "_bad_words_token_ids" if field == "bad_words_token_ids" else field
    expected = getattr(SamplingParams(), attribute) if value is None else value
    assert getattr(actual, attribute) == expected
    assert actual.logprobs == (None if field == "logprob_token_ids" and value else 2)


def test_legacy_collision_precedence_differs_from_current(old_builder):
    request = internal_request(dual_write({"allowed_token_ids": [0]}))
    request["sampling_options"]["allowed_token_ids"] = [1]
    assert old_builder(request, {}).allowed_token_ids == [0]
    with pytest.raises(ProtocolExtensionError, match="conflicts with canonical"):
        apply_sampling_extensions(SamplingParams(), request)


def test_current_reader_preserves_old_writer_representable_fields(release):
    if vllm.__version__ != "0.30.0":
        pytest.skip("Current-reader baseline requires vLLM 0.30.0")
    source = release["writer"]["source"]
    assert hashlib.sha256(source.encode()).hexdigest() == release["writer"]["sha256"]
    fields = {
        "detokenize": False,
        "allowed_token_ids": [0],
        "bad_words_token_ids": [[1, 2]],
        "logprob_token_ids": [3],
    }
    legacy = {key: fields[key] for key in release["writer"]["keys"]}
    actual = SamplingParams()
    apply_sampling_extensions(actual, internal_request({"sampling_options": legacy}))
    assert actual.allowed_token_ids == [0]
    assert actual._bad_words_token_ids == [[1, 2]]
    assert actual.detokenize is False
    assert actual.logprob_token_ids == (
        [3] if release["dynamo_release"] == "1.5.0" else None
    )


def test_fixture_provenance_and_release_vocabulary(release):
    expected = {
        "1.4.0": ("03014943323e78feb5bd672ef08b72caea0918ac", "0.26.0"),
        "1.5.0": ("b83b1d9304ebfc624709ac46db32b1b6f1ff1615", "0.28.0"),
    }
    assert (release["dynamo_commit"], release["engine_version"]) == expected[
        release["dynamo_release"]
    ]
    expected_keys = ["detokenize", "allowed_token_ids", "bad_words_token_ids"]
    if release["dynamo_release"] == "1.5.0":
        expected_keys.append("logprob_token_ids")
    assert release["writer"]["keys"] == expected_keys
