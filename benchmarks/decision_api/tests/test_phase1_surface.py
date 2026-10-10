# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.resources import files

import pytest
import yaml
from dynamo_decision_perf.contracts import (
    DecisionContractError,
    validate_request,
    validate_response,
)
from dynamo_decision_perf.runner import ENDPOINTS, RunSpec
from dynamo_decision_perf.workloads import DIALECTS, Shape, questions_for, wire_payload
from test_contracts import fixture

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_phase1_surface_has_two_public_contracts_and_internal_scoring():
    assert set(DIALECTS) == {"oai", "systemone"}
    assert set(ENDPOINTS) == {"oai", "systemone", "native_score"}
    plugins = yaml.safe_load(
        files("dynamo_decision_perf").joinpath("plugins.yaml").read_text()
    )
    assert set(plugins["endpoint"]) == {
        "decision_oai",
        "decision_systemone",
        "native_score",
    }


@pytest.mark.parametrize("dialect", ["oai", "systemone"])
@pytest.mark.parametrize(
    "extension", [{}, {"format": "oai"}, {"format": "sglang_native"}, None]
)
def test_phase1_rejects_every_nvext_envelope(dialect, extension):
    request, _ = fixture(dialect)
    request["nvext"] = extension
    with pytest.raises(DecisionContractError):
        validate_request(request, dialect)


def test_deferred_public_dialect_cannot_generate_or_run_workloads():
    with pytest.raises(ValueError, match="unsupported dialect"):
        wire_payload("m", "text", questions_for(Shape("base")), "sglang_native")
    with pytest.raises(ValueError, match="unsupported dialect"):
        RunSpec("sglang_native", "http://localhost:8000", "m")
    request, response = fixture("oai")
    with pytest.raises(DecisionContractError, match="Unknown decision dialect"):
        validate_request(request, "sglang_native")
    with pytest.raises(DecisionContractError, match="Unknown decision dialect"):
        validate_response(response, "sglang_native")


@pytest.mark.parametrize("change", ["images", "thinking", "boolean_choice"])
def test_unsupported_jev_controls_are_not_ignored(change):
    request, _ = fixture("systemone")
    if change == "images":
        request["images"] = ["image"]
    if change == "thinking":
        request["chat_template_kwargs"] = {"enable_thinking": True}
    if change == "boolean_choice":
        request["questions"]["q"]["criteria"] = {True: "boolean", "true": "string"}
    with pytest.raises(DecisionContractError):
        validate_request(request, "systemone")


def test_mixed_openai_question_fields_are_rejected():
    request, _ = fixture("oai")
    request["questions"][0]["levels"] = [{"label": "low"}, {"label": "high"}]
    with pytest.raises(DecisionContractError):
        validate_request(request, "oai")
