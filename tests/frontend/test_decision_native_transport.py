# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU transport gate for native decision profiles, not model qualification.

The metadata-only worker produces deterministic synthetic distributions. No
weights, tokenizer, inference engine, GPU, etcd, or NATS are used. This verifies
discovery, typed private RPC, HTTP projection, and the two official clients.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.metadata
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
import requests

from tests.utils.decision_api import (
    assert_distribution,
    assert_request_headers,
    decision_payload,
)
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

PROFILES = (
    ("d3", "vllm_d3_decision_v1", False),
    ("clef-flash", "vllm_clef_flash_joint_decision_v1", False),
    ("clef-omni", "vllm_clef_omni_joint_decision_v1", False),
    ("liquid", "vllm_liquid_decision_v1", True),
    ("diffusion", "vllm_diffusion_canvas_decision_v1", True),
)
MODULE = "tests.frontend.test_decision_native_transport"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(180),
]


def _model(name):
    return f"cpu-native-{name}"


def _command(module, *args):
    python = os.environ.get("DYNAMO_DECISION_TRANSPORT_SERVER_PYTHON", sys.executable)
    bootstrap = os.environ.get("DYNAMO_DECISION_TRANSPORT_BOOTSTRAP")
    if bootstrap:
        return [python, bootstrap, module, *args]
    return [python, "-m", module, *args]


@pytest.fixture(scope="module")
def native_transport(request, tmp_path_factory):
    for distribution, module, version in (
        ("openai", "openai", "3.26.0"),
        ("typesafe-sdk", "typesafe_sdk", "0.7.2"),
    ):
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            message = f"CPU native transport requires {distribution}=={version}, got {installed}"
            if os.environ.get("DYNAMO_DECISION_REQUIRE_SDKS") == "1":
                pytest.fail(message)
            pytest.skip(message)
        importlib.import_module(module)

    root = tmp_path_factory.mktemp("decision-native-transport")
    traces = root / "requests"
    traces.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DYN_FILE_KV", str(root / "discovery"))
        patch.setenv("DYN_TEST_OUTPUT_PATH", str(root / "logs"))
        patch.setenv("DYN_TCP_RPC_PORT", "0")
        patch.setenv("DYN_TCP_RESPONSE_STREAM_PORT", "0")
        patch.setenv("DYN_TCP_RPC_HOST", "127.0.0.1")
        patch.setenv("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
        patch.delenv("NATS_SERVER", raising=False)
        patch.delenv("DYN_SYSTEM_PORT", raising=False)
        frontend = DynamoFrontendProcess(
            request,
            frontend_port=0,
            extra_args=[
                "--discovery-backend",
                "file",
                "--request-plane",
                "tcp",
                "--event-plane",
                "zmq",
                "--http-host",
                "127.0.0.1",
                "--enable-systemone-api",
            ],
            terminate_all_matching_process_names=False,
            display_name="decision-native-frontend",
        )
        frontend.command = _command("dynamo.frontend", *frontend.command[3:])
        origin = f"http://127.0.0.1:{frontend.http_port}"
        expected = {_model(name) for name, _, _ in PROFILES} | {
            _model("ambiguous"),
            _model("unsupported"),
        }

        def ready(response):
            return response.status_code == 200 and expected <= {
                item["id"] for item in response.json().get("data", [])
            }

        with (
            frontend,
            ManagedProcess(
                command=_command(
                    MODULE,
                    "--worker",
                    "--namespace",
                    f"cpu-{uuid.uuid4().hex}",
                    "--traces",
                    str(traces),
                ),
                health_check_urls=[(f"{origin}/v1/models", ready)],
                timeout=60,
                terminate_all_matching_process_names=False,
                log_dir=str(root / "worker"),
                display_output=True,
                display_name="decision-native-worker",
            ),
        ):
            yield origin, traces


def _assert_response(body, model, native_read, dialect):
    assert body["model"] == model
    answers = body["answers"]
    if dialect == "oai":
        assert [item["name"] for item in answers] == ["route", "urgent", "severity"]
        route, urgent, severity = answers
        assert [item["value"] for item in route["probabilities"]] == [
            "billing",
            "technical",
        ]
        assert [item["value"] for item in severity["probabilities"]] == [0, 1, 2]
        assert [item["label"] for item in severity["probabilities"]] == [
            "Low",
            "Moderate",
            "High",
        ]
        for answer in (route, severity):
            assert_distribution(
                [item["probability"] for item in answer["probabilities"]]
            )
        assert urgent["probability"] == pytest.approx(1 / 3)
        assert all(
            "label_mass" not in item and "x_label_mass" not in item for item in answers
        )
        assert body["usage"]["input_tokens_details"]["cached_tokens"] == 0
        assert body["usage"]["total_tokens"] == 32 + (3 if native_read else 0)
    else:
        assert list(answers) == ["route", "urgent", "severity"]
        route, urgent, severity = answers.values()
        assert list(route["probabilities"]) == ["billing", "technical"]
        assert list(severity["probabilities"]) == ["0", "1", "2"]
        assert severity["legend"] == {"0": "Low", "1": "Moderate", "2": "High"}
        for answer in (route, severity):
            assert_distribution(list(answer["probabilities"].values()))
        assert urgent["noul"] == pytest.approx(1 / 3)
        for answer in answers.values():
            if native_read:
                assert answer["x_label_mass"] == pytest.approx(0.75)
            else:
                assert "x_label_mass" not in answer
    assert route["choice"] == "technical"
    assert severity["score"] == pytest.approx(4 / 3)
    assert body["usage"]["input_tokens"] == 32
    assert body["usage"]["output_tokens"] == (3 if native_read else 0)


def _private_request(traces, request_id, native_read):
    matches = [json.loads(path.read_text()) for path in traces.glob("*.json")]
    suffix = "-read" if native_read else "-head"
    (record,) = [item for item in matches if item["context_id"] == request_id + suffix]
    wire = record["request"]
    key = "decision_read" if native_read else "decision_head"
    assert set(wire) == {key}
    return wire[key]


def _assert_private_trace(traces, request_id, profile, native_read):
    canonical = _private_request(traces, request_id, native_read)
    assert canonical["protocol_version"] == 1 and canonical["profile"] == profile
    assert [question["ordinal"] for question in canonical["questions"]] == [0, 1, 2]
    assert [len(question["candidates"]) for question in canonical["questions"]] == [
        2,
        2,
        3,
    ]
    assert [
        candidate["value"] for candidate in canonical["questions"][1]["candidates"]
    ] == [True, False]
    assert canonical["cache_salt"] and canonical["max_input_tokens"] > 0


@pytest.mark.parametrize("name,profile,native_read", PROFILES)
def test_native_transport_openai_sdk(native_transport, name, profile, native_read):
    import openai

    origin, traces = native_transport
    model = _model(name)
    with openai.OpenAI(
        base_url=f"{origin}/v1",
        api_key="test",
        max_retries=0,
        timeout=30,
        _strict_response_validation=True,
    ) as client:
        response = client.decisions.with_raw_response.create(**decision_payload(model))
        assert_request_headers(response)
        result = response.parse()
        assert result._request_id == response.headers["x-request-id"]
        _assert_response(result.model_dump(), model, native_read, "oai")
        _assert_private_trace(traces, result._request_id, profile, native_read)
        with pytest.raises(openai.BadRequestError) as error:
            client.decisions.create(
                **{**decision_payload(model), "input": "backend-value-error"}
            )
        assert error.value.status_code == 400 and error.value.request_id


@pytest.mark.parametrize("name,profile,native_read", PROFILES)
def test_native_transport_typesafe_sdk(native_transport, name, profile, native_read):
    import typesafe_sdk as sdk

    origin, traces = native_transport
    questions = {
        "route": sdk.Choice(
            instructions="Choose the responsible team.",
            criteria={"billing": "Payments", "technical": "Software"},
        ),
        "urgent": sdk.Noul(instructions="Action is needed today."),
        "severity": sdk.Score(
            instructions="Rate the impact.", criteria=["Low", "Moderate", "High"]
        ),
    }
    with sdk.TypeSafeClient(
        base_url=origin,
        model=_model(name),
        api_key="test",
        retry=sdk.RetryPolicy(max_retries=0),
        timeout=30,
    ) as client:
        result = client.system_one(
            state=decision_payload(_model(name))["input"], questions=questions
        )
        assert_request_headers(result.raw_http_response)
        assert result.request_id == result.raw_http_response.headers["x-request-id"]
        assert list(result.answers) == ["route", "urgent", "severity"]
        _assert_response(
            result.raw_http_response.json(), _model(name), native_read, "systemone"
        )
        _assert_private_trace(traces, result.request_id, profile, native_read)
        with pytest.raises(sdk.TypeSafeUnprocessableEntityError) as error:
            client.system_one(state="backend-value-error", questions=questions)
        assert error.value.status == 422


@pytest.mark.parametrize("name", ("d3", "liquid", "diffusion"))
@pytest.mark.parametrize(
    "dialect,route,error_status",
    (("oai", "decisions", 400), ("systemone", "systemone", 422)),
)
def test_native_transport_error_and_cache_contract(
    native_transport, name, dialect, route, error_status
):
    origin, _ = native_transport
    state_key = "input" if dialect == "oai" else "state"
    for mode, status in (
        ("backend-value-error", error_status),
        ("backend-unavailable", 503),
        ("malformed-backend", 500),
        ("unknown-cache", 502 if dialect == "oai" else 200),
    ):
        payload = {**decision_payload(_model(name), dialect), state_key: mode}
        response = requests.post(f"{origin}/v1/{route}", json=payload, timeout=30)
        assert response.status_code == status, response.text
        assert_request_headers(response)
        body = response.json()
        if status == 200:
            assert "cached_tokens" not in json.dumps(body["usage"])
        else:
            assert "answers" not in body
            assert "private diagnostic" not in response.text
            if mode == "unknown-cache":
                assert body["error"]["code"] == "invalid_executor_result"
                assert "measured cache reads" in body["error"]["message"]


@pytest.mark.parametrize("name", ("d3", "liquid", "diffusion"))
def test_native_transport_preserves_typed_choice_values(native_transport, name):
    import openai

    origin, traces = native_transport
    values = [True, "true", False, "false"]
    with openai.OpenAI(
        base_url=f"{origin}/v1",
        api_key="test",
        max_retries=0,
        timeout=30,
        _strict_response_validation=True,
    ) as client:
        result = client.decisions.create(
            model=_model(name),
            input="typed values",
            questions=[
                {
                    "type": "choice",
                    "instructions": "Choose the typed value.",
                    "choices": [{"value": value} for value in values],
                }
            ],
        )
        probabilities = result.model_dump()["answers"][0]["probabilities"]
        assert [item["value"] for item in probabilities] == values
        assert [type(item["value"]) for item in probabilities] == [bool, str, bool, str]
        canonical = _private_request(traces, result._request_id, name != "d3")
        assert [question["ordinal"] for question in canonical["questions"]] == [0]
        candidates = canonical["questions"][0]["candidates"]
        assert [item["value"] for item in candidates] == values
        assert [type(item["value"]) for item in candidates] == [bool, str, bool, str]


@pytest.mark.parametrize(
    "dialect,route,status", (("oai", "decisions", 400), ("systemone", "systemone", 422))
)
def test_native_transport_rejects_before_dispatch(
    native_transport, dialect, route, status
):
    origin, traces = native_transport
    before = set(traces.glob("*.json"))
    cases = [
        (
            {
                **decision_payload(_model("liquid"), dialect),
                "nvext": {"format": "invalid"},
            },
            status,
        ),
        (decision_payload(_model("ambiguous"), dialect), 400),
        (decision_payload(_model("unsupported"), dialect), 400),
    ]
    for payload, expected_status in cases:
        response = requests.post(f"{origin}/v1/{route}", json=payload, timeout=30)
        assert response.status_code == expected_status, response.text
        assert_request_headers(response)
        assert "answers" not in response.json()
    assert set(traces.glob("*.json")) == before


async def _worker(namespace, traces):
    from dynamo.llm import (
        ModelInput,
        ModelRuntimeConfig,
        ModelType,
        WorkerType,
        register_model,
    )
    from dynamo.llm.exceptions import HttpError
    from dynamo.runtime import DistributedRuntime

    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )

    async def generate(request, context):
        (traces / f"{uuid.uuid4().hex}.json").write_text(
            json.dumps({"context_id": context.id(), "request": request})
        )
        (key,) = request
        canonical = request[key]
        if canonical["input"] == "backend-value-error":
            raise ValueError("synthetic native input rejected")
        if canonical["input"] == "backend-unavailable":
            raise HttpError(503, "private diagnostic")
        native_read = key == "decision_read"
        outcomes = [
            {
                "ordinal": question["ordinal"],
                "probabilities": [
                    (index + 1) / sum(range(1, len(question["candidates"]) + 1))
                    for index in range(len(question["candidates"]))
                ],
                **({"label_mass": 0.75} if native_read else {}),
            }
            for question in canonical["questions"]
        ]
        if canonical["input"] == "malformed-backend":
            outcomes = [
                {**outcome, "probabilities": [2.0] * len(outcome["probabilities"])}
                for outcome in outcomes
            ]
        yield {
            "protocol_version": 1,
            "profile": canonical["profile"],
            "outcomes": outcomes,
            "usage": {
                "input_tokens": 32,
                "output_tokens": len(outcomes) if native_read else 0,
                "cached_tokens": None if canonical["input"] == "unknown-cache" else 0,
            },
        }

    specs = (*PROFILES, ("ambiguous", None, True), ("unsupported", None, True))
    endpoints = []
    for index, (name, profile, native_read) in enumerate(specs):
        endpoint = runtime.endpoint(f"{namespace}.profile{index}.generate")
        config = ModelRuntimeConfig()
        config.context_length = 4096
        config.max_num_seqs = 8
        config.data_parallel_size = 1
        capabilities = (
            (profile,)
            if profile
            else (() if name == "unsupported" else (PROFILES[3][1], PROFILES[4][1]))
        )
        for capability in capabilities:
            config.set_engine_specific(capability, "true")
        await register_model(
            ModelInput.Text,
            ModelType.Chat if native_read else ModelType.Classify | ModelType.Pooling,
            endpoint,
            _model(name),
            model_name=_model(name),
            runtime_config=config,
            worker_type=WorkerType.Aggregated,
            skip_model_assets=True,
        )
        endpoints.append(endpoint)
    await asyncio.gather(*(endpoint.serve_endpoint(generate) for endpoint in endpoints))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--traces", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_worker(args.namespace, args.traces))
