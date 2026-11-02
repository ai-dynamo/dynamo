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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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


@contextmanager
def _native_transport(request, tmp_path_factory):
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
        patch.setenv("DYN_RUNTIME_INHIBITED_DURATION_SECS", "120")
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
            _model("missing-limit"),
            _model("qualification-failed"),
        }

        def ready(response):
            if response.status_code != 200 or not expected <= {
                item["id"] for item in response.json().get("data", [])
            }:
                return False
            failed = requests.post(
                f"{origin}/v1/decisions",
                json=decision_payload(_model("qualification-failed-read")),
                timeout=3,
            )
            return (
                failed.status_code == 400
                and failed.json().get("error", {}).get("code")
                == "model_qualification_failed"
            )

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


@pytest.fixture(scope="module")
def native_transport(request, tmp_path_factory):
    with _native_transport(request, tmp_path_factory) as transport:
        yield transport


@pytest.fixture
def isolated_native_transport(request, tmp_path_factory):
    with _native_transport(request, tmp_path_factory) as transport:
        yield transport


def _assert_response(body, model, native_read, dialect):
    assert "finish_reason" not in body
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
    assert record["dispatch"] == ("read" if native_read else "head")
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


@pytest.mark.parametrize("dialect", ("oai", "systemone"))
@pytest.mark.parametrize("name", ("qualification-failed", "qualification-failed-read"))
def test_native_qualification_failure_is_non_retryable_in_official_clients(
    native_transport, dialect, name
):
    import openai
    import typesafe_sdk as sdk

    origin, traces = native_transport
    model = _model(name)
    before = set(traces.glob("*.json"))
    if dialect == "oai":
        with openai.OpenAI(
            base_url=f"{origin}/v1", api_key="test", timeout=10
        ) as client:
            with pytest.raises(openai.BadRequestError) as caught:
                client.decisions.create(**decision_payload(model))
        error = caught.value
        assert error.status_code == 400
        body = error.response.json()
        assert_request_headers(error.response)
    else:
        with sdk.TypeSafeClient(
            base_url=origin, model=model, api_key="test", timeout=10
        ) as client:
            with pytest.raises(sdk.TypeSafeBadRequestError) as caught:
                client.system_one(
                    state="Classify the request.",
                    questions={"urgent": sdk.Noul(instructions="Urgent?")},
                )
        error = caught.value
        assert error.status == 400
        body = error.body
        assert_request_headers(SimpleNamespace(headers=error.headers))
    assert error.request_id
    assert body.get("error", body)["code"] == "model_qualification_failed"
    assert "answers" not in body
    assert set(traces.glob("*.json")) == before


@pytest.mark.parametrize("name", ("d3", "liquid", "diffusion"))
@pytest.mark.parametrize(
    "dialect,route,error_status",
    (("oai", "decisions", 400), ("systemone", "systemone", 422)),
)
def test_native_transport_error_and_cache_contract(
    native_transport, name, dialect, route, error_status
):
    origin, traces = native_transport
    state_key = "input" if dialect == "oai" else "state"
    modes = (
        ("backend-value-error", error_status),
        ("backend-unavailable", 503),
        ("malformed-backend", 502),
        ("unknown-cache", 502),
        ("adapter-execution-error", 502),
        ("adapter-unexpected-type-error", 500),
        ("adapter-engine-generate-error", 503),
    )
    if name != "d3":
        modes = (*modes, ("renderer-missing-salt", 502), ("renderer-changed-salt", 502))
    for mode, status in modes:
        payload = {**decision_payload(_model(name), dialect), state_key: mode}
        response = requests.post(f"{origin}/v1/{route}", json=payload, timeout=30)
        assert response.status_code == status, response.text
        assert_request_headers(response)
        canonical = _private_request(
            traces, response.headers["x-request-id"], name != "d3"
        )
        assert canonical["input"] == mode
        body = response.json()
        assert "answers" not in body
        assert "private diagnostic" not in response.text
        if mode in {
            "malformed-backend",
            "unknown-cache",
            "adapter-execution-error",
        } or mode.startswith("renderer-"):
            assert body.get("error", body)["code"] == "invalid_executor_result"


@pytest.mark.parametrize("name", ("d3", "liquid", "diffusion"))
@pytest.mark.parametrize(
    "dialect,route", (("oai", "decisions"), ("systemone", "systemone"))
)
@pytest.mark.parametrize(
    "mode", ("adapter-engine-generate-error", "adapter-engine-dead-error")
)
def test_native_engine_fault_reaches_worker_and_preserves_status(
    request, name, dialect, route, mode
):
    origin, traces = request.getfixturevalue(
        "isolated_native_transport"
        if mode == "adapter-engine-dead-error"
        else "native_transport"
    )
    state_key = "input" if dialect == "oai" else "state"
    payload = {**decision_payload(_model(name), dialect), state_key: mode}
    response = requests.post(f"{origin}/v1/{route}", json=payload, timeout=30)
    assert response.status_code == 503, response.text
    assert_request_headers(response)
    canonical = _private_request(traces, response.headers["x-request-id"], name != "d3")
    assert canonical["input"] == mode
    assert "answers" not in response.json()
    assert "private diagnostic" not in response.text
    assert (
        response.json().get("error", response.json())["code"]
        != "invalid_executor_result"
    )


def _ordinary_probe(origin, name):
    native_read = name != "d3"
    route = "chat/completions" if native_read else "classify"
    payload = {"model": _model(name)}
    if native_read:
        payload["messages"] = [{"role": "user", "content": "ordinary probe"}]
    else:
        payload["input"] = "ordinary probe"
    return requests.post(f"{origin}/v1/{route}", json=payload, timeout=3)


@pytest.mark.parametrize("name", ("d3", "liquid"))
@pytest.mark.parametrize("dead", (False, True))
def test_native_engine_fault_preserves_ordinary_worker_only_when_recoverable(
    isolated_native_transport, name, dead
):
    origin, traces = isolated_native_transport
    baseline = _ordinary_probe(origin, name)
    assert baseline.status_code == 200, baseline.text
    (baseline_record,) = [
        json.loads(path.read_text()) for path in traces.glob("*.json")
    ]
    mode = "adapter-engine-dead-error" if dead else "adapter-engine-generate-error"
    response = requests.post(
        f"{origin}/v1/decisions",
        json={**decision_payload(_model(name)), "input": mode},
        timeout=30,
    )
    assert response.status_code == 503, response.text
    assert_request_headers(response)
    assert "answers" not in response.json()
    assert "private diagnostic" not in response.text
    assert (
        _private_request(traces, response.headers["x-request-id"], name != "d3")[
            "input"
        ]
        == mode
    )
    before = set(traces.glob("*.json"))
    ordinary = _ordinary_probe(origin, name)
    assert ordinary.status_code == (503 if dead else 200), ordinary.text
    records = [
        json.loads(path.read_text()) for path in set(traces.glob("*.json")) - before
    ]
    if dead:
        assert records == []
    else:
        (record,) = records
        assert record["dispatch"] is None
        assert record["request"]["model"] == _model(name)
        assert record["worker"] == baseline_record["worker"]
        assert ordinary.json() == baseline.json()


@pytest.mark.parametrize("name", [profile[0] for profile in PROFILES])
@pytest.mark.parametrize(
    "dialect,route", (("oai", "decisions"), ("systemone", "systemone"))
)
def test_native_thinking_off_controls_follow_public_schema(
    native_transport, name, dialect, route
):
    origin, traces = native_transport
    payload = decision_payload(_model(name), dialect)
    baseline = requests.post(
        f"{origin}/v1/{route}",
        json={
            key: value
            for key, value in payload.items()
            if key != "chat_template_kwargs"
        },
        timeout=30,
    )
    assert baseline.status_code == 200, baseline.text
    before = set(traces.glob("*.json"))
    for controls in (
        {"enable_thinking": False},
        {"thinking": False},
        {"enable_thinking": False, "thinking": False},
    ):
        response = requests.post(
            f"{origin}/v1/{route}",
            json={**payload, "chat_template_kwargs": controls},
            timeout=30,
        )
        assert_request_headers(response)
        if dialect == "oai":
            assert response.status_code == 400, response.text
            assert "answers" not in response.json()
            assert set(traces.glob("*.json")) == before
        else:
            assert response.status_code == 200, response.text
            assert response.json()["answers"] == baseline.json()["answers"]


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
        (
            {
                **decision_payload(_model("liquid"), dialect),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            status,
        ),
        (decision_payload(_model("ambiguous"), dialect), 400),
        (decision_payload(_model("unsupported"), dialect), 400),
        (decision_payload(_model("missing-limit"), dialect), 400),
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
    from dynamo.vllm.decision_dispatch import DECISION_DISPATCH_METADATA_KEY
    from dynamo.vllm.decision_heads import HeadExecutionAdapter
    from dynamo.vllm.decision_native_errors import NativeExecutionError
    from dynamo.vllm.decision_native_wire import measured_cache_tokens
    from dynamo.vllm.decision_reads import ReadExecutionAdapter
    from dynamo.vllm.tests.decision_pinned_source import load_pinned_engine_errors

    engine_errors = load_pinned_engine_errors()

    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )

    async def generate(request, context):
        (traces / f"{uuid.uuid4().hex}.json").write_text(
            json.dumps(
                {
                    "context_id": context.id(),
                    "worker": namespace,
                    "request": request,
                    "dispatch": context.metadata.get(DECISION_DISPATCH_METADATA_KEY),
                }
            )
        )
        if context.metadata.get(DECISION_DISPATCH_METADATA_KEY) is None:
            model = request["model"]
            if "messages" in request:
                yield {
                    "id": "cpu-ordinary",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            else:
                yield {
                    "id": "cpu-ordinary",
                    "object": "list",
                    "created": 1,
                    "model": model,
                    "data": [
                        {
                            "index": 0,
                            "label": "ok",
                            "probs": [0.25, 0.75],
                            "num_classes": 2,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 0,
                        "total_tokens": 3,
                    },
                }
            return
        (key,) = request
        canonical = request[key]
        if canonical["input"] == "backend-value-error":
            raise ValueError("synthetic native input rejected")
        if canonical["input"] == "backend-unavailable":
            raise HttpError(503, "private diagnostic")
        if canonical["input"] == "adapter-execution-error":
            raise NativeExecutionError("private diagnostic")
        if canonical["input"] in {
            "adapter-unexpected-type-error",
            "adapter-engine-generate-error",
            "adapter-engine-dead-error",
        }:
            adapter_type = (
                ReadExecutionAdapter if key == "decision_read" else HeadExecutionAdapter
            )
            adapter = object.__new__(adapter_type)
            adapter.native = SimpleNamespace(
                overload_errors=(),
                engine_generate_error=engine_errors.EngineGenerateError,
                engine_dead_error=engine_errors.EngineDeadError,
                retryable_request_error=(),
            )
            # Keep the stub alive; only actual engine death must inhibit routing.
            adapter.on_engine_dead = None

            async def broken_prepare(wire):
                if canonical["input"] == "adapter-engine-generate-error":
                    raise engine_errors.EngineGenerateError("private diagnostic")
                if canonical["input"] == "adapter-engine-dead-error":
                    raise engine_errors.EngineDeadError()
                raise TypeError("private diagnostic")

            adapter._prepare = broken_prepare
            await adapter.execute(
                request,
                context,
                request_id=context.id(),
                cancelled=lambda: False,
            )
        if canonical["input"] in {"renderer-missing-salt", "renderer-changed-salt"}:
            validator = object.__new__(ReadExecutionAdapter)
            validator.engine = SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=4096)
            )
            validator.spec = SimpleNamespace(name=canonical["profile"])
            prompt = {"type": "token", "prompt_token_ids": [1]}
            if canonical["input"] == "renderer-changed-salt":
                prompt = {**prompt, "cache_salt": "different-isolation-domain"}
            validator._validate_plan(
                prompt, SimpleNamespace(max_tokens=1), canonical["cache_salt"]
            )
        cached = measured_cache_tokens(
            None if canonical["input"] == "unknown-cache" else 0, 32
        )
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
            "finish_reason": "stop",
            "profile": canonical["profile"],
            "outcomes": outcomes,
            "usage": {
                "input_tokens": 32,
                "output_tokens": len(outcomes) if native_read else 0,
                "cached_tokens": cached,
            },
        }

    specs = (
        *PROFILES,
        ("ambiguous", None, True),
        ("unsupported", None, True),
        ("missing-limit", PROFILES[0][1], False),
        ("qualification-failed", None, False),
        ("qualification-failed-read", None, True),
    )
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
            else (
                ()
                if name
                in {"unsupported", "qualification-failed", "qualification-failed-read"}
                else (PROFILES[3][1], PROFILES[4][1])
            )
        )
        for capability in capabilities:
            config.set_engine_specific(capability, "true")
        if capabilities and name != "missing-limit":
            config.set_engine_specific("vllm_native_decision_max_candidates", "26")
        if name in {"qualification-failed", "qualification-failed-read"}:
            config.set_engine_specific(
                "vllm_native_decision_qualification_failure_v1",
                json.dumps({"profile": PROFILES[3 if native_read else 0][1]}),
            )
        await register_model(
            (
                ModelInput.Tokens
                if name == "qualification-failed-read"
                else ModelInput.Text
            ),
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
