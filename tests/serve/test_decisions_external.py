# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in qualification against an already running decision deployment.

Set DYNAMO_DECISION_TEST_URL and DYNAMO_DECISION_TEST_MODEL explicitly. This
client does not launch or download a model; large-model qualification remains
outside the shared CI GPU budget. Set DYNAMO_DECISION_REQUIRE_SDKS=1 to make
missing official clients a failure instead of an optional-dependency skip.

Use the server origin without /v1 and the exact registered model alias. The
recorded client profile is openai==3.26.0 and typesafe==0.7.2 in an isolated
client environment; do not replace shared repository dependency pins. This
module exercises synchronous Python clients, not the complete SDK matrix.
For independent native numerical parity, also set DYNAMO_DECISION_NATIVE_URL
and DYNAMO_DECISION_TOKENIZER_PATH to the reference server and pinned local
tokenizer artifacts. These raw HTTP fixtures assume a trusted local test
deployment without gateway authentication and do not qualify authenticated
gateways, benchmark quality, throughput, or physical cancellation settlement.
"""

import importlib
import importlib.util
import os
from urllib.parse import urlsplit

import pytest
import requests

from tests.serve.test_systemone_sglang import assert_native_sglang_parity
from tests.utils.decision_api import (
    assert_decision_body,
    assert_request_headers,
    decision_payload,
)

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.timeout(180),
]


@pytest.fixture
def external_decision_server():
    base_url = os.environ.get("DYNAMO_DECISION_TEST_URL")
    if not base_url:
        pytest.skip("Set DYNAMO_DECISION_TEST_URL for an explicitly provisioned server")
    parsed = urlsplit(base_url)
    assert parsed.scheme in ("http", "https") and parsed.netloc
    assert parsed.path in ("", "/"), "Use the server origin, without /v1"
    model = os.environ.get("DYNAMO_DECISION_TEST_MODEL")
    assert model, "Set the exact registered model in DYNAMO_DECISION_TEST_MODEL"
    return base_url.rstrip("/"), model


def _sdk(module_name):
    if importlib.util.find_spec(module_name) is None:
        if os.environ.get("DYNAMO_DECISION_REQUIRE_SDKS") == "1":
            pytest.fail(f"Required official SDK is unavailable: {module_name}")
        pytest.skip(f"Optional official SDK is unavailable: {module_name}")
    return importlib.import_module(module_name)


@pytest.mark.parametrize("dialect", ("oai", "sglang_native", "systemone"))
def test_external_decision_wire_contract(external_decision_server, dialect):
    base_url, model = external_decision_server
    route = "systemone" if dialect == "systemone" else "decisions"
    response = requests.post(
        f"{base_url}/v1/{route}",
        json=decision_payload(model, dialect),
        timeout=60,
    )
    assert response.status_code == 200, response.text
    assert_request_headers(response)
    assert_decision_body(response.json(), model, dialect)


def test_external_openai_sdk_evaluation(external_decision_server):
    base_url, model = external_decision_server
    sdk = _sdk("openai")
    with sdk.OpenAI(
        base_url=f"{base_url}/v1",
        api_key=os.environ.get("DYNAMO_DECISION_TEST_API_KEY", "local-test-key"),
        max_retries=0,
        timeout=60,
        _strict_response_validation=True,
    ) as client:
        if not hasattr(client, "decisions"):
            message = f"OpenAI SDK {sdk.__version__} has no Decisions resource; qualify with 3.26.0"
            if os.environ.get("DYNAMO_DECISION_REQUIRE_SDKS") == "1":
                pytest.fail(message)
            pytest.skip(message)
        response = client.decisions.with_raw_response.create(**decision_payload(model))
        assert_request_headers(response)
        result = response.parse()
        assert_decision_body(result.model_dump(), model, "oai")
        assert result._request_id == response.headers["x-request-id"]
        with pytest.raises(sdk.BadRequestError) as error:
            client.decisions.create(
                **decision_payload(model), extra_body={"nvext": {"format": "invalid"}}
            )
        assert error.value.status_code == 400
        assert error.value.request_id


def test_external_typesafe_sdk_evaluation(external_decision_server):
    base_url, model = external_decision_server
    sdk = _sdk("typesafe_sdk")
    payload = decision_payload(model, "systemone")
    questions = {
        "route": sdk.Choice(
            **{k: v for k, v in payload["questions"]["route"].items() if k != "type"}
        ),
        "urgent": sdk.Noul(instructions="Action is needed today."),
        "severity": sdk.Score(
            instructions="Rate the impact.", criteria=["Low", "Moderate", "High"]
        ),
    }
    with sdk.TypeSafeClient(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("DYNAMO_DECISION_TEST_API_KEY", "local-test-key"),
        retry=sdk.RetryPolicy(max_retries=0),
        timeout=60,
    ) as client:
        result = client.system_one(state=payload["state"], questions=questions)
        assert list(result.answers) == ["route", "urgent", "severity"]
        assert result.choices["route"].choice in ("billing", "technical")
        assert 0 <= result.nouls["urgent"].noul <= 1
        assert 0 <= result.scores["severity"].score <= 2
        assert result.request_id
        assert_request_headers(result.raw_http_response)
        assert (
            result.request_id
            == result.raw_http_response.headers["x-typesafe-request-id"]
        )
        assert_decision_body(result.raw_http_response.json(), model, "systemone")
        with pytest.raises(sdk.TypeSafeUnprocessableEntityError) as error:
            client.system_one(
                state=payload["state"],
                questions=questions,
                extra_body={"questions": {}},
            )
        assert error.value.status == 422


def test_external_native_sglang_numeric_parity(external_decision_server):
    base_url, model = external_decision_server
    native_url = os.environ.get("DYNAMO_DECISION_NATIVE_URL")
    if not native_url:
        pytest.skip(
            "Set DYNAMO_DECISION_NATIVE_URL to an independently provisioned native SGLang server"
        )
    assert (
        native_url.rstrip("/") != base_url
    ), "Native parity requires an independent server, not Dynamo /generate"
    tokenizer_path = os.environ.get("DYNAMO_DECISION_TOKENIZER_PATH")
    assert_native_sglang_parity(base_url, native_url.rstrip("/"), model, tokenizer_path)
