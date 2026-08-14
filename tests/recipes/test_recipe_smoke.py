# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional checks against an already-deployed recipe.

These never create, patch or delete a deployment: they take a frontend URL from
``--endpoint-url`` and assert on the inference response. That makes them safe to
run against a shared or production deployment, and it is what lets one suite
cover a recipe on Kubernetes, a staging cluster, or a local
``python -m dynamo.frontend`` without changing a line.

Run::

    kubectl port-forward svc/<deployment>-frontend 8000:8000 -n <ns>
    pytest tests/recipes -m endpoint_only --endpoint-url=http://localhost:8000

Deliberately NOT here: anything asserting on which worker served a request,
worker ``/metrics``, pod logs, or replica counts. Those need a deployment handle
and belong behind ``topology_dependent``.

Also NOT here: whether the generated text is *language*. These tests go through
``run_payloads``, which returns None and so exposes no response body to assert
on, and they are the transport/envelope checks. Output quality -- and the
per-replica version of it -- lives in ``test_recipe_output_quality.py``. Do not
read a pass here as evidence the deployment is generating sensible tokens; a
replica with a stuck decoder passes every assertion in this module.
"""

import pytest

from tests.utils.payloads import ChatPayload, CompletionPayload
from tests.utils.verification import run_payloads

pytestmark = [
    pytest.mark.endpoint_only,
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_0,  # the runner needs no GPU; the deployment already has them
]


def test_endpoint_lists_a_model(attached_endpoint):
    """GET /v1/models advertises the model the fixture resolved."""
    import json
    import urllib.request

    request = urllib.request.Request(
        f"{attached_endpoint.base_url}/v1/models",
        headers=dict(attached_endpoint.headers or {}),
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    ids = [entry["id"] for entry in payload.get("data", [])]
    assert attached_endpoint.model in ids, (
        f"fixture resolved model {attached_endpoint.model!r} but /v1/models "
        f"advertises {ids}"
    )


def test_endpoint_serves_chat(attached_endpoint):
    """A chat completion comes back with non-empty assistant output.

    ``expected_response=[]`` is passed to the constructor directly rather than
    through ``chat_payload_default``: that builder does ``expected_response or
    [...]``, and ``[]`` is falsy, so it would silently reinstate a keyword list
    ("AI", "knock", "joke", ...) and assert an arbitrary model says one of them.

    ``max_tokens`` is generous because a reasoning-parser deployment (this
    recipe sets ``--dyn-reasoning-parser``) spends its first tokens inside
    ``<think>``; a short budget never closes the block and ``content`` arrives
    empty through no fault of the deployment.
    """
    payload = ChatPayload(
        body={
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 512,
            "stream": False,
        },
        expected_response=[],
        expected_log=[],  # required field; non-empty would make this topology_dependent
        min_content_length=0,
        timeout=300,
    ).bind(attached_endpoint)
    run_payloads([payload])


def test_endpoint_serves_completion(attached_endpoint):
    """The /v1/completions surface answers, not just /v1/chat/completions."""
    payload = CompletionPayload(
        body={"prompt": "The capital of France is", "max_tokens": 32, "stream": False},
        expected_response=[],
        expected_log=[],  # required field; non-empty would make this topology_dependent
        min_content_length=0,
        timeout=300,
    ).bind(attached_endpoint)
    run_payloads([payload])


def test_completion_rejects_empty_array_prompt(attached_endpoint):
    """``prompt: []`` is a 400 from the frontend, never reaching a worker.

    Model-, backend- and topology-independent, and it generates zero tokens --
    the cheapest real assertion available against an expensive deployment.
    """
    import json
    import urllib.error
    import urllib.request

    body = json.dumps(
        {"model": attached_endpoint.model, "prompt": [], "max_tokens": 1}
    ).encode()
    headers = {
        "Content-Type": "application/json",
        **dict(attached_endpoint.headers or {}),
    }
    request = urllib.request.Request(
        f"{attached_endpoint.base_url}/v1/completions", data=body, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            pytest.fail(
                f"empty-array prompt should be rejected, got HTTP {response.status}"
            )
    except urllib.error.HTTPError as exc:
        assert exc.code == 400, f"expected HTTP 400 for empty prompt, got {exc.code}"
