# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tests/serve/test_sample.py, rewritten against the component harness.

The original builds an EngineConfig naming a launch script, a port, and three
payload objects, then hands it to a runner that knows how to start Dynamo in
the same container. It can only run where that runner can launch processes.

These assert the same three things -- chat, completion, metrics -- but take a
`dynamo` fixture and never mention ports, scripts, or HTTP. The same file runs
against a container the test deployed, or against a frontend somebody else is
already running, with no edits.
"""

import pytest

pytestmark = [pytest.mark.e2e]


def test_frontend_serves_a_model(dynamo):
    """The deployment advertises the model it was asked to serve."""
    models = dynamo.frontend.models()
    assert models, "frontend advertises no models"
    assert dynamo.frontend.model in models


def test_chat_completion(dynamo):
    """Chat returns usable text, not merely a 200."""
    answer = dynamo.frontend.query("What is the capital of France?", max_tokens=300)
    assert answer.strip(), "chat returned empty content"
    assert "paris" in answer.lower(), f"unexpected answer: {answer[:200]!r}"


def test_text_completion(dynamo):
    """The /v1/completions surface answers too, not just chat."""
    text = dynamo.frontend.complete("The capital of France is", max_tokens=32)
    assert text.strip(), "completion returned empty text"


def test_metrics_exposed(dynamo):
    """Prometheus surface is present and has counted the requests above.

    Asserts that *a* request counter advanced rather than naming one: the
    exact family differs between builds (released 1.4.0 exposes
    ``dynamo_frontend_requests_*`` but no ``dynamo_component_requests_total``),
    and pinning the name tests the build, not the behaviour.

    Ordering note: this runs after the two inference tests, so the counter is
    non-zero. That ordering is a property of this file, not of the harness.
    """
    body = dynamo.frontend.metrics()
    assert "dynamo_" in body, "no dynamo metrics exposed at /metrics"

    counters = dynamo.frontend.metric_samples("dynamo_frontend_requests")
    if not counters:
        families = sorted(
            {
                line.split("{")[0]
                for line in body.splitlines()
                if line.startswith("dynamo_")
            }
        )
        raise AssertionError(f"no request counter found; families seen: {families[:8]}")
    assert sum(counters.values()) >= 1, f"request counter did not advance: {counters}"


def test_empty_prompt_is_rejected(dynamo):
    """Model-, backend- and deployment-independent, and generates no tokens."""
    status = dynamo.frontend.expect_rejected(
        "/v1/completions",
        {"model": dynamo.frontend.model, "prompt": [], "max_tokens": 1},
    )
    assert status == 400, f"expected HTTP 400 for an empty prompt, got {status}"


@pytest.mark.needs_deployment
def test_restart_with_different_flags(dynamo):
    """Config/topology testing, made portable.

    Today this class of test restarts a process inside the Dynamo container.
    Here the test asks the component to come back with different flags and the
    deployment decides how -- container recreate here, a DGD patch on
    Kubernetes.
    """
    deployment = dynamo.require_deployment()
    deployment.restart()
    assert dynamo.frontend.wait_until_serving(timeout=900)
    assert "paris" in dynamo.frontend.query("What is the capital of France?").lower()
