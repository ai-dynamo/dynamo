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
from dynamo_harness import Capability, Verdict

# Small enough to differ from any model's default, large enough for the
# prompts in this file.
NARROWED_CONTEXT = 2048

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
    """Configuration testing, made portable.

    Today this class of test restarts a process inside the Dynamo container
    with different flags. Here the test names *which component* to restart and
    the flags to give it; the deployment decides how -- a container recreate
    under Docker, a DGD patch once a Kubernetes provider exists.

    Both processes share one container in this topology, so recreating it
    bounces the frontend too; a container-per-component provider would not.
    The flags are routed to the named component either way, which is what the
    last two assertions check.

    The flags are asserted to have taken effect two independent ways: a value
    the frontend reports (``context_window``) and a capability derived from the
    launch configuration. A restart that silently ignored them would satisfy
    neither -- which is the failure a bare `restart()` plus a liveness check
    cannot see.

    Note this mutates session state: the deployment keeps the new flags for any
    test that runs after it.
    """
    dynamo.require_deployment()

    before = dynamo.frontend.model_info().get("context_window")
    assert before != NARROWED_CONTEXT, (
        f"deployment already runs with context_window={before}; this test needs "
        "to change it to prove the restart applied new flags"
    )
    assert dynamo.check(Capability.REASONING_PARSER).verdict is Verdict.UNSATISFIED

    # Named component, worker-specific flags. --max-model-len and
    # --dyn-reasoning-parser configure the inference backend, not the frontend,
    # and the harness routes them to the worker's argument list alone.
    dynamo.worker.restart(max_model_len=NARROWED_CONTEXT, dyn_reasoning_parser="qwen3")
    dynamo.wait_until_serving(timeout=900)

    assert "--max-model-len" in dynamo.worker.launch_args()
    assert (
        "--max-model-len" not in dynamo.frontend.launch_args()
    ), "worker flags leaked into the frontend's launch arguments"

    after = dynamo.frontend.model_info().get("context_window")
    assert (
        after == NARROWED_CONTEXT
    ), f"restart did not apply --max-model-len: context_window {before} -> {after}"
    assert dynamo.check(Capability.REASONING_PARSER).verdict is Verdict.SATISFIED

    # and it still serves
    assert "paris" in dynamo.frontend.query("What is the capital of France?").lower()
