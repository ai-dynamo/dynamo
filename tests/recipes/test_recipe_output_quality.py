# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert the deployment emits language, and that *every* replica does.

Written after a 3-replica Kimi-K2.5 deployment on GB300 served pure token-0
(``!``) output from one of its three replicas while every Kubernetes signal
reported health: pods ``1/1 Running``, 0 restarts, ``Ready=True``, no error in
any worker log, and a TRT-LLM config byte-identical to the healthy replicas.
Nothing in a readiness probe samples whether a worker's output is language.

Two properties are needed to catch that, and both are easy to get wrong:

1. **Assert language, not truthiness.** ``"!!!!"`` is a non-empty string, so
   every ``assert response.content`` in this suite passed against it.
2. **Vary the prompt.** KV-aware routing pins a prefix to one worker, so a
   fixed prompt repeated N times measures one replica N times. The
   deployment-wide failure rate is only visible with unique prefixes.
"""

import collections

import pytest

from tests.recipes.helpers import (
    assert_natural_language,
    degeneracy_reason,
    message_text,
    post,
    unique_prompt,
    worker_id_of,
)

pytestmark = [
    pytest.mark.endpoint_only,
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
]

# Enough unique prompts to reach every replica of a small fleet with high
# probability without making the test expensive. With R replicas and uniform
# routing, the chance of missing any given replica is (1 - 1/R)**PROBE_COUNT --
# under 2% for R=3, under 8% for R=5. Coverage is probabilistic by nature: the
# endpoint does not advertise its replica count, so the test asserts on the
# workers it actually observed and reports how many that was.
PROBE_COUNT = 12

_PROMPT = "In one sentence, what is the capital of France?"


def test_response_is_natural_language(attached_endpoint):
    """A single response is language rather than repeated-token output."""
    body = post(
        attached_endpoint,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_tokens": 256,
        },
    )
    assert_natural_language(message_text(body), "chat response")


def test_every_replica_that_serves_produces_language(attached_endpoint):
    """No replica in the fleet emits degenerate output.

    Sends ``PROBE_COUNT`` requests with unique prefixes so routing spreads them,
    attributes each response to the worker that served it, and fails naming the
    specific replicas that produced garbage. On a single-replica deployment this
    degrades gracefully into a repeated version of the test above.
    """
    seen = collections.defaultdict(list)
    unattributed = []

    for _ in range(PROBE_COUNT):
        body = post(
            attached_endpoint,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": unique_prompt(_PROMPT)}],
                "max_tokens": 128,
                "nvext": {"extra_fields": ["worker_id"]},
            },
        )
        text = message_text(body)
        reason = degeneracy_reason(text)
        worker = worker_id_of(body)
        if worker is None:
            unattributed.append(reason)
        else:
            seen[worker].append(reason)

    if unattributed:
        # No worker_id surfaced (nvext disabled, or no KV router). The fleet-wide
        # assertion is impossible, but the language assertion still holds.
        bad = [reason for reason in unattributed if reason]
        assert not bad, (
            f"{len(bad)}/{len(unattributed)} responses were not natural "
            f"language (worker attribution unavailable): {bad[0]}"
        )
        pytest.skip(
            "no worker_id in responses, so per-replica coverage cannot be "
            "asserted; the language assertion passed for all "
            f"{len(unattributed)} responses"
        )

    assert seen, "no responses were attributed to any worker"

    broken = {
        worker: [reason for reason in reasons if reason]
        for worker, reasons in seen.items()
    }
    broken = {worker: reasons for worker, reasons in broken.items() if reasons}

    assert not broken, (
        "replica(s) produced degenerate output while the deployment reported "
        "healthy:\n"
        + "\n".join(
            f"  worker {worker}: {len(reasons)}/{len(seen[worker])} responses "
            f"degenerate -- {reasons[0]}"
            for worker, reasons in broken.items()
        )
        + f"\nhealthy workers: "
        + ", ".join(
            str(worker) for worker in seen if worker not in broken
        )
        + f"\n({len(seen)} distinct worker(s) observed across {PROBE_COUNT} requests)"
    )
