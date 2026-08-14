# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert the deployment answers correctly, and that *every* replica does.

Written after a 3-replica Kimi-K2.5 deployment on GB300 served pure token-0
(``!``) output from one of its three replicas while every Kubernetes signal
reported health: pods ``1/1 Running``, 0 restarts, ``Ready=True``, no error in
any worker log, and a TRT-LLM config byte-identical to the healthy replicas.
Nothing in a readiness probe samples what a worker actually produces.

Three properties are needed to catch that class of fault, and each is easy to
get wrong:

1. **Assert language, not truthiness.** ``"!!!!"`` is a non-empty string, so
   every ``assert response.content`` in this suite passed against it.
2. **Assert the answer, not just language.** A numerically degraded replica can
   emit fluent prose and still be wrong; only a known-answer question separates
   "generating text" from "computing correctly".
3. **Vary the prompt.** KV-aware routing pins a prefix to one worker, so a
   fixed prompt repeated N times measures one replica N times. The
   deployment-wide failure rate is only visible with unique prefixes.
"""

import collections

import pytest

from tests.recipes.helpers import (
    KNOWN_ANSWER_PROBES,
    answer_text,
    assert_answers,
    degeneracy_reason,
    post,
    unique_prompt,
    worker_id_of,
    wrong_answer_reason,
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

# Generous, because a reasoning deployment spends its first tokens inside the
# think block; too small a budget leaves `content` empty for reasons that have
# nothing to do with correctness.
ANSWER_TOKENS = 512

_FLEET_PROMPT, _FLEET_ACCEPTED = KNOWN_ANSWER_PROBES[0]


@pytest.mark.parametrize(
    "prompt,accepted",
    KNOWN_ANSWER_PROBES,
    ids=[prompt.split("?")[0][:28] for prompt, _ in KNOWN_ANSWER_PROBES],
)
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_known_answer_questions_are_answered_correctly(
    attached_endpoint, prompt, accepted
):
    """The deployment computes the right answer, not merely well-formed text."""
    body = post(
        attached_endpoint,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": ANSWER_TOKENS,
        },
    )
    assert_answers(answer_text(body), accepted, f"answer to {prompt!r}")


def test_every_replica_that_serves_answers_correctly(attached_endpoint):
    """No replica in the fleet is degenerate or wrong.

    Sends ``PROBE_COUNT`` requests with unique prefixes so routing spreads them,
    attributes each response to the worker that served it, and fails naming the
    specific replicas at fault. On a single-replica deployment this degrades
    gracefully into a repeated version of the test above.
    """
    seen = collections.defaultdict(list)
    unattributed = []

    for _ in range(PROBE_COUNT):
        body = post(
            attached_endpoint,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": unique_prompt(_FLEET_PROMPT)}],
                "max_tokens": ANSWER_TOKENS,
                "nvext": {"extra_fields": ["worker_id"]},
            },
        )
        text = answer_text(body)
        # Degeneracy first: "repeating one token" is a far more actionable
        # report than "paris not found" for the same broken replica.
        reason = degeneracy_reason(text) or wrong_answer_reason(text, _FLEET_ACCEPTED)
        worker = worker_id_of(body)
        if worker is None:
            unattributed.append(reason)
        else:
            seen[worker].append(reason)

    if unattributed:
        # No worker_id surfaced (nvext disabled, or no KV router). The fleet-wide
        # assertion is impossible, but the per-response assertion still holds.
        bad = [reason for reason in unattributed if reason]
        assert not bad, (
            f"{len(bad)}/{len(unattributed)} responses were degenerate or "
            f"incorrect (worker attribution unavailable): {bad[0]}"
        )
        pytest.skip(
            "no worker_id in responses, so per-replica coverage cannot be "
            f"asserted; all {len(unattributed)} responses answered correctly"
        )

    assert seen, "no responses were attributed to any worker"

    broken = {
        worker: [reason for reason in reasons if reason]
        for worker, reasons in seen.items()
    }
    broken = {worker: reasons for worker, reasons in broken.items() if reasons}

    assert not broken, (
        "replica(s) produced degenerate or incorrect output while the "
        "deployment reported healthy:\n"
        + "\n".join(
            f"  worker {worker}: {len(reasons)}/{len(seen[worker])} responses "
            f"bad -- {reasons[0]}"
            for worker, reasons in broken.items()
        )
        + "\nhealthy workers: "
        + ", ".join(str(worker) for worker in seen if worker not in broken)
        + f"\n({len(seen)} distinct worker(s) observed across {PROBE_COUNT} requests)"
    )
