# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay agent turns through the frontend with the mocker emitting their exact tokens.

The fixture is a hand-written agent trajectory in the nvidia/Open-SWE-Traces
schema (reasoning, parallel tool calls, whitespace-edged edit arguments), built
by `python -m tests.frontend.trace_replay build --teacher qwen3.5-0.8b
--rows-file .../source_rows.json --variant-every 1`. Every case is sent through
chat completions, Responses and Messages, streaming and unary; each parsed reply
must match the trajectory within the tolerance tiers of
`tests/frontend/trace_replay/compare.py`, apart from the known gaps below.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Generator

import pytest

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.frontend.trace_replay.runner import Fixtures, Replay, RunConfig
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import ServicePorts

TEST_MODEL = "Qwen/Qwen3.5-0.8B"
FIXTURES = Path(__file__).parent / "trace_replay" / "fixtures" / "handwritten_qwen35"
ACCEPTED_TIERS = {"exact", "ws_edges", "trunc_ok"}
# Frontend gaps this replay currently exposes, keyed by (endpoint, mode, tier,
# category or stream violation); "*" matches any endpoint or mode. Remove an
# entry once the gap is fixed so the test guards the fix.
KNOWN_GAPS: dict[tuple[str, str, str], str] = {
    ("*", "*", "ws_args"): "qwen3_coder trims whitespace at the edges of string "
    "arguments, e.g. the indentation and newline of a str_replace new_str",
    ("chat", "stream", "delta-after-finish"): "a script cut inside a parallel "
    "tool call streams finish_reason=tool_calls, then the jail flushes the rest",
    ("chat", "stream", "multiple-finish-reasons"): "same flush as delta-after-finish",
    ("responses", "stream", "done-sentinel"): "Responses streams end with the "
    "chat-completions [DONE] sentinel after response.completed",
    ("messages", "stream", "openai-done-sentinel"): "Anthropic streams end with "
    "the chat-completions [DONE] sentinel after message_stop",
    ("messages", "stream", "delta-for-closed-block"): "the newline between "
    "parallel tool calls is sent as a text_delta after its block was stopped",
}


def _known_gap(endpoint: str, mode: str, problem: str) -> bool:
    return any(
        (e, m, problem) in KNOWN_GAPS for e in (endpoint, "*") for m in (mode, "*")
    )


pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
]


@pytest.fixture(scope="function")
def replay_frontend(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports: object,
    dynamo_dynamic_ports: ServicePorts,
    predownload_tokenizers: object,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]
    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--enable-anthropic-api"],
        terminate_all_matching_process_names=False,
    ):
        with MockerWorkerProcess(
            request,
            TEST_MODEL,
            frontend_port,
            system_port,
            extra_args=[
                "--response-replay-trace-path",
                str(FIXTURES / "replay.jsonl"),
                "--dyn-tool-call-parser",
                "qwen3_coder",
                "--dyn-reasoning-parser",
                "qwen3",
            ],
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=TEST_MODEL
            )
            yield frontend_port


@pytest.mark.timeout(300)
def test_replayed_agent_turns_match_trajectory(replay_frontend: int) -> None:
    replay = Replay(
        Fixtures.load(FIXTURES),
        RunConfig(
            base_url=f"http://localhost:{replay_frontend}",
            model=TEST_MODEL,
            concurrency=4,
        ),
    )
    asyncio.run(replay.run())

    assert replay.results, "no replay requests were sent"
    failures = []
    for result in replay.results:
        problems = set(result.violations)
        if result.tier not in ACCEPTED_TIERS:
            problems.update(result.categories or [result.tier])
        unexplained = sorted(
            problem
            for problem in problems
            if not _known_gap(result.endpoint, result.mode, problem)
        )
        if unexplained:
            failures.append(
                f"{result.key} {result.endpoint}/{result.mode}: {unexplained} "
                f"{result.detail}"
            )
    assert not failures, "\n".join(failures)

    cross = replay.cross_checks()["counts"]
    disagreements = {
        cell: {k: v for k, v in counts.items() if k != "compared"}
        for cell, counts in cross.items()
        if set(counts) - {"compared"}
    }
    assert not disagreements, disagreements
