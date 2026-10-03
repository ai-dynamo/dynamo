# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tool-calling helpers that the test suites share."""

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.utils import tool_calling

openai = pytest.importorskip("openai")
ChatCompletionChunk = pytest.importorskip("openai.types.chat").ChatCompletionChunk

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _tool_call_chunk() -> ChatCompletionChunk:
    """One streamed chunk in which the model asks for the ``echo`` tool."""
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }
    )


class _FakeClient:
    """Answers every request with a tool call, and records the timeouts it gets.

    ``turn_seconds`` advances the fake clock once per request, in place of a
    real request taking that long.
    """

    def __init__(self, clock, timeout, turn_seconds=0.0):
        self.clock = clock
        self.timeout = timeout
        self.turn_seconds = turn_seconds
        self.requests = 0
        self.timeouts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, *, timeout):
        self.timeouts.append(timeout)
        return self

    def _create(self, **request):
        self.requests += 1
        self.clock.now += self.turn_seconds
        return iter([_tool_call_chunk()])


@pytest.fixture
def clock(monkeypatch):
    """A fake clock for ``tool_calling`` only, so no test waits for real."""
    fake = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        tool_calling, "time", SimpleNamespace(monotonic=lambda: fake.now)
    )
    return fake


def _loop(client, deadline):
    return tool_calling.run_tool_loop(
        client,
        "m",
        [{"role": "user", "content": "go"}],
        _TOOLS,
        {"echo": lambda args: "ok"},
        deadline=deadline,
    )


def test_no_turn_starts_after_the_deadline(clock):
    """A model that keeps calling tools would otherwise run all six turns."""
    client = _FakeClient(clock, timeout=120.0, turn_seconds=10.0)

    with pytest.raises(AssertionError, match="passed its deadline after 1 turn"):
        _loop(client, deadline=clock.now + 5.0)

    assert client.requests == 1


@pytest.mark.parametrize(
    ("configured", "left", "expected"),
    [
        pytest.param(120.0, 30.0, 30.0, id="number-capped"),
        pytest.param(120.0, 500.0, 120.0, id="number-kept"),
        pytest.param(
            openai.Timeout(connect=5.0, read=600.0, write=600.0, pool=600.0),
            30.0,
            openai.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0),
            id="httpx-timeout-capped",
        ),
    ],
)
def test_each_request_waits_at_most_the_time_left(clock, configured, left, expected):
    client = _FakeClient(clock, timeout=configured)

    with pytest.raises(AssertionError, match="never produced a final text answer"):
        _loop(client, deadline=clock.now + left)

    assert client.timeouts == [expected] * tool_calling.MAX_TOOL_TURNS


def test_without_a_deadline_the_client_is_used_as_given(clock):
    """The SGLang suite passes no deadline and must see no change."""
    client = _FakeClient(clock, timeout=120.0, turn_seconds=10.0)

    with pytest.raises(AssertionError, match="never produced a final text answer"):
        _loop(client, deadline=None)

    assert client.timeouts == []
    assert client.requests == tool_calling.MAX_TOOL_TURNS


@pytest.mark.timeout(120)
@pytest.mark.parametrize("workers", [0, 2], ids=["in-process", "xdist"])
def test_a_recorded_property_reaches_the_junit_xml_from_an_xfail_test(
    tmp_path, workers
):
    """The body runs, and its property reaches the JUnit XML.

    In-process, pytest's ``record_property`` errors at setup whenever pytest
    writes JUnit XML, and the xfail marker still reports XFAIL with the body
    never run. CI runs the SGLang tool-calling module that way, with ``-n 0``.
    The xdist case checks that the value also arrives from a worker.
    """
    if workers:
        pytest.importorskip("xdist")
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import pathlib

            import pytest

            from tests.utils.tool_calling import property_recorder


            @pytest.mark.xfail(strict=False, reason="capability limit")
            def test_capability(request):
                pathlib.Path(__file__).with_name("body_ran").write_text("yes")
                property_recorder(request)("capability", "unsupported")
                raise AssertionError("cannot chain")
            """
        )
    )
    xml = tmp_path / "out.xml"
    # CI puts --ddtrace in PYTEST_ADDOPTS; the inner session must not report.
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(_REPO_ROOT / "pyproject.toml"),
            "--rootdir",
            str(_REPO_ROOT),
            "-p",
            "no:cacheprovider",
            "-q",
            f"--junitxml={xml}",
            *(["-n", str(workers)] if workers else []),
            str(probe),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
    )

    assert (tmp_path / "body_ran").exists(), result.stdout[-3000:]
    recorded = [(p.get("name"), p.get("value")) for p in ET.parse(xml).iter("property")]
    assert recorded == [("capability", "unsupported")], result.stdout[-3000:]
