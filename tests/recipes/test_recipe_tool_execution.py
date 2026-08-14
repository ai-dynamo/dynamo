# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tool execution: the model calls a tool, the tool actually runs.

Every other tool-calling test in this repo verifies the *protocol* -- that a
well-formed ``tool_calls`` object comes back and that a hand-written
``role: tool`` message is consumed. None of them execute anything: the "tool
result" is a literal the test author typed, so a model that ignored the result
and hallucinated a plausible answer would still pass.

These tests close that loop. A real subprocess runs, and the value it returns is
a **secret generated at runtime and never present in the prompt**. The model
cannot produce it by guessing, by memorisation, or by reasoning -- the only path
from the question to the answer runs through the tool actually executing. That
is what makes the assertion un-fakeable.

What this exercises that protocol tests cannot:
  * the arguments the model emitted are usable by a real callee, not merely
    schema-valid (a plausible-but-wrong ``city`` passes a schema; a wrong
    ``user`` returns DENIED here)
  * a result the test did not author flows back through the chat template
  * multi-step chaining where the second call depends on the first call's
    real output
"""

import json
import os
import subprocess
import sys
import uuid

import pytest

pytest.importorskip("openai")

pytestmark = [
    pytest.mark.endpoint_only,
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
]

MAX_TURNS = 6


# --------------------------------------------------------------------------
# Real tools. These shell out on purpose: an in-process Python function would
# leave "did anything outside the test actually run?" unanswered.
# --------------------------------------------------------------------------
def _run_cli(script: str, arg: str, env_extra: dict) -> str:
    result = subprocess.run(
        [sys.executable, "-c", script, arg],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, **env_extra},
        check=True,
    )
    return result.stdout.strip()


_LOOKUP_CODE = (
    "import os, sys; "
    "print(os.environ['ACCESS_CODE'] if sys.argv[1].strip().lower() == 'alice' "
    "else 'DENIED')"
)
_LOOKUP_ID = (
    "import os, sys; "
    "print(os.environ['USER_ID'] if sys.argv[1].strip().lower() == 'alice' "
    "else 'UNKNOWN')"
)
_LOOKUP_QUOTA = (
    "import os, sys; "
    "print(os.environ['QUOTA'] if sys.argv[1].strip() == os.environ['USER_ID'] "
    "else '-1')"
)


def _tool_loop(client, model, messages, tools, dispatch, max_turns=MAX_TURNS):
    """Drive a real tool-execution loop until the model answers with text.

    Returns (final_text, calls) where ``calls`` records every tool the model
    asked for and what the tool actually returned -- so a test can assert the
    tool ran, not merely that the answer looks right.
    """
    calls = []
    convo = list(messages)
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model, messages=convo, tools=tools, max_tokens=1024
        )
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        if not tool_calls:
            return (choice.message.content or ""), calls

        # Echo the assistant turn back, including its tool_calls, or the model
        # has no record of having asked.
        convo.append(
            {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                pytest.fail(
                    f"model emitted unparseable arguments for {name}: "
                    f"{tc.function.arguments!r} ({exc})"
                )
            if name not in dispatch:
                pytest.fail(
                    f"model called {name!r}, which was never offered. "
                    f"Offered: {sorted(dispatch)}"
                )
            output = dispatch[name](args)  # <-- the tool really runs here
            calls.append({"name": name, "args": args, "output": output})
            convo.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(output)}
            )
    pytest.fail(
        f"model never produced a final text answer within {max_turns} turns; "
        f"calls so far: {calls}"
    )


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_model_call_executes_a_real_tool_and_uses_its_output(
    endpoint_client, attached_endpoint
):
    """The answer contains a secret only the executed tool could have supplied.

    The secret is generated per run and never appears in any prompt, so the
    assertion cannot be satisfied by hallucination, memorisation, or luck.
    """
    secret = uuid.uuid4().hex[:12]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_access_code",
                "description": (
                    "Look up the access code for a user. This is the only way "
                    "to obtain an access code."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string", "description": "the username"}
                    },
                    "required": ["user"],
                },
            },
        }
    ]

    def lookup_access_code(args):
        return _run_cli(
            _LOOKUP_CODE, str(args.get("user", "")), {"ACCESS_CODE": secret}
        )

    text, calls = _tool_loop(
        endpoint_client,
        attached_endpoint.model,
        [
            {
                "role": "user",
                "content": (
                    "Look up the access code for user alice, then tell me the "
                    "code exactly as returned."
                ),
            }
        ],
        tools,
        {"lookup_access_code": lookup_access_code},
    )

    assert calls, "the model never called the tool, so nothing was executed"
    assert calls[0]["name"] == "lookup_access_code"
    assert calls[0]["args"].get("user", "").strip().lower() == "alice", (
        f"model passed an unusable argument: {calls[0]['args']!r} -- schema-valid "
        "but wrong, which protocol-only tests cannot catch"
    )
    assert calls[0]["output"] == secret, (
        f"tool returned {calls[0]['output']!r}, expected the generated secret; "
        "the subprocess did not receive the argument it should have"
    )
    assert secret in text, (
        f"final answer did not contain the executed tool's output.\n"
        f"secret={secret!r}\nanswer={text[:400]!r}"
    )


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_chained_tools_second_call_uses_first_calls_real_output(
    endpoint_client, attached_endpoint
):
    """Two real executions, where the second is only satisfiable using the first.

    ``get_quota`` returns -1 unless handed the exact id ``get_user_id`` produced,
    and both are random per run. A correct final number therefore proves the
    model threaded real output from one execution into the next.
    """
    user_id = f"U-{uuid.uuid4().hex[:8]}"
    quota = str(uuid.uuid4().int % 9000 + 1000)  # 4 digits, unguessable
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_user_id",
                "description": "Resolve a username to its internal user id.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_quota",
                "description": (
                    "Get the storage quota for an internal user id. Requires the "
                    "id from get_user_id, not a username."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            },
        },
    ]

    dispatch = {
        "get_user_id": lambda a: _run_cli(
            _LOOKUP_ID, str(a.get("name", "")), {"USER_ID": user_id}
        ),
        "get_quota": lambda a: _run_cli(
            _LOOKUP_QUOTA,
            str(a.get("user_id", "")),
            {"USER_ID": user_id, "QUOTA": quota},
        ),
    }

    text, calls = _tool_loop(
        endpoint_client,
        attached_endpoint.model,
        [
            {
                "role": "user",
                "content": (
                    "What is alice's storage quota? Look up her user id first, "
                    "then use that id to get the quota. Report the number."
                ),
            }
        ],
        tools,
        dispatch,
    )

    names = [c["name"] for c in calls]
    assert "get_user_id" in names, f"never resolved the id; calls={names}"
    assert "get_quota" in names, f"never fetched the quota; calls={names}"
    assert names.index("get_user_id") < names.index(
        "get_quota"
    ), f"called get_quota before get_user_id: {names}"

    quota_call = calls[names.index("get_quota")]
    assert quota_call["args"].get("user_id") == user_id, (
        f"second call used {quota_call['args'].get('user_id')!r} instead of the "
        f"id the first call actually returned ({user_id!r})"
    )
    assert (
        quota_call["output"] == quota
    ), f"get_quota returned {quota_call['output']!r} -- it was handed the wrong id"
    assert quota in text, (
        f"final answer omitted the quota the tool actually returned.\n"
        f"quota={quota!r}\nanswer={text[:400]!r}"
    )
