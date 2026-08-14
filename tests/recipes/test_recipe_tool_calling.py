# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-calling protocol checks against an already-deployed recipe.

These do not restate the assertions. ``tests/frontend/test_tool_calling_sglang.py``
already expresses them as protocol classes whose methods take ``(client, model)``
-- an OpenAI client and a model id -- and nothing else. That signature is
deployment-agnostic by construction, so the same logic runs against any serving
frontend once it is handed a client pointed at one.

The upstream module launches SGLang and drives those classes from a single
pre-merge test. Here they are individually parametrized against
``--endpoint-url``, so a failure names the scenario rather than the whole batch.

Run::

    kubectl port-forward svc/<deployment>-frontend 8000:8000 -n <ns>
    pytest tests/recipes -m endpoint_only --endpoint-url=http://localhost:8000
"""

import pytest

openai = pytest.importorskip("openai")
pytest.importorskip("jsonschema")  # the protocol classes validate arg schemas

from tests.frontend.test_tool_calling_sglang import (  # noqa: E402
    TestToolCallingModelBehavior as _ModelBehaviour,
)
from tests.frontend.test_tool_calling_sglang import (  # noqa: E402
    TestToolCallingMultiTurn as _MultiTurn,
)
from tests.frontend.test_tool_calling_sglang import (  # noqa: E402
    TestToolCallingProtocol as _Protocol,
)

# Aliased deliberately: a name starting with `Test` in this namespace would be
# collected by pytest as a test class and error on fixtures it cannot see.

pytestmark = [
    pytest.mark.endpoint_only,
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
]

# Scenarios that need the model to *choose* well, rather than the protocol to be
# correctly implemented. They are inherently model-dependent, so upstream reruns
# them on AssertionError. Kept separate here so a behavioural wobble is legible
# as such instead of looking like a protocol regression.
# Scenarios that require the *decoder* to be constrained. Verified against this
# deployment: `tool_choice: "required"` returns finish_reason='stop' with plain
# text and no tool_calls, with reasoning on or off. The request is accepted
# (HTTP 200) and ignored, because the recipe configures no
# `guided_decoding_backend` and no structural tags. Not strict: they should
# start passing the moment a deployment enables guided decoding, and that is a
# pass we want to see rather than a new failure.
_NEEDS_GUIDED_DECODING = {
    "test_tool_choice_required_forces_a_tool_call",
}
# NOT here, though it needs the same machinery in principle:
# test_named_tool_choice_forces_specific_function PASSES on this deployment --
# naming a function steers the model reliably even unconstrained, whereas
# "required" does not. Measured, not assumed.

_BEHAVIOURAL = {
    "test_named_tool_choice_forces_specific_function",
    "test_parallel_multi_tool_request_includes_all_expected_tools",
    "test_array_argument_schema_valid",
    "test_no_tools_is_plain_text",
    "test_tool_result_is_consumed_and_final_answer_is_text",
    "test_chained_tool_use_search_then_calculate",
    "test_multiple_prior_tool_results_synthesize_to_text",
    "test_many_tools_prefers_calculator_for_math_question",
    "test_unicode_arguments_are_preserved",
    "test_system_instruction_encourages_tool_use",
}


def _cases():
    for cls in (_Protocol, _MultiTurn, _ModelBehaviour):
        for name in sorted(n for n in dir(cls) if n.startswith("test_")):
            yield pytest.param(cls, name, id=name)


@pytest.mark.parametrize("cls,method", _cases())
def test_tool_calling(cls, method, endpoint_client, attached_endpoint, request):
    """Drive one upstream tool-calling scenario against the deployment."""
    if method in _NEEDS_GUIDED_DECODING:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="needs a constrained decoder; this deployment sets no "
                "guided_decoding_backend and accepts-but-ignores tool_choice",
                strict=False,
            )
        )
    fn = getattr(cls(), method)
    if method not in _BEHAVIOURAL:
        fn(endpoint_client, attached_endpoint.model)
        return
    # Mirror upstream's rerun-on-AssertionError for model-choice scenarios
    # (see _run_with_assertion_reruns in the source module).
    last = None
    for _ in range(3):
        try:
            fn(endpoint_client, attached_endpoint.model)
            return
        except AssertionError as exc:
            last = exc
    raise AssertionError(
        f"{method} failed 3 attempts (model-behaviour dependent): {last}"
    )
