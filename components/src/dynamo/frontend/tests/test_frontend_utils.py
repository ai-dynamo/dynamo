#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from dynamo.frontend.utils import (
    backend_invalid_argument_to_http_error,
    handle_engine_error,
    make_backend_error,
    make_internal_error,
    resolve_chat_template,
    validate_legacy_guided_decoding_constraints,
)
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestMakeBackendError:  # FRONTEND.8 — BackendError construction
    def test_extracts_message(self):
        resp = {"status": "error", "message": "image load failed: 403"}
        err = make_backend_error(resp)
        assert err["error"]["message"] == "image load failed: 403"
        assert err["error"]["type"] == "backend_error"

    def test_none_message_uses_fallback(self):
        resp = {"status": "error", "message": None}
        err = make_backend_error(resp)
        assert err["error"]["message"] == "unknown backend error"

    def test_missing_message_uses_fallback(self):
        resp = {"status": "error"}
        err = make_backend_error(resp)
        assert err["error"]["message"] == "unknown backend error"


class TestValidateLegacyGuidedDecodingConstraints:
    @pytest.mark.parametrize(
        "payload",
        [
            {"guided_json": {"type": "object"}, "guided_regex": "a+"},
            {"guided_regex": "a+", "guided_grammar": 'root ::= "a"'},
        ],
    )
    def test_rejects_multiple_constraints(self, payload):
        with pytest.raises(
            InvalidArgument,
            match="Only one guided-decoding constraint can be set; received:",
        ):
            validate_legacy_guided_decoding_constraints(payload)

    def test_accepts_one_constraint_and_ignores_modifiers(self):
        validate_legacy_guided_decoding_constraints(
            {
                "guided_json": {"type": "object"},
                "guided_whitespace_pattern": r"[\n ]?",
                "guided_decoding_backend": "xgrammar",
            }
        )

    def test_empty_choice_is_not_an_active_constraint(self):
        # An empty list is a caller supplying no choices, which is no constraint.
        validate_legacy_guided_decoding_constraints({"guided_choice": []})

    @pytest.mark.parametrize("choice", ["yes", ("x", "y"), 5])
    def test_non_list_choice_is_rejected(self, choice):
        # Ignoring a malformed choice would generate unconstrained text for a
        # caller who believes it constrained the output.
        with pytest.raises(InvalidArgument, match="guided_choice must be a list"):
            validate_legacy_guided_decoding_constraints({"guided_choice": choice})

    def test_empty_string_message_uses_fallback(self):
        resp = {"status": "error", "message": ""}
        err = make_backend_error(resp)
        assert err["error"]["message"] == "unknown backend error"


class TestResolveChatTemplate:
    def test_jinja_backend_file_semantics(self, tmp_path):
        template_file = tmp_path / "chat_template.jinja"
        template_file.write_text("custom template\\n\n", encoding="utf-8")

        assert resolve_chat_template(str(tmp_path)) == "custom template\\n\n"
        assert resolve_chat_template(str(tmp_path), backend="vllm") == (
            "custom template\\n\n"
        )
        assert resolve_chat_template(str(tmp_path), backend="sglang") == (
            "custom template\n"
        )


class TestMakeInternalError:  # FRONTEND.8 — InternalError construction
    def test_default_message(self):
        err = make_internal_error("req-42")
        assert err["error"]["message"] == "Invalid engine response for request req-42"
        assert err["error"]["type"] == "internal_error"

    def test_custom_detail(self):
        err = make_internal_error("req-42", "connection reset")
        assert err["error"]["message"] == "connection reset"

    def test_none_detail_uses_default(self):
        err = make_internal_error("req-42", None)
        assert err["error"]["message"] == "Invalid engine response for request req-42"


class TestHandleEngineError:  # FRONTEND.8 — engine error → HTTP-friendly mapping
    def test_backend_error_dict(self):
        resp = {"status": "error", "message": "403 Forbidden"}
        err = handle_engine_error(resp, "req-1", logging.getLogger("test"))
        assert err["error"]["type"] == "backend_error"
        assert err["error"]["message"] == "403 Forbidden"

    def test_none_response(self):
        err = handle_engine_error(None, "req-1", logging.getLogger("test"))
        assert err["error"]["type"] == "internal_error"

    def test_missing_token_ids(self):
        err = handle_engine_error({"other": "data"}, "req-1", logging.getLogger("test"))
        assert err["error"]["type"] == "internal_error"


class TestBackendInvalidArgumentToHttpError:  # FRONTEND.8 — backend 4xx must not surface as 500
    def test_parses_code_and_message(self):
        err = backend_invalid_argument_to_http_error(
            ValueError(
                'BackendInvalidArgument: {"message":"The min_p and logit_bias '
                "sampling parameters are not yet supported with speculative "
                'decoding.","code":400}'
            )
        )
        assert err is not None
        assert err.code == 400
        assert "min_p" in err.message
        # The serialized envelope must not leak into the client-visible text.
        assert "BackendInvalidArgument" not in err.message

    def test_honours_a_non_400_status(self):
        err = backend_invalid_argument_to_http_error(
            ValueError('BackendInvalidArgument: {"message":"bad media","code":415}')
        )
        assert err is not None and err.code == 415

    def test_prefix_after_an_outer_wrapper(self):
        err = backend_invalid_argument_to_http_error(
            ValueError('Unknown: BackendInvalidArgument: {"message":"nope","code":400}')
        )
        assert err is not None and err.message == "nope"

    def test_unstructured_payload_still_becomes_400(self):
        err = backend_invalid_argument_to_http_error(
            ValueError("BackendInvalidArgument: grammar is not valid")
        )
        assert err is not None
        assert err.code == 400 and err.message == "grammar is not valid"

    def test_missing_code_defaults_to_400(self):
        err = backend_invalid_argument_to_http_error(
            ValueError('BackendInvalidArgument: {"message":"nope"}')
        )
        assert err is not None and err.code == 400

    @pytest.mark.parametrize("code", [0, 99, 600, 1000, -1])
    def test_out_of_range_code_defers_to_the_caller(self, code):
        # Rust degrades these to a 500, which is what the generic handler
        # already produces -- returning None is how we defer to it.
        assert (
            backend_invalid_argument_to_http_error(
                ValueError(f'BackendInvalidArgument: {{"message":"x","code":{code}}}')
            )
            is None
        )

    def test_boolean_code_is_not_a_status(self):
        err = backend_invalid_argument_to_http_error(
            ValueError('BackendInvalidArgument: {"message":"x","code":true}')
        )
        assert err is not None and err.code == 400

    @pytest.mark.parametrize(
        "text",
        [
            "CUDA out of memory",
            "BackendUnknown: something broke",
            '{"message":"no prefix","code":400}',
            "",
        ],
    )
    def test_unrelated_errors_are_left_alone(self, text):
        assert backend_invalid_argument_to_http_error(ValueError(text)) is None
