#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from dynamo.frontend.utils import (
    handle_engine_error,
    make_backend_error,
    make_internal_error,
    request_id_from_context,
    resolve_chat_template,
)

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


class TestRequestIdFromContext:
    def test_uses_context_id_method(self):
        class Context:
            def id(self):
                return "ctx-123"

        assert request_id_from_context(Context()) == "ctx-123"

    def test_falls_back_without_context(self):
        request_id = request_id_from_context(None)

        assert len(request_id) == 16
        int(request_id, 16)


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
