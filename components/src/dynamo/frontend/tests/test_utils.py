#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from dynamo.frontend.utils import (
    handle_engine_error,
    make_backend_error,
    make_internal_error,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestMakeBackendError:
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


class TestMakeInternalError:
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


class TestHandleEngineError:
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
