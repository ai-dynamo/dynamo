#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


from dynamo.frontend.utils import make_backend_error, make_internal_error


class TestMakeBackendError:
    def test_extracts_message(self):
        resp = {"status": "error", "message": "image load failed: 403"}
        err = make_backend_error(resp, "req-1")
        assert err["error"]["message"] == "image load failed: 403"
        assert err["error"]["type"] == "backend_error"

    def test_none_message_uses_fallback(self):
        resp = {"status": "error", "message": None}
        err = make_backend_error(resp, "req-1")
        assert err["error"]["message"] == "unknown backend error"

    def test_missing_message_uses_fallback(self):
        resp = {"status": "error"}
        err = make_backend_error(resp, "req-1")
        assert err["error"]["message"] == "unknown backend error"

    def test_empty_string_message_uses_fallback(self):
        resp = {"status": "error", "message": ""}
        err = make_backend_error(resp, "req-1")
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
