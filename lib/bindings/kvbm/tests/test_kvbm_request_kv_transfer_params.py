# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the `kv_transfer_params_json` FFI hop on `KvbmRequest`.

These tests exercise the pyo3 shim directly — they do not require vLLM.
They cover both the golden path (well-formed params serialize and reach the
Rust side without mutation) and adversarial inputs (non-serializable Python
objects, invalid JSON, empty/None cases, exotic dict shapes).
"""

from __future__ import annotations

import json

import kvbm
import pytest

if not kvbm.v2.is_available():
    pytest.skip(
        "kvbm v2 feature not compiled; skipping KvbmRequest tests",
        allow_module_level=True,
    )

KvbmRequest = kvbm.v2.KvbmRequest


# ---------------------------------------------------------------------------
# Golden paths
# ---------------------------------------------------------------------------


def _build(kv_transfer_params_json=None, **overrides):
    """Construct a KvbmRequest with sensible defaults for these tests."""
    kwargs = {
        "request_id": "req-test",
        "tokens": [1, 2, 3, 4],
        "lora_name": None,
        "salt_hash": None,
        "max_tokens": 128,
        "kv_transfer_params_json": kv_transfer_params_json,
    }
    kwargs.update(overrides)
    return KvbmRequest(**kwargs)


class TestGolden:
    def test_accepts_none(self):
        """No params supplied is the common case — must not raise."""
        req = _build(kv_transfer_params_json=None)
        assert req.request_id == "req-test"

    def test_accepts_empty_object(self):
        """An empty `{}` is valid JSON and should be accepted."""
        _build(kv_transfer_params_json=json.dumps({}))

    def test_accepts_typical_mooncake_payload(self):
        """Real-world vLLM Mooncake-style params must pass through."""
        payload = {
            "transfer_id": "abc-123",
            "remote_engine_id": "engine-7",
            "remote_host": "10.0.0.1",
            "remote_port": 9000,
            "remote_block_ids": [1, 2, 3, 4, 5],
            "do_remote_prefill": True,
            "do_remote_decode": False,
        }
        _build(kv_transfer_params_json=json.dumps(payload))

    def test_accepts_nested_structures(self):
        """Arbitrary nesting permitted under `dict[str, Any]`."""
        payload = {
            "peers": [
                {"host": "a", "port": 1},
                {"host": "b", "port": 2, "opts": {"enabled": True}},
            ],
            "counts": {"blocks": 128, "tokens": 4096},
            "nullable": None,
        }
        _build(kv_transfer_params_json=json.dumps(payload))

    @pytest.mark.parametrize(
        "value",
        [
            {"flag": True},
            {"int_val": -42},
            {"float_val": 3.14},
            {"str_val": "hello"},
            {"list_val": [1, 2.0, "three", None, True]},
            {"unicode": "日本語🚀"},
        ],
    )
    def test_accepts_all_json_primitive_types(self, value):
        _build(kv_transfer_params_json=json.dumps(value))


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------


class TestAdversarialJson:
    """Strings that parse poorly or not at all must raise on the Rust side."""

    @pytest.mark.parametrize(
        "bad",
        [
            "not json at all",
            "{unterminated",
            "{'single': 'quotes'}",  # single quotes are not valid JSON
            "{trailing: comma,}",
            "",  # empty string is not valid JSON
        ],
    )
    def test_malformed_json_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="invalid kv_transfer_params JSON"):
            _build(kv_transfer_params_json=bad)


class TestAdversarialPython:
    """
    The Python side of the contract is that callers serialize with
    ``json.dumps`` before handing the string to the shim. These tests pin
    that the loud failure mode (TypeError from json.dumps) happens in
    Python as expected for values that are not JSON-serializable.
    """

    @pytest.mark.parametrize(
        "unserializable",
        [
            {"blob": b"raw bytes"},
            {"set": {1, 2, 3}},
            {"custom": object()},
        ],
    )
    def test_json_dumps_raises_on_unserializable(self, unserializable):
        with pytest.raises(TypeError):
            json.dumps(unserializable)


# ---------------------------------------------------------------------------
# Boundary interaction with other KvbmRequest fields
# ---------------------------------------------------------------------------


class TestFieldInteractions:
    def test_max_tokens_still_required_even_with_params(self):
        """max_tokens=None must still raise regardless of kv_transfer_params."""
        with pytest.raises(ValueError, match="max_tokens is required"):
            KvbmRequest(
                request_id="req-x",
                tokens=[1, 2, 3],
                lora_name=None,
                salt_hash=None,
                max_tokens=None,
                kv_transfer_params_json=json.dumps({"transfer_id": "z"}),
            )

    def test_params_can_coexist_with_lora_and_salt(self):
        _build(
            lora_name="lora-a",
            salt_hash="salt-42",
            kv_transfer_params_json=json.dumps({"do_remote_prefill": True}),
        )
