# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared health-check payload helper."""

from __future__ import annotations

import json

import pytest

from dynamo.common.backend.health_check import (
    HEALTH_CHECK_KEY,
    bos_token_id_or,
    build_health_check_payload,
    is_probe,
    parse_health_check_payload_cli,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_build_merges_extras_and_stamps_marker(monkeypatch):
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    payload = build_health_check_payload(
        bos_token_id=42, extras={"bootstrap_info": {"bootstrap_host": "fake"}}
    )
    assert payload["token_ids"] == [42]
    assert payload["bootstrap_info"] == {"bootstrap_host": "fake"}
    assert payload[HEALTH_CHECK_KEY] is True


def test_env_override_replaces_default_and_extras(monkeypatch):
    """Env override fully replaces the engine default + extras. Marker survives."""
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", json.dumps({"prompt": "hello"}))
    payload = build_health_check_payload(
        bos_token_id=42, extras={"bootstrap_info": {"bootstrap_host": "fake"}}
    )
    assert payload == {"prompt": "hello", HEALTH_CHECK_KEY: True}


def test_is_probe_strict_bool_match():
    # Only the bool True trips it; truthy non-bools (string "false",
    # integer 1) must not be mistaken for the marker.
    assert is_probe({"_HEALTH_CHECK": True}) is True
    assert is_probe({"_HEALTH_CHECK": False}) is False
    assert is_probe({"_HEALTH_CHECK": "true"}) is False
    assert is_probe({"_HEALTH_CHECK": 1}) is False
    assert is_probe({"token_ids": [1]}) is False


def test_env_override_empty_dict_wins_over_engine_default(monkeypatch):
    # Explicit `{}` is a valid operator override (e.g. canary that just
    # tests transport with no payload fields); must not fall back to the
    # engine default.
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", "{}")
    payload = build_health_check_payload(bos_token_id=42)
    assert payload == {HEALTH_CHECK_KEY: True}


def test_bos_token_id_or_accepts_direct_and_wrapped_tokenizers():
    class HFTok:
        bos_token_id = 7

    class Wrapper:
        tokenizer = HFTok()

    assert bos_token_id_or(HFTok()) == 7
    assert bos_token_id_or(Wrapper()) == 7
    assert bos_token_id_or(None) == 1
    assert bos_token_id_or(object(), default=5) == 5


@pytest.mark.parametrize(
    "value",
    ["not json", "[1, 2, 3]", "@/nonexistent/path/probe.json"],
    ids=["invalid-json", "non-object", "missing-file"],
)
def test_parse_cli_returns_none_on_bad_input(value):
    assert parse_health_check_payload_cli(value) is None


def test_parse_cli_json_object_stamps_marker():
    assert parse_health_check_payload_cli('{"token_ids": [7]}') == {
        "token_ids": [7],
        HEALTH_CHECK_KEY: True,
    }


def test_parse_cli_at_file_loads(tmp_path):
    p = tmp_path / "probe.json"
    p.write_text('{"token_ids": [9]}')
    assert parse_health_check_payload_cli(f"@{p}") == {
        "token_ids": [9],
        HEALTH_CHECK_KEY: True,
    }
