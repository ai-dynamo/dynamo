# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.roles`."""

import pytest
from dynamo_test.roles import (
    Policy,
    PortName,
    Process,
    Role,
    RoleBinding,
    RoleTable,
    Sel,
    UnknownRole,
    at,
)


def binding(role, service=None, log_key=None, **kw):
    role = Role(role) if not isinstance(role, Role) else role
    return RoleBinding(
        role=role,
        service=service or str(role).title(),
        log_key=log_key or str(role),
        **kw,
    )


@pytest.fixture
def table():
    return RoleTable(
        {
            Role.FRONTEND: binding(Role.FRONTEND, service="Frontend"),
            Role.DECODE: binding(
                Role.DECODE,
                service="TRTLLMDecodeWorker",
                log_key="decode",
                processes={Process.ENGINE: "trtllm_worker"},
                ports={PortName.SERVICE: 8000, PortName.METRICS: 9090},
            ),
        }
    )


# ----------------------------------------------------------------- selectors


def test_at_accepts_a_plain_string_for_scenario_documents():
    assert at("decode").role is Role.DECODE
    assert at("DECODE").role is Role.DECODE


def test_at_rejects_an_unknown_role_at_the_call_site():
    """A typo must fail where it is written, not as a later lookup miss."""
    with pytest.raises(UnknownRole) as exc:
        at("decoder")
    assert "decode" in str(exc.value)


def test_a_selector_cannot_both_name_and_choose_a_replica():
    with pytest.raises(ValueError, match="either names a replica"):
        Sel(role=Role.DECODE, replica=1, policy=Policy.RANDOM)


@pytest.mark.parametrize("bad", [{"replica": -1}, {"fraction": 0}, {"fraction": 1.5}])
def test_a_selector_rejects_impossible_values(bad):
    with pytest.raises(ValueError):
        at("decode", **bad)


def test_selector_describes_itself_stably():
    assert at("decode", replica=1, rank=3).describe() == "decode/replica=1/rank=3"
    assert at("worker", policy=Policy.HOTTEST).describe() == "worker/policy=hottest"


def test_selectors_are_values():
    assert at("decode", replica=1) == at("decode", replica=1)
    assert at("decode", replica=1) != at("decode", replica=2)


# ---------------------------------------------------------------- resolution


def test_require_returns_the_binding(table):
    assert table.require(Role.DECODE).service == "TRTLLMDecodeWorker"
    assert table.require("decode").service == "TRTLLMDecodeWorker"


def test_an_unknown_role_raises_and_names_what_exists(table):
    """Never a default. An empty result reads exactly like a real measurement.

    A service name that resolved to nothing is *present-and-empty*, not absent,
    so a downstream absence check cannot recover the mistake.
    """
    with pytest.raises(UnknownRole) as exc:
        table.require("prefill")
    assert "decode" in str(exc.value)
    assert "frontend" in str(exc.value)


def test_lookup_and_require_agree(table):
    assert table["decode"] is table.require("decode")


def test_two_roles_may_not_share_a_log_key():
    """Sharing a key means one role's evidence silently overwrites the other's."""
    with pytest.raises(ValueError, match="reused"):
        RoleTable(
            {
                Role.PREFILL: binding(Role.PREFILL, log_key="worker"),
                Role.DECODE: binding(Role.DECODE, log_key="worker"),
            }
        )


def test_a_binding_filed_under_the_wrong_role_is_rejected():
    with pytest.raises(ValueError, match="declares role"):
        RoleTable({Role.PREFILL: binding(Role.DECODE)})


def test_table_is_a_mapping(table):
    assert len(table) == 2
    assert set(table) == {Role.FRONTEND, Role.DECODE}
    assert table.known() == ("decode", "frontend")


def test_resolve_goes_through_the_selector(table):
    assert table.resolve(at("decode", replica=2)).log_key == "decode"


# ------------------------------------------------------------------ bindings


def test_a_missing_process_names_the_ones_declared(table):
    with pytest.raises(KeyError, match="engine"):
        table.require("decode").process_pattern(Process.RANK)


def test_a_missing_port_names_the_ones_declared(table):
    assert table.require("decode").port() == 8000
    assert table.require("decode").port(PortName.METRICS) == 9090
    with pytest.raises(KeyError, match="service, metrics"):
        table.require("decode").port(PortName.GRPC)


def test_the_table_serialises_for_the_run_record(table):
    """Recording resolution is what makes it provable rather than assumed."""
    record = table.to_record()
    assert record["decode"]["service"] == "TRTLLMDecodeWorker"
    assert record["decode"]["log_key"] == "decode"
    assert record["decode"]["ports"] == {"service": 8000, "metrics": 9090}
    assert list(record) == ["decode", "frontend"]


def test_the_run_record_is_json_safe(table):
    import json

    assert (
        json.loads(json.dumps(table.to_record()))["frontend"]["service"] == "Frontend"
    )


def test_one_log_key_serves_both_the_directory_and_the_scrape(table):
    """The measured bug exists because these were two strings, not one.

    A lowercase directory name plus a hard-coded PascalCase alias list means any
    service outside the list resolves to nothing, silently.
    """
    decode = table.require("decode")
    assert decode.service == "TRTLLMDecodeWorker"
    assert decode.log_key == "decode"
    # There is exactly one spelling to use downstream, and it came from here.
    assert table.to_record()["decode"]["log_key"] == decode.log_key
