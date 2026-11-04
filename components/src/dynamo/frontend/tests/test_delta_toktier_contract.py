# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned vendor contract and optional staged family-matrix tests."""
import json
from types import SimpleNamespace

import pytest

from dynamo.frontend import toktier_session as adapter
from dynamo.frontend.delta_tokenizer import DeltaConfig, UnsupportedTransition

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class MissingSpecialTokenFlag:
    """A backend without the tokenizers>=0.23 encode_special_tokens flag."""

    def to_str(self):
        return json.dumps({"truncation": None, "padding": None})


class FakeFastTokenizer:
    __module__ = "transformers.fake"
    split_special_tokens = False

    def __init__(self):
        self.backend_tokenizer = MissingSpecialTokenFlag()

    def encode(self, *args, **kwargs):
        raise AssertionError("the adapter should decline before encoding")

    def __call__(self, *args, **kwargs):
        raise AssertionError("the adapter should decline before calling")


def report():
    return {
        "certification": {
            "effective_verdict": "certified",
            "state": "certified_source",
        },
        "probe": {"oracle_version": "0.23.3"},
        "supported_untested": [],
        "experimental_waivers": [],
        "planned_backend": "fast_cpu",
        "backend": "fast_cpu",
        "session_repair": {"status": "active", "last": {"reason": None}},
    }


def test_missing_special_token_flag_is_unavailable(monkeypatch):
    monkeypatch.setattr(adapter, "_AVAILABLE", True)
    monkeypatch.setattr(adapter, "ToktierError", RuntimeError, raising=False)
    monkeypatch.setattr(
        adapter,
        "version",
        lambda package: "0.3.0" if package == "toktier" else "0.22.2",
    )
    monkeypatch.setattr(
        adapter, "PreTrainedTokenizerFast", FakeFastTokenizer, raising=False
    )
    monkeypatch.setattr(
        adapter, "HfCoreTokenizer", MissingSpecialTokenFlag, raising=False
    )

    assert (
        adapter.create_delta_tokenizer(FakeFastTokenizer(), ".", "model", DeltaConfig())
        is None
    )


@pytest.mark.parametrize("value", [None, [], {}, {"certification": []}, {"probe": {}}])
def test_unknown_certification_schema_is_declined(value):
    assert not adapter._certified(value)


def install_vendor(monkeypatch, patch, identity):
    pytest.importorskip("toktier")
    closed = []
    session = SimpleNamespace(append_patch=lambda text: patch)
    context = SimpleNamespace(
        __enter__=lambda: session,
        __exit__=lambda *args: None,
    )
    tokenizer = SimpleNamespace(
        explain=report, session=lambda: context, close=lambda: closed.append(True)
    )
    monkeypatch.setattr(adapter, "load", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(adapter, "check_core_identity", lambda *args: identity)
    return closed


@pytest.mark.parametrize("identity", [None, {}, {"status": "unknown"}])
def test_unknown_loader_identity_closes_and_declines(monkeypatch, identity):
    closed = install_vendor(monkeypatch, None, identity)
    with pytest.raises(UnsupportedTransition):
        adapter._Session("family", object(), 1024)
    assert closed == [True]


@pytest.mark.parametrize(
    "cut, ids",
    [(-1, []), (0.5, []), (True, []), (0, [-1]), (0, [2**32]), (0, ["1"])],
)
def test_invalid_vendor_patch_is_declined(monkeypatch, cut, ids):
    closed = install_vendor(
        monkeypatch,
        SimpleNamespace(replace_from=cut, replacement_ids=ids),
        {"status": "equivalent"},
    )
    session = adapter._Session("family", object(), 1024)
    try:
        with pytest.raises(UnsupportedTransition):
            session.append("text")
    finally:
        session.close()
    assert closed == [True]
