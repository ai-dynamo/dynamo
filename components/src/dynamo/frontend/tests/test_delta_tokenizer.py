# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-state tests use a deterministic patch provider, never model weights."""

import gc
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from dynamo.frontend.delta_tokenizer import (
    DEFAULT_METRICS,
    DeltaBudget,
    DeltaConfig,
    DeltaMetrics,
    DeltaTokenizer,
    TokenizerIdentity,
    UnsupportedTransition,
    delta_request_context,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def reference(text):
    # A whole-input function whose boundary changes when "a" gains an accent.
    return [ord(char) for char in text.replace("a\u0301", "á")]


class PatchSession:
    base_bytes = 0

    def __init__(self):
        self.text = ""
        self.tokens = []
        self.closed = False

    def append(self, text):
        self.text += text
        tokens = reference(self.text)
        cut = 0
        while (
            cut < min(len(tokens), len(self.tokens)) and tokens[cut] == self.tokens[cut]
        ):
            cut += 1
        self.tokens = tokens
        return cut, tokens[cut:]

    def close(self):
        self.closed = True
        self.text = ""
        self.tokens = []


@pytest.fixture
def identity():
    return TokenizerIdentity(
        "model", "revision", "artifact", "loader", "config", "default"
    )


@pytest.fixture
def manager(identity):
    sessions = []

    def factory():
        session = PatchSession()
        sessions.append(session)
        return session

    instance = DeltaTokenizer(
        reference,
        identity,
        factory,
        DeltaConfig(),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    )
    instance.test_sessions = sessions
    yield instance
    instance.close()


def encode(manager, text, **kwargs):
    return manager.tokenize(
        "session", text, scope="tenant", template_identity="template", **kwargs
    )


@pytest.mark.parametrize(
    "pieces",
    [
        ["", "a", "\u0301", " ", " hello", "\n", "🙂", "漢字", "\0", "\ud800"],
        ["a" * 20000, "a" * 2000, "\u0301"],
        [" leading ", "  ", "\n\n", "trailing "],
        ["<BOS>", "<added>", "<EOS>", '{"tool_calls":[]}', "tool result"],
    ],
)
def test_append_parity(manager, pieces):
    text = ""
    for piece in pieces:
        text += piece
        assert encode(manager, text) == reference(text)
    assert manager.stats()["mismatches"] == 0


def test_duplicate_and_non_append(manager):
    assert encode(manager, "abc") == reference("abc")
    assert encode(manager, "abc") == reference("abc")
    assert encode(manager, "ax") == reference("ax")
    assert encode(manager, "a") == reference("a")
    assert manager.stats()["duplicate"] == 1
    assert manager.stats()["invalidated_non_append"] == 2


def test_template_change(manager):
    encode(manager, "abc")
    assert manager.tokenize(
        "session", "abcdef", scope="tenant", template_identity="new"
    ) == reference("abcdef")
    assert manager.stats()["invalidated_template"] == 1


@pytest.mark.parametrize(
    "field", ["model", "revision", "artifact", "loader", "configuration", "backend"]
)
def test_identity_change_is_not_admitted(manager, identity, field):
    encode(manager, "abc")
    assert encode(
        manager, "abcd", tokenizer_identity=replace(identity, **{field: "changed"})
    ) == reference("abcd")
    assert manager.stats()["fallback_identity"] == 1


def test_isolation_and_missing_key(manager):
    encode(manager, "abc")
    assert manager.tokenize(
        "session", "abcd", scope="other", template_identity="template"
    ) == reference("abcd")
    assert manager.tokenize(
        None, "x", scope="tenant", template_identity="template"
    ) == reference("x")
    assert manager.tokenize(
        "session", "x", scope=None, template_identity="template"
    ) == reference("x")
    assert manager.stats()["cold"] == 2
    assert manager.stats()["fallback_missing_session"] == 2


def test_ttl_evicts_raw_state(manager):
    now = [0.0]
    manager.clock = lambda: now[0]
    encode(manager, "secret")
    now[0] = 301.0
    manager.expire()
    assert manager.stats()["entries"] == 0
    assert all(session.closed for session in manager.test_sessions)
    assert all(
        session.text == "" and session.tokens == [] for session in manager.test_sessions
    )
    assert manager.stats()["expired"] == 1
    assert encode(manager, "secret more") == reference("secret more")


def test_entry_bound_and_lru(identity):
    sessions = []

    def factory():
        session = PatchSession()
        sessions.append(session)
        return session

    with DeltaTokenizer(
        reference,
        identity,
        factory,
        DeltaConfig(max_entries=1),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        encode(instance, "a")
        instance.tokenize("other", "b", scope="tenant", template_identity="template")
        assert instance.stats()["entries"] == 1
        assert instance.stats()["evicted"] == 1
        assert sessions[0].closed


def test_byte_bound_falls_back(identity):
    with DeltaTokenizer(
        reference,
        identity,
        PatchSession,
        DeltaConfig(max_bytes=1024),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        assert encode(instance, "x" * 100) == reference("x" * 100)
        assert instance.stats()["retained_bytes"] == 0
        assert instance.stats()["fallback_budget"] == 1


def test_shadow_mismatch_never_served(identity, caplog):
    class WrongSession(PatchSession):
        def append(self, text):
            cut, tokens = super().append(text)
            if len(self.text) > 1:
                tokens = [999]
            return cut, tokens

    with DeltaTokenizer(
        reference,
        identity,
        WrongSession,
        DeltaConfig(verify_rate=1),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        encode(instance, "secret")
        assert encode(instance, "secret suffix") == reference("secret suffix")
        assert instance.stats()["mismatches"] == 1
        assert instance.stats()["entries"] == 0
    assert "secret" not in caplog.text
    assert "session" not in caplog.text


def test_unsupported_transition_uses_reference(identity):
    class RefusingSession(PatchSession):
        def append(self, text):
            raise UnsupportedTransition

    with DeltaTokenizer(
        reference,
        identity,
        RefusingSession,
        DeltaConfig(),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        assert encode(instance, "abc") == reference("abc")
        assert instance.stats()["fallback_uncertified"] == 1


def test_concurrent_same_session_has_one_writer_and_cancelled_writer_cannot_publish(
    identity,
):
    entered, release = Event(), Event()
    cancelled = Event()

    class BlockingSession(PatchSession):
        def append(self, text):
            if text == "d":
                entered.set()
                assert release.wait(5)
            return super().append(text)

    with DeltaTokenizer(
        reference,
        identity,
        BlockingSession,
        DeltaConfig(),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        encode(instance, "abc")
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(encode, instance, "abcd", cancelled=cancelled.is_set)
            assert entered.wait(5)
            assert encode(instance, "abce") == reference("abce")
            cancelled.set()
            release.set()
            assert first.result() == reference("abcd")
        assert instance.stats()["fallback_busy"] == 1
        assert instance.stats()["entries"] == 0
        assert encode(instance, "abcdef") == reference("abcdef")


def test_independent_sessions_do_not_hold_global_lock(identity):
    entered, release = Event(), Event()

    class BlockingSession(PatchSession):
        def append(self, text):
            if text == "blocked":
                entered.set()
                assert release.wait(5)
            return super().append(text)

    with DeltaTokenizer(
        reference,
        identity,
        BlockingSession,
        DeltaConfig(),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(encode, instance, "blocked")
            assert entered.wait(5)
            second = pool.submit(
                instance.tokenize,
                "other",
                "free",
                scope="tenant",
                template_identity="template",
            )
            try:
                assert second.result(timeout=2) == reference("free")
            finally:
                release.set()
            assert first.result() == reference("blocked")


def test_close_during_encode_prevents_stale_publication(identity):
    entered, release = Event(), Event()

    class BlockingSession(PatchSession):
        def append(self, text):
            entered.set()
            assert release.wait(5)
            return super().append(text)

    instance = DeltaTokenizer(
        reference,
        identity,
        BlockingSession,
        DeltaConfig(),
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    )
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(encode, instance, "abc")
        assert entered.wait(5)
        instance.close()
        release.set()
        assert first.result() == reference("abc")
    assert instance.stats()["retained_bytes"] == 0
    assert instance.stats()["entries"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_entries": 0},
        {"max_bytes": 0},
        {"ttl_secs": 0},
        {"ttl_secs": float("inf")},
        {"verify_rate": -0.1},
        {"verify_rate": 1.1},
        {"verify_rate": float("nan")},
    ],
)
def test_config_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        DeltaConfig(**kwargs)


def test_random_long_append_sequence(manager):
    rng = random.Random(42)
    text = ""
    for _ in range(1000):
        text += rng.choice(
            ["a", "\u0301", " ", "\n", "🙂", "漢", "<added>", "tool result"]
        )
        assert encode(manager, text) == reference(text)


def test_shared_budget_covers_multiple_models(identity):
    budget = DeltaBudget(1, 1000000)
    first = DeltaTokenizer(
        reference,
        identity,
        PatchSession,
        DeltaConfig(),
        budget=budget,
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    )
    second = DeltaTokenizer(
        reference,
        replace(identity, model="other"),
        PatchSession,
        DeltaConfig(),
        budget=budget,
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    )
    try:
        encode(first, "abc")
        assert encode(second, "xyz") == reference("xyz")
        assert second.stats()["fallback_budget"] == 1
        first.close()
        assert encode(second, "xyz") == reference("xyz")
        assert second.stats()["entries"] == 1
    finally:
        first.close()
        second.close()


def test_metrics_have_no_prompt_or_session_labels(manager):
    encode(manager, "private prompt")
    encode(manager, "private prompt longer")
    exposition = generate_latest(manager.metrics.registry).decode()
    assert "private prompt" not in exposition
    assert "session" not in exposition
    assert 'outcome="cold"' in exposition
    assert 'outcome="incremental"' in exposition


def test_expiry_during_encode_retires_generation(identity):
    entered, release = Event(), Event()
    now = [0.0]

    class BlockingSession(PatchSession):
        def append(self, text):
            entered.set()
            assert release.wait(5)
            return super().append(text)

    with DeltaTokenizer(
        reference,
        identity,
        BlockingSession,
        DeltaConfig(),
        clock=lambda: now[0],
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        with ThreadPoolExecutor(1) as pool:
            request = pool.submit(encode, instance, "abc")
            assert entered.wait(5)
            now[0] = 301
            instance.expire()
            assert instance.stats()["retained_bytes"] > 0
            release.set()
            assert request.result() == reference("abc")
        assert instance.stats()["retained_bytes"] == 0
        assert instance.stats()["entries"] == 0


@pytest.mark.parametrize(
    "origin", [None, "responses", "completions", "generate", "anthropic"]
)
def test_unsupported_api_context_never_admits(origin, manager):
    context = SimpleNamespace(
        metadata={
            "dynamo.frontend.api": origin,
            "delta-tokenizer-session": "session",
            "delta-tokenizer-scope": "tenant",
        }
    )
    assert delta_request_context(manager, context) == {}
    assert manager.stats()["entries"] == 0


def test_chat_context_requires_explicit_scope_and_session(manager):
    context = SimpleNamespace(
        metadata={"dynamo.frontend.api": "chat_completions"},
        is_stopped=lambda: False,
        is_killed=lambda: False,
    )
    assert delta_request_context(manager, context) == {}
    context.metadata.update(
        {"delta-tokenizer-session": "session", "delta-tokenizer-scope": "tenant"}
    )
    args = delta_request_context(manager, context)
    assert args["delta_tokenizer"] is manager
    assert args["delta_session_id"] == "session"
    assert args["delta_cancelled"]() is False


def test_background_seed_returns_reference_without_waiting(identity):
    entered, release = Event(), Event()

    class SlowSession(PatchSession):
        def append(self, text):
            entered.set()
            assert release.wait(5)
            return super().append(text)

    with DeltaTokenizer(
        reference,
        identity,
        SlowSession,
        DeltaConfig(),
        background_seed=True,
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        try:
            assert encode(instance, "ab") == reference("ab")
            assert encode(instance, "abc") == reference("abc")
            assert entered.wait(5)
            assert encode(instance, "abcd") == reference("abcd")
            assert instance.stats()["fallback_busy"] == 1
        finally:
            release.set()
        instance.wait_for_idle()
        assert encode(instance, "abcde") == reference("abcde")
        assert instance.stats()["incremental"] == 1


def test_cancelled_background_seed_cannot_publish(identity):
    entered, release, cancelled = Event(), Event(), Event()

    class SlowSession(PatchSession):
        def append(self, text):
            entered.set()
            assert release.wait(5)
            return super().append(text)

    with DeltaTokenizer(
        reference,
        identity,
        SlowSession,
        DeltaConfig(),
        background_seed=True,
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        assert encode(instance, "ab") == reference("ab")
        assert encode(instance, "abc", cancelled=cancelled.is_set) == reference("abc")
        assert entered.wait(5)
        cancelled.set()
        release.set()
        instance.wait_for_idle()
        assert instance.stats()["entries"] == 0
        assert instance.stats()["retained_bytes"] == 0


def test_reference_snapshot_budget_released_after_last_writer(identity):
    with DeltaTokenizer(
        reference,
        identity,
        PatchSession,
        DeltaConfig(),
        fixed_bytes=100,
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    ) as instance:
        assert instance.stats()["retained_bytes"] == 100
        encode(instance, "abc")
    assert instance.stats()["retained_bytes"] == 0


def test_model_retirement_releases_process_budget(identity):
    budget = DeltaBudget(1, 1000000)
    instance = DeltaTokenizer(
        reference,
        identity,
        PatchSession,
        DeltaConfig(),
        budget=budget,
        fixed_bytes=100,
        metrics=DeltaMetrics(CollectorRegistry()),
        start_cleanup=False,
    )
    encode(instance, "abc")
    del instance
    gc.collect()
    assert budget.adjust(1, 1000000)
    assert budget.adjust(-1, -1000000)


@pytest.mark.parametrize("field", ["text", "tokens"])
def test_corrupt_cached_state_uses_reference(manager, field):
    encode(manager, "abcdef")
    entry = next(iter(manager._entries.values()))
    if field == "text":
        entry.text = "abc"
    else:
        entry.tokens = (999,) + entry.tokens[1:]
    assert encode(manager, "abcdefg") == reference("abcdefg")
    assert manager.stats()["fallback_corrupt"] == 1
    assert manager.stats()["entries"] == 0


def test_downstream_mutation_and_failed_request_cannot_mutate_cached_tokens(manager):
    result = encode(manager, "abc")
    result[0] = 999
    assert encode(manager, "abc") == reference("abc")
    assert encode(manager, "abcd") == reference("abcd")


def test_loader_fallback_metrics_only_when_enabled():
    labels = {"outcome": "fallback_loader"}
    name = "dynamo_frontend_delta_tokenizer_events_total"
    before = DEFAULT_METRICS.registry.get_sample_value(name, labels) or 0
    assert delta_request_context(None, None) == {}
    assert (DEFAULT_METRICS.registry.get_sample_value(name, labels) or 0) == before
    assert delta_request_context(None, None, enabled=True) == {}
    assert DEFAULT_METRICS.registry.get_sample_value(name, labels) == before + 1


def test_stale_generation_publication_does_not_replace_state(manager):
    encode(manager, "abc")
    key, entry = next(iter(manager._entries.items()))
    assert manager._publish(key, entry, "wrong", [999], entry.generation - 1) is False
    assert encode(manager, "abcd") == reference("abcd")


def test_out_of_range_cached_ids_fail_closed(manager):
    encode(manager, "abc")
    entry = next(iter(manager._entries.values()))
    entry.tokens = (-1,)
    assert encode(manager, "abcd") == reference("abcd")
    assert manager.stats()["fallback_corrupt"] == 1
