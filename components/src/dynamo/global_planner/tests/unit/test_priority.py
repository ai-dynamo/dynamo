# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pool priority declaration and resolution."""

import pytest
from pydantic import ValidationError

from dynamo.global_planner.priority import (
    DEFAULT_POOL_PRIORITY,
    DEFAULT_SELECTOR,
    LOWEST_PRIORITY,
    PriorityCondition,
    PriorityConfig,
    PriorityContext,
    PriorityResolver,
    outranks,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _resolver(**kwargs) -> PriorityResolver:
    return PriorityResolver(PriorityConfig(**kwargs))


# ---------------------------------------------------------------------------- #
# Polarity                                                                     #
# ---------------------------------------------------------------------------- #


def test_higher_priority_is_more_important():
    assert LOWEST_PRIORITY == 0
    assert outranks(1, 0)
    assert outranks(100, 10)
    assert not outranks(0, 1)
    assert not outranks(5, 5)


def test_negative_priority_is_rejected():
    with pytest.raises(ValidationError):
        PriorityConfig(pools=[{"selector": "ns/a", "priority": -1}])
    with pytest.raises(ValidationError):
        PriorityConfig(default=-1)


# ---------------------------------------------------------------------------- #
# Defaults for unseen pools                                                    #
# ---------------------------------------------------------------------------- #


def test_unmatched_pool_takes_the_default():
    resolved = _resolver().resolve("ns/never-seen", "prefill")
    assert resolved.priority == DEFAULT_POOL_PRIORITY
    assert resolved.selector == DEFAULT_SELECTOR


def test_default_is_configurable():
    assert _resolver(default=7).resolve("ns/a", "decode").priority == 7


# ---------------------------------------------------------------------------- #
# Declare coarse, resolve fine                                                 #
# ---------------------------------------------------------------------------- #


def test_participant_selector_covers_every_pool_under_it():
    resolver = _resolver(pools=[{"selector": "prod/chat", "priority": 900}])
    assert resolver.resolve("prod/chat", "prefill").priority == 900
    assert resolver.resolve("prod/chat", "decode").priority == 900
    assert resolver.resolve("prod/batch", "decode").priority == DEFAULT_POOL_PRIORITY


def test_pool_selector_overrides_participant_selector():
    resolver = _resolver(
        pools=[
            {"selector": "prod/chat", "priority": 900},
            {"selector": "prod/chat/prefill", "priority": 950},
        ]
    )
    assert resolver.resolve("prod/chat", "prefill").priority == 950
    assert resolver.resolve("prod/chat", "decode").priority == 900


def test_specificity_beats_declaration_order():
    # Same policies, reversed in the file: resolution must not change.
    forward = _resolver(
        pools=[
            {"selector": "prod/*", "priority": 200},
            {"selector": "prod/chat", "priority": 900},
        ]
    )
    reverse = _resolver(
        pools=[
            {"selector": "prod/chat", "priority": 900},
            {"selector": "prod/*", "priority": 200},
        ]
    )
    for resolver in (forward, reverse):
        assert resolver.resolve("prod/chat", "decode").priority == 900
        assert resolver.resolve("prod/other", "decode").priority == 200


def test_pool_glob_is_more_specific_than_participant_glob():
    resolver = _resolver(
        pools=[
            {"selector": "dev/*", "priority": 300},
            {"selector": "dev/*/prefill", "priority": 90},
        ]
    )
    assert resolver.resolve("dev/x", "prefill").priority == 90
    assert resolver.resolve("dev/x", "decode").priority == 300


def test_single_star_does_not_span_a_slash():
    # The trailing segment forces '*' to be the only thing between 'a' and
    # 'prefill'; a whole-string glob would also match the deeper path.
    resolver = _resolver(pools=[{"selector": "a/*/prefill", "priority": 42}])
    assert resolver.resolve("a/b", "prefill").priority == 42
    assert resolver.resolve("a/b/c", "prefill").priority == DEFAULT_POOL_PRIORITY


def test_double_star_spans_any_number_of_segments():
    resolver = _resolver(pools=[{"selector": "a/**/prefill", "priority": 42}])
    assert resolver.resolve("a/b", "prefill").priority == 42
    assert resolver.resolve("a/b/c/d", "prefill").priority == 42
    assert resolver.resolve("a/b/c/d", "decode").priority == DEFAULT_POOL_PRIORITY


def test_selectors_may_be_any_depth():
    # Participant ids are '<k8s_ns>/<dgd>' today, but the matcher must not
    # hard-code that shape -- deeper hierarchies are on the roadmap.
    resolver = _resolver(
        pools=[
            {
                "selector": "global-pool/east-coast-regions/*/long-context/*",
                "priority": 950,
            }
        ]
    )
    deep = "global-pool/east-coast-regions/us-east-1/long-context"
    assert resolver.resolve(deep, "decode").priority == 950
    assert resolver.resolve(deep, "prefill").priority == 950
    other = "global-pool/east-coast-regions/us-east-1/short-context"
    assert resolver.resolve(other, "decode").priority == DEFAULT_POOL_PRIORITY


def test_bare_double_star_matches_everything_and_ranks_last():
    resolver = _resolver(
        pools=[
            {"selector": "**", "priority": 1},
            {"selector": "prod/chat", "priority": 900},
        ]
    )
    assert resolver.resolve("anything/at-all", "decode").priority == 1
    assert resolver.resolve("prod/chat", "decode").priority == 900


def test_resolution_reports_provenance():
    resolver = _resolver(pools=[{"selector": "prod/chat", "priority": 300}])
    resolved = resolver.resolve("prod/chat", "prefill")
    assert (resolved.selector, resolved.rule_index) == ("prod/chat", 0)


# ---------------------------------------------------------------------------- #
# Static priorities are degenerate conditionals                                #
# ---------------------------------------------------------------------------- #


def test_shorthand_normalizes_into_a_single_unconditional_rule():
    policy = PriorityConfig(pools=[{"selector": "ns/a", "priority": 4}]).pools[0]
    assert policy.priority is None
    assert len(policy.rules) == 1
    assert policy.rules[0].when is None
    assert policy.rules[0].priority == 4


def test_config_survives_a_model_dump_round_trip():
    # resolve_config() merges CLI flags via model_dump() -> model_validate();
    # a shorthand left alongside 'rules' would fail revalidation there.
    config = PriorityConfig(default=9, pools=[{"selector": "ns/a", "priority": 1}])
    assert PriorityConfig.model_validate(config.model_dump()) == config


def test_explicit_rules_are_accepted():
    resolver = _resolver(
        pools=[
            {
                "selector": "ns/a",
                "rules": [{"when": {}, "priority": 2}, {"priority": 8}],
            }
        ]
    )
    resolved = resolver.resolve("ns/a", "decode", PriorityContext())
    assert (resolved.priority, resolved.rule_index) == (2, 0)


def test_unknown_condition_fields_are_rejected():
    # Guards the dangerous alternative: parsing a typo'd predicate into a
    # condition that silently matches everything.
    with pytest.raises(ValidationError):
        PriorityConfig(
            pools=[
                {
                    "selector": "ns/a",
                    "rules": [
                        {"when": {"predicted_requests_above": 100}, "priority": 0},
                        {"priority": 50},
                    ],
                }
            ]
        )


def test_rules_must_end_unconditional():
    with pytest.raises(ValidationError, match="unconditional rule"):
        PriorityConfig(
            pools=[{"selector": "ns/a", "rules": [{"when": {}, "priority": 0}]}]
        )


# ---------------------------------------------------------------------------- #
# Config validation                                                            #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "entry",
    [
        {"selector": "ns/a"},  # neither priority nor rules
        {"selector": "ns/a", "priority": 1, "rules": [{"priority": 2}]},  # both
        {"selector": "ns/a", "rules": []},  # empty
    ],
)
def test_policy_must_declare_exactly_one_form(entry):
    with pytest.raises(ValidationError):
        PriorityConfig(pools=[entry])


@pytest.mark.parametrize("selector", ["ns//prefill", ""])
def test_selectors_with_empty_segments_are_rejected(selector):
    with pytest.raises(ValidationError):
        PriorityConfig(pools=[{"selector": selector, "priority": 0}])


@pytest.mark.parametrize("selector", ["ns", "ns/a", "ns/a/prefill", "a/b/c/d/e"])
def test_selectors_of_any_depth_are_accepted(selector):
    assert PriorityConfig(pools=[{"selector": selector, "priority": 0}])


def test_duplicate_selectors_are_rejected():
    with pytest.raises(ValidationError, match="duplicate selector"):
        PriorityConfig(
            pools=[
                {"selector": "ns/a", "priority": 900},
                {"selector": "ns/a", "priority": 5},
            ]
        )


def test_unknown_config_field_is_rejected():
    with pytest.raises(ValidationError):
        PriorityConfig(defualt=5)  # codespell:ignore defualt


# ---------------------------------------------------------------------------- #
# Load-conditional priorities                                                  #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "requests, expected",
    [
        (None, False),  # no signal -> never matches
        (49.9, True),
        (50.0, False),  # boundary is exclusive
        (500.0, False),
    ],
)
def test_predicted_requests_below(requests, expected):
    condition = PriorityCondition(predicted_requests_below=50)
    ctx = PriorityContext(predicted_num_requests=requests)
    assert condition.matches(ctx) is expected


@pytest.mark.parametrize(
    "requests, expected",
    [
        (None, False),  # no signal -> never matches
        (99.0, False),
        (100.0, True),  # boundary is inclusive
        (500.0, True),
    ],
)
def test_predicted_requests_at_least(requests, expected):
    condition = PriorityCondition(predicted_requests_at_least=100)
    ctx = PriorityContext(predicted_num_requests=requests)
    assert condition.matches(ctx) is expected


def test_predicates_combine_as_a_band():
    condition = PriorityCondition(
        predicted_requests_at_least=10, predicted_requests_below=100
    )
    assert condition.matches(PriorityContext(predicted_num_requests=50))
    assert not condition.matches(PriorityContext(predicted_num_requests=5))
    assert not condition.matches(PriorityContext(predicted_num_requests=100))


def test_unsatisfiable_band_is_rejected():
    with pytest.raises(ValidationError, match="no request rate can satisfy"):
        PriorityCondition(predicted_requests_at_least=100, predicted_requests_below=10)


def test_conditional_policy_selects_the_matching_rule():
    resolver = _resolver(
        pools=[
            {
                "selector": "prod/batch",
                "rules": [
                    {"when": {"predicted_requests_below": 50}, "priority": 800},
                    {"priority": 100},
                ],
            }
        ]
    )
    quiet = resolver.resolve(
        "prod/batch", "decode", PriorityContext(predicted_num_requests=5)
    )
    assert (quiet.priority, quiet.rule_index) == (800, 0)

    busy = resolver.resolve(
        "prod/batch", "decode", PriorityContext(predicted_num_requests=500)
    )
    assert (busy.priority, busy.rule_index) == (100, 1)

    # No signal at all falls through to the unconditional rule.
    unknown = resolver.resolve("prod/batch", "decode", PriorityContext())
    assert (unknown.priority, unknown.rule_index) == (100, 1)


# ---------------------------------------------------------------------------- #
# Building a context from the wire                                             #
# ---------------------------------------------------------------------------- #


def test_context_from_predicted_load():
    ctx = PriorityContext.from_predicted_load(
        {"num_requests": 12, "isl": 2048.0, "osl": 256}
    )
    assert ctx.predicted_num_requests == 12.0
    assert (ctx.predicted_isl, ctx.predicted_osl) == (2048.0, 256.0)


@pytest.mark.parametrize("payload", [None, {}, {"num_requests": None}])
def test_absent_predicted_load_yields_an_empty_context(payload):
    assert PriorityContext.from_predicted_load(payload) == PriorityContext()


@pytest.mark.parametrize("value", ["100", True, False, [1], {"a": 1}])
def test_malformed_predicted_load_degrades_to_no_signal(value):
    # Caller-supplied payload: a bad prediction must not fail a scale request,
    # and bool must not sneak through as an int.
    ctx = PriorityContext.from_predicted_load({"num_requests": value})
    assert ctx.predicted_num_requests is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_unusable_numeric_rates_degrade_to_no_signal(value):
    # NaN is the dangerous one: every comparison against it is False, so
    # without an explicit guard it slips past both predicate checks and
    # satisfies any condition instead of none of them.
    assert (
        PriorityContext.from_predicted_load(
            {"num_requests": value}
        ).predicted_num_requests
        is None
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rate_matches_nothing(value):
    # PriorityContext is a plain dataclass, so guard at the decision point too
    # rather than trusting only the from_predicted_load boundary.
    ctx = PriorityContext(predicted_num_requests=value)
    assert not PriorityCondition(predicted_requests_below=50).matches(ctx)
    assert not PriorityCondition(predicted_requests_at_least=10).matches(ctx)
