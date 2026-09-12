# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.site`.

The capability table is asserted in **both directions**: every capability a site
has is exercised, and every capability it lacks has a non-empty reason. A table
tested only one way drifts into claiming things it cannot do, which is the exact
failure this module exists to prevent.
"""

import pytest
from dynamo_test.site import (
    BUILTIN_SITES,
    LOCAL,
    Capability,
    Ownership,
    Site,
    Topology,
    UnknownSite,
    capabilities,
    resolve,
    why,
)

# The four arrangements, as they would actually be declared.
ALL_IN_ONE = LOCAL

SIDECAR = Site(
    name="sidecar",
    topology=Topology.SIDECAR,
    ownership=Ownership.IMPOSE,
    shares_process_tree=False,
    shares_network=True,  # same compose network; the SUT can dial back
    substrate_owned=True,
    provider="compose",
)

COMPOSE = Site(
    name="compose",
    topology=Topology.COMPOSE,
    ownership=Ownership.IMPOSE,
    shares_process_tree=False,
    shares_network=True,
    substrate_owned=True,
    provider="compose",
)

K8S_SHARED = Site(
    name="k3s-shared",
    topology=Topology.KUBERNETES,
    ownership=Ownership.IMPOSE,
    shares_process_tree=False,
    shares_network=False,  # the cluster cannot dial a laptop's ephemeral port
    substrate_owned=False,  # a shared namespace
    provider="k8s",
)

ATTACHED = Site(
    name="ci-all-in-one-attached",
    topology=Topology.ALL_IN_ONE,
    ownership=Ownership.VERIFY,
    shares_process_tree=True,
    shares_network=True,
    substrate_owned=False,
    provider="local",
)

ALL = [ALL_IN_ONE, SIDECAR, COMPOSE, K8S_SHARED, ATTACHED]


# ------------------------------------------------------- the table, both ways


@pytest.mark.parametrize("site", ALL, ids=lambda s: s.name)
def test_every_absent_capability_has_a_reason(site):
    """A missing capability must always say which property removed it.

    "Not supported here" teaches nothing and gets copy-pasted into the next
    skip. Naming the rung tells the reader whether to fix the site, choose a
    different one, or accept the test belongs to one tier.
    """
    have = capabilities(site)
    for cap in Capability:
        if cap in have:
            assert (
                why(site, cap) is None
            ), f"{cap} is available but why() explains it away"
        else:
            reason = why(site, cap)
            assert reason, f"{site.name} lacks {cap} with no reason given"
            assert (
                len(reason) > 20
            ), f"reason for {cap} is too terse to act on: {reason}"


@pytest.mark.parametrize("site", ALL, ids=lambda s: s.name)
def test_capabilities_are_a_subset_of_the_ceiling(site):
    """Computed, never declared: a site cannot end up with more than its
    arrangement allows."""
    assert capabilities(site) <= frozenset(Capability)


# ---------------------------------------------------------- the rungs, singly


def test_leaving_the_process_tree_removes_signalling_and_log_files():
    """The single largest difference between T1 and everything above it."""
    assert Capability.SIGNAL_INNER_PROCESS in capabilities(ALL_IN_ONE)
    assert Capability.READ_LOG_FILE in capabilities(ALL_IN_ONE)

    assert Capability.SIGNAL_INNER_PROCESS not in capabilities(COMPOSE)
    assert Capability.READ_LOG_FILE not in capabilities(COMPOSE)
    assert "another process namespace" in why(COMPOSE, Capability.SIGNAL_INNER_PROCESS)


def test_losing_the_reverse_network_removes_the_sink():
    """The dangerous one: a canary that is never dialled looks like a canary
    that correctly refused."""
    assert Capability.SINK in capabilities(COMPOSE)
    assert Capability.SINK not in capabilities(K8S_SHARED)
    assert "never arrive" in why(K8S_SHARED, Capability.SINK)


def test_verify_may_observe_but_not_mutate():
    """Attaching to someone else's deployment must not permit restarting it."""
    have = capabilities(ATTACHED)
    assert Capability.RESTART_ROLE not in have
    assert Capability.STOP_ROLE not in have
    assert Capability.SCALE_REPLICAS not in have
    # ...but it keeps everything observational, which is the point of attaching.
    assert Capability.READ_LOG_FILE in have
    assert Capability.SIGNAL_INNER_PROCESS in have
    assert "does not own" in why(ATTACHED, Capability.RESTART_ROLE)


def test_a_shared_substrate_forbids_scrubbing():
    assert Capability.SCRUB in capabilities(COMPOSE)
    assert Capability.SCRUB not in capabilities(K8S_SHARED)
    assert "other runs" in why(K8S_SHARED, Capability.SCRUB)


def test_the_first_rung_is_the_one_reported():
    """Later rungs would remove it anyway; the actionable one is the first."""
    doubly = Site(
        name="doubly-blocked",
        topology=Topology.KUBERNETES,
        ownership=Ownership.VERIFY,
        shares_process_tree=False,
        shares_network=False,
        substrate_owned=False,
        provider="k8s",
    )
    # RESTART_ROLE is removed by VERIFY only, so that is what should be reported.
    assert "does not own" in why(doubly, Capability.RESTART_ROLE)


# --------------------------------------------------------- narrowing, not widening


def test_a_site_may_narrow():
    narrowed = Site(
        name="no-exec",
        topology=Topology.COMPOSE,
        ownership=Ownership.IMPOSE,
        shares_process_tree=False,
        shares_network=True,
        substrate_owned=True,
        narrow=frozenset({Capability.EXEC_IN}),
    )
    assert Capability.EXEC_IN not in capabilities(narrowed)
    assert "narrows" in why(narrowed, Capability.EXEC_IN)


def test_there_is_no_way_to_widen():
    """The absence of a `widen` field is the design, so assert it stays absent.

    A site that could claim a capability it lacks would produce the vacuous pass
    this whole module exists to prevent.
    """
    assert not hasattr(Site, "widen")
    assert "widen" not in {f for f in Site.__dataclass_fields__}


def test_narrowing_an_unknown_capability_is_rejected():
    with pytest.raises(ValueError, match="unknown capabilities"):
        Site(
            name="typo",
            topology=Topology.COMPOSE,
            ownership=Ownership.IMPOSE,
            shares_process_tree=False,
            shares_network=True,
            substrate_owned=True,
            narrow=frozenset({"restart_rol"}),  # type: ignore[arg-type]
        )


def test_the_provider_is_physics_not_policy():
    """A provider that has not implemented a verb cannot perform it, whatever
    the arrangement permits."""
    partial = frozenset({Capability.EXEC_IN})
    assert capabilities(COMPOSE, partial) == partial
    assert "does not implement" in why(COMPOSE, Capability.RESTART_ROLE, partial)


# ------------------------------------------------------------------ validation


def test_only_all_in_one_may_claim_a_shared_process_tree():
    """Believing otherwise makes every signal-based fault test vacuous."""
    with pytest.raises(ValueError, match="shared process tree"):
        Site(
            name="wrong",
            topology=Topology.KUBERNETES,
            ownership=Ownership.IMPOSE,
            shares_process_tree=True,
            shares_network=False,
            substrate_owned=True,
        )


def test_a_site_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        Site(
            name="",
            topology=Topology.COMPOSE,
            ownership=Ownership.IMPOSE,
            shares_process_tree=False,
            shares_network=True,
            substrate_owned=True,
        )


# --------------------------------------------------------------- resolution


def test_the_default_is_todays_behaviour():
    """Running the suite as it runs now must require no site definition."""
    assert resolve(None) is LOCAL
    assert LOCAL.topology is Topology.ALL_IN_ONE
    assert LOCAL.ownership is Ownership.IMPOSE
    assert LOCAL.shares_process_tree and LOCAL.shares_network


def test_an_unknown_site_names_the_known_ones():
    with pytest.raises(UnknownSite, match="local"):
        resolve("nosuch")


def test_extra_sites_resolve():
    assert resolve("compose", {"compose": COMPOSE}) is COMPOSE
    assert set(BUILTIN_SITES) == {"local"}


# ------------------------------------------------------------------- record


def test_the_site_serialises_with_its_computed_capabilities():
    """The run record must carry what the site *could* do, not just what it
    declared, so a later reader can tell a skip from a gap."""
    import json

    record = K8S_SHARED.to_record()
    assert record["topology"] == "kubernetes"
    assert record["ownership"] == "impose"
    assert "sink" not in record["capabilities"]
    assert "scrub" not in record["capabilities"]
    assert json.loads(json.dumps(record))


def test_topology_is_a_label_nothing_branches_on():
    """Guards the design decision: behaviour keys off the three properties.

    If a capability decision ever keys off the topology *name*, adding a fifth
    arrangement means auditing every branch. The ceiling table is the one
    permitted use.
    """
    import inspect

    from dynamo_test import site as site_module

    source = inspect.getsource(site_module.capabilities)
    assert "Topology." not in source, (
        "capabilities() branches on a topology name; it should read the "
        "physical properties instead"
    )


# --------------------------------------------- refusal, and why it must raise


def test_refusal_is_raised_never_returned():
    """The single most important line in the ladder.

    A `stop()` that quietly no-ops because the site does not own the deployment
    turns a fault-tolerance test into a test of nothing: the fault is never
    injected, the system stays healthy, and the assertion that it recovered
    passes. A raise is noisy and correct; a no-op is quiet and wrong.
    """
    from dynamo_test.site import Refused

    assert issubclass(Refused, Exception)
    with pytest.raises(Refused) as exc:
        raise Refused("stop", ATTACHED, why(ATTACHED, Capability.STOP_ROLE))
    message = str(exc.value)
    assert "stop()" in message
    assert ATTACHED.name in message
    assert "does not own" in message  # the rung, not just "unavailable"


def test_a_verify_site_cannot_reach_a_mutating_verb_at_all():
    """Belt and braces: the capability is absent *and* the refusal explains it."""
    for cap in (
        Capability.STOP_ROLE,
        Capability.RESTART_ROLE,
        Capability.SCALE_REPLICAS,
    ):
        assert cap not in capabilities(ATTACHED)
        assert why(ATTACHED, cap)


def test_divergence_records_unverified_rather_than_guessing():
    """`declared=KNOWN, observed=UNKNOWN` is neither agreement nor divergence.

    Scoring it either way invents a result. The comparison has to be able to say
    "I could not check this".
    """
    from dynamo_test.facts import Fact
    from dynamo_test.site import Divergence

    could_not_look = Divergence(
        role="frontend",
        field="model",
        declared=Fact.known("Qwen/Qwen3-0.6B", "stack"),
        observed=Fact.unknown("/v1/models", "connection refused"),
    )
    assert could_not_look.is_unverified
    assert "could not observe" in could_not_look.describe()
    assert "connection refused" in could_not_look.describe()

    real = Divergence(
        role="frontend",
        field="model",
        declared=Fact.known("Qwen/Qwen3-0.6B", "stack"),
        observed=Fact.known("meta-llama/Llama-3.1-8B", "/v1/models"),
    )
    assert not real.is_unverified
    assert "Llama" in real.describe()


def test_mismatch_names_every_divergence_and_counts_the_unverified():
    """Raised at bind time. The alternative is that attaching to a deployment
    serving a different model first shows up as a 404 at query time, which reads
    like a routing bug rather than "you are looking at the wrong deployment"."""
    from dynamo_test.facts import Fact
    from dynamo_test.site import Divergence, Mismatch

    with pytest.raises(Mismatch) as exc:
        raise Mismatch(
            [
                Divergence(
                    "frontend", "model", Fact.known("A", "s"), Fact.known("B", "o")
                ),
                Divergence(
                    "worker",
                    "replicas",
                    Fact.known(2, "s"),
                    Fact.unknown("o", "no api"),
                ),
            ]
        )
    message = str(exc.value)
    assert "2 divergence(s)" in message
    assert "1 unverified" in message
    assert "frontend.model" in message and "worker.replicas" in message
