# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The A and B deployments must differ only in their routing path."""

import copy

import pytest

from benchmarks.gateway_ab import manifests as m

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


def _components(dgd):
    return {c["name"]: c for c in dgd["spec"]["components"]}


def _main(component):
    return next(
        c for c in component["podTemplate"]["spec"]["containers"] if c["name"] == "main"
    )


@pytest.mark.parametrize("topology", m.TOPOLOGIES)
def test_workers_identical_across_sides(topology):
    cfg = m.BenchConfig()
    a = _components(m.build_dgd(cfg, topology, "frontend"))
    b = _components(m.build_dgd(cfg, topology, "epp"))
    worker_names = {"decode"} if topology == "agg" else {"prefill", "decode"}
    assert set(a) == {"Frontend"} | worker_names
    assert set(b) == {"Epp"} | worker_names
    for name in worker_names:
        wa, wb = copy.deepcopy(a[name]), copy.deepcopy(b[name])
        # The sidecar is the B side's routing path; everything else must match.
        assert wb.pop("frontendSidecar") == m.SIDECAR_NAME
        sidecars = [
            c
            for c in wb["podTemplate"]["spec"]["containers"]
            if c["name"] == m.SIDECAR_NAME
        ]
        assert len(sidecars) == 1
        assert sidecars[0]["args"][-2:] == ["--router-mode", "direct"]
        wb["podTemplate"]["spec"]["containers"].remove(sidecars[0])
        assert wa == wb


@pytest.mark.parametrize("topology", m.TOPOLOGIES)
def test_direct_arm_is_a_copy_of_a(topology):
    cfg = m.BenchConfig(with_direct=True)
    assert m.arms(cfg) == ("direct", "frontend", "epp")
    assert m.arms(m.BenchConfig()) == ("frontend", "epp")
    a = m.build_dgd(cfg, topology, "frontend")
    direct = m.build_dgd(cfg, topology, "direct")
    # A copy, not A itself: replaying A's prompts would hit A's caches.
    assert direct["metadata"]["name"] != a["metadata"]["name"]
    assert direct["spec"] == a["spec"]
    names = [o["metadata"]["name"] for o in m.build_all(cfg, (topology,))]
    assert m.dgd_name(cfg, topology, "direct") in names
    # Only A and B go through the Gateway.
    with pytest.raises(ValueError):
        m.build_httproute(cfg, topology, "direct")


def test_both_routers_run_the_same_kv_router_and_read_the_block_size_from_cards():
    cfg = m.BenchConfig(block_size=64)
    a = _components(m.build_dgd(cfg, "disagg", "frontend"))
    b = _components(m.build_dgd(cfg, "disagg", "epp"))
    fe, epp = _main(a["Frontend"]), _main(b["Epp"])
    assert b["Epp"]["type"] == "epp"
    assert {"name": "DYN_ROUTER_MODE", "value": "kv"} in fe["env"]
    assert fe["image"] == epp["image"] == cfg.frontend_image
    assert fe["resources"] == epp["resources"]
    # The EPP takes the block size from the workers' model cards, as the
    # Frontend does, so neither router is told it any other way.
    assert "env" not in epp
    for name in ("prefill", "decode"):
        args = _main(b[name])["args"]
        assert args[args.index("--block-size") + 1] == "64"


def test_routes_differ_only_in_backend_and_target():
    cfg = m.BenchConfig(name_prefix="x", namespace="ns")
    a = m.build_httproute(cfg, "disagg", "frontend")
    b = m.build_httproute(cfg, "disagg", "epp")
    assert b["spec"]["rules"][0]["backendRefs"] == [
        {
            "group": "inference.networking.k8s.io",
            "kind": "InferencePool",
            "name": "x-disagg-epp-pool",
            "port": 8000,
        }
    ]
    assert a["spec"]["rules"][0]["backendRefs"] == [
        {"group": "", "kind": "Service", "name": "x-disagg-fe-frontend", "port": 8000}
    ]
    for route, side in ((a, "frontend"), (b, "epp")):
        (match,) = route["spec"]["rules"][0]["matches"]
        # The exact path and every header the arm's requests carry, so a
        # broader route on a shared Gateway cannot outrank it.
        assert match["path"] == {"type": "Exact", "value": m.CHAT_PATH}
        headers = {h["name"]: h["value"] for h in match["headers"]}
        assert headers == m.request_headers(cfg, "disagg", side)
        assert route["spec"]["parentRefs"][0]["namespace"] == cfg.gateway_namespace
    # The target names the namespace: a run with the same prefix in another
    # namespace selects other routes.
    assert m.request_headers(cfg, "disagg", "epp")[m.TARGET_HEADER] == "ns/x-disagg-epp"
    for route in (a, b):
        route["metadata"].pop("name")
        rule = route["spec"]["rules"][0]
        rule.pop("backendRefs")
        for header in rule["matches"][0]["headers"]:
            if header["name"] == m.TARGET_HEADER:
                header.pop("value")
    assert a == b


def test_every_object_carries_its_run_prefix():
    cfg = m.BenchConfig(name_prefix="u1", with_direct=True)
    for obj in m.build_all(cfg, m.TOPOLOGIES):
        assert obj["metadata"]["labels"][m.INSTANCE] == "u1", obj["kind"]
    # A loadgen from another run, or one built before the labels, is replaced.
    other = m.build_loadgen_pod(m.BenchConfig(name_prefix="u2"))
    mine = m.build_loadgen_pod(cfg)
    assert (
        other["metadata"]["annotations"][m.LOADGEN_SPEC_HASH]
        != mine["metadata"]["annotations"][m.LOADGEN_SPEC_HASH]
    )


def test_secrets_reach_every_container():
    cfg = m.BenchConfig(hf_secret="hf", image_pull_secret="pull", with_direct=True)
    for obj in m.build_all(cfg, m.TOPOLOGIES):
        if obj["kind"] == "HTTPRoute":
            continue
        specs = (
            [obj["spec"]]
            if obj["kind"] == "Pod"
            else [c["podTemplate"]["spec"] for c in obj["spec"]["components"]]
        )
        for spec in specs:
            assert spec["imagePullSecrets"] == [{"name": "pull"}]
            for c in spec["containers"]:
                assert c["envFrom"] == [{"secretRef": {"name": "hf"}}]
