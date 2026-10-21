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
    assert direct["metadata"]["labels"]["gateway-ab/side"] == "direct"
    names = [o["metadata"]["name"] for o in m.build_all(cfg, (topology,))]
    assert m.dgd_name(cfg, topology, "direct") in names
    # Only A and B go through the Gateway.
    with pytest.raises(ValueError):
        m.build_httproute(cfg, topology, "direct")


@pytest.mark.parametrize("topology", m.TOPOLOGIES)
def test_router_tiers_use_kv_routing_and_same_image(topology):
    cfg = m.BenchConfig()
    fe = _main(_components(m.build_dgd(cfg, topology, "frontend"))["Frontend"])
    epp_component = _components(m.build_dgd(cfg, topology, "epp"))["Epp"]
    epp = _main(epp_component)
    assert epp_component["type"] == "epp"
    assert {"name": "DYN_ROUTER_MODE", "value": "kv"} in fe["env"]
    assert fe["image"] == epp["image"] == cfg.frontend_image
    assert fe["resources"] == epp["resources"]


def test_block_size_reaches_the_mocker_and_the_epp_reads_the_card():
    cfg = m.BenchConfig(block_size=64)
    comps = _components(m.build_dgd(cfg, "disagg", "epp"))
    # Only the standalone EPP reads DYN_KV_CACHE_BLOCK_SIZE; in Dynamo mode it
    # takes the block size from the workers' model cards, as the Frontend does.
    assert "env" not in _main(comps["Epp"])
    for name in ("prefill", "decode"):
        args = _main(comps[name])["args"]
        assert args[args.index("--block-size") + 1] == "64"


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


def test_disagg_modes():
    comps = _components(m.build_dgd(m.BenchConfig(), "disagg", "frontend"))
    for name in ("prefill", "decode"):
        args = _main(comps[name])["args"]
        assert args[args.index("--disaggregation-mode") + 1] == name
    agg = _components(m.build_dgd(m.BenchConfig(), "agg", "frontend"))
    assert "--disaggregation-mode" not in _main(agg["decode"])["args"]


def test_worker_counts():
    cfg = m.BenchConfig(agg_workers=3, prefill_workers=1, decode_workers=4)
    assert _components(m.build_dgd(cfg, "agg", "epp"))["decode"]["replicas"] == 3
    comps = _components(m.build_dgd(cfg, "disagg", "epp"))
    assert comps["prefill"]["replicas"] == 1
    assert comps["decode"]["replicas"] == 4
    with pytest.raises(ValueError):
        m.BenchConfig(agg_workers=0)


def test_routes_share_the_gateway_and_differ_in_backend_and_header():
    cfg = m.BenchConfig(name_prefix="x")
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
        header = route["spec"]["rules"][0]["matches"][0]["headers"][0]
        assert header["name"] == m.TARGET_HEADER
        assert (
            header["value"] == m.request_headers(cfg, "disagg", side)[m.TARGET_HEADER]
        )
        assert route["spec"]["parentRefs"][0]["namespace"] == cfg.gateway_namespace
    # Apart from the backend and the header value, the two routes are the same.
    for route in (a, b):
        route["metadata"].pop("name")
        rule = route["spec"]["rules"][0]
        rule.pop("backendRefs")
        rule["matches"][0]["headers"][0].pop("value")
    assert a == b


def test_images_follow_version():
    cfg = m.BenchConfig(dynamo_version="9.9.9")
    assert cfg.frontend_image.endswith("dynamo-frontend:9.9.9")
    assert cfg.worker_image.endswith("dynamo-planner:9.9.9")
    override = m.BenchConfig(frontend_image="reg/fe:pr", worker_image="reg/w:pr")
    assert override.frontend_image == "reg/fe:pr"


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


def test_no_cpu_limits():
    for obj in m.build_all(m.BenchConfig(with_direct=True), m.TOPOLOGIES):
        if obj["kind"] == "HTTPRoute":
            continue
        specs = (
            [obj["spec"]]
            if obj["kind"] == "Pod"
            else [c["podTemplate"]["spec"] for c in obj["spec"]["components"]]
        )
        for spec in specs:
            for c in spec["containers"]:
                assert "limits" not in c.get("resources", {})


def test_unknown_topology_or_side():
    with pytest.raises(ValueError):
        m.dgd_name(m.BenchConfig(), "hybrid", "frontend")
    with pytest.raises(ValueError):
        m.build_dgd(m.BenchConfig(), "agg", "router")
