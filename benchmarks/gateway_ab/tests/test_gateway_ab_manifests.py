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
    b = _components(m.build_dgd(cfg, topology, "gateway"))
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
def test_router_tiers_use_kv_routing_and_same_image(topology):
    cfg = m.BenchConfig()
    fe = _main(_components(m.build_dgd(cfg, topology, "frontend"))["Frontend"])
    epp_component = _components(m.build_dgd(cfg, topology, "gateway"))["Epp"]
    epp = _main(epp_component)
    assert epp_component["type"] == "epp"
    assert {"name": "DYN_ROUTER_MODE", "value": "kv"} in fe["env"]
    assert fe["image"] == epp["image"] == cfg.frontend_image
    assert fe["resources"] == epp["resources"]


def test_block_size_reaches_the_mocker_and_the_epp_reads_the_card():
    cfg = m.BenchConfig(block_size=64)
    comps = _components(m.build_dgd(cfg, "disagg", "gateway"))
    # Only the standalone EPP reads DYN_KV_CACHE_BLOCK_SIZE; in Dynamo mode it
    # takes the block size from the workers' model cards, as the Frontend does.
    assert "env" not in _main(comps["Epp"])
    for name in ("prefill", "decode"):
        args = _main(comps[name])["args"]
        assert args[args.index("--block-size") + 1] == "64"


def test_every_object_carries_its_run_prefix():
    cfg = m.BenchConfig(name_prefix="u1")
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
    assert _components(m.build_dgd(cfg, "agg", "gateway"))["decode"]["replicas"] == 3
    comps = _components(m.build_dgd(cfg, "disagg", "gateway"))
    assert comps["prefill"]["replicas"] == 1
    assert comps["decode"]["replicas"] == 4
    with pytest.raises(ValueError):
        m.BenchConfig(agg_workers=0)


def test_route_targets_pool_and_header():
    cfg = m.BenchConfig(name_prefix="x")
    route = m.build_httproute(cfg, "disagg")
    rule = route["spec"]["rules"][0]
    assert rule["backendRefs"][0]["name"] == "x-disagg-gw-pool"
    header = rule["matches"][0]["headers"][0]
    assert header["name"] == m.TARGET_HEADER
    assert header["value"] == m.request_headers(cfg, "disagg")[m.TARGET_HEADER]
    assert route["spec"]["parentRefs"][0]["namespace"] == cfg.gateway_namespace


def test_images_follow_version():
    cfg = m.BenchConfig(dynamo_version="9.9.9")
    assert cfg.frontend_image.endswith("dynamo-frontend:9.9.9")
    assert cfg.worker_image.endswith("dynamo-planner:9.9.9")
    override = m.BenchConfig(frontend_image="reg/fe:pr", worker_image="reg/w:pr")
    assert override.frontend_image == "reg/fe:pr"


def test_secrets_reach_every_container():
    cfg = m.BenchConfig(hf_secret="hf", image_pull_secret="pull")
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
    for obj in m.build_all(m.BenchConfig(), m.TOPOLOGIES):
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
