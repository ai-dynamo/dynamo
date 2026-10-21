# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Docker backend: same workers as Kubernetes, Envoy as GAIE programs it."""

import json
import subprocess

import pytest

from benchmarks.gateway_ab import cli, local_backend, manifests

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


class _RecordingDocker(local_backend.Docker):
    """Records every docker invocation; answers the few that return data."""

    def __init__(self, fail=()):
        self.calls = []
        # Docker subcommands that fail, as (args prefix, stderr).
        self.fail = list(fail)

    def run(self, args, *, input=None, check=True, timeout=600):
        self.calls.append(list(args))
        for prefix, stderr in self.fail:
            if args[: len(prefix)] == prefix:
                return subprocess.CompletedProcess(args, 1, "", stderr)
        out = ""
        if args[:2] == ["inspect", "--format"] and "IPAddress" in args[2]:
            out = "172.18.0.9\n"
        elif args[:2] == ["run", "--rm"]:
            # The model-card count: report every worker registered.
            out = json.dumps(
                {
                    f"ab-{topology}-{tag}": 2 if topology == "agg" else 4
                    for topology in manifests.TOPOLOGIES
                    for tag in manifests.TAGS.values()
                }
            )
        elif args[:1] == ["ps"]:
            out = ""
        return subprocess.CompletedProcess(args, 0, out, "")


def _backend(*extra, docker=None):
    args = cli.build_parser().parse_args(["all", "--backend", "local", *extra])
    bench = cli.Bench(args)
    docker = docker or _RecordingDocker()
    bench.backend.docker = docker
    return bench.backend, docker


def _runs(docker):
    """`docker run -d` calls, keyed by container name."""
    out = {}
    for call in docker.calls:
        if call[:2] == ["run", "-d"]:
            out[call[call.index("--name") + 1]] = call
    return out


def _env(call):
    return dict(call[i + 1].split("=", 1) for i, a in enumerate(call) if a == "-e")


def _command(call, image):
    return call[call.index(image) + 1 :]


@pytest.mark.parametrize("topology", manifests.TOPOLOGIES)
def test_every_arm_runs_identical_workers(topology):
    backend, docker = _backend("--with-direct")
    backend.deploy_topology(topology, "fp")
    runs = _runs(docker)
    cfg = backend.cfg
    for ctype, replicas, mode in manifests.worker_roles(cfg, topology):
        for i in range(replicas):
            a = runs[f"gwab-ab-{topology}-fe-{ctype}{i}"]
            assert _command(a, cfg.worker_image) == [
                "-m",
                "dynamo.mocker",
                *manifests.mocker_args(cfg, mode),
            ]
            for tag in manifests.TAGS.values():
                other = runs[f"gwab-ab-{topology}-{tag}-{ctype}{i}"]
                assert _command(other, cfg.worker_image) == _command(
                    a, cfg.worker_image
                )
                # Each arm in its own Dynamo namespace, so none shares a cache.
                assert _env(other)["DYN_NAMESPACE"] == f"ab-{topology}-{tag}"
    # The direct arm's Frontend is A's, in the direct arm's namespace.
    a_fe = runs[f"gwab-ab-{topology}-frontend"]
    direct_fe = runs[f"gwab-ab-{topology}-dir-frontend"]
    assert _command(a_fe, cfg.frontend_image) == _command(direct_fe, cfg.frontend_image)
    assert _env(direct_fe) == dict(_env(a_fe), DYN_NAMESPACE=f"ab-{topology}-dir")


def test_direct_arm_is_deployed_only_when_asked():
    backend, docker = _backend()
    backend.deploy_topology("agg", "fp")
    assert not any("-dir-" in name for name in _runs(docker))


def test_epp_points_at_the_direct_frontend_without_kubernetes():
    backend, docker = _backend("--epp-binary", "/tmp/epp")
    backend.deploy_topology("agg", "fp")
    runs = _runs(docker)
    epp = runs["gwab-ab-agg-epp"]
    env = _env(epp)
    assert env["DYN_EPP_STATIC_ENDPOINT"] == "172.18.0.9:8000"
    assert env["DYN_SECURE_SERVING"] == "false"
    assert env["DYN_NAMESPACE"] == "ab-agg-epp"
    # A Dynamo-mode EPP reads the model and block size from the model cards.
    assert "DYN_KV_CACHE_BLOCK_SIZE" not in env and "DYN_MODEL_NAME" not in env
    assert "/tmp/epp:/epp-dev:ro" in epp and "/epp-dev" in epp
    direct = runs["gwab-ab-agg-epp-frontend"]
    assert _command(direct, backend.cfg.frontend_image)[-2:] == [
        "--router-mode",
        "direct",
    ]
    assert _env(runs["gwab-ab-agg-frontend"])["DYN_ROUTER_MODE"] == "kv"
    for name, call in runs.items():
        assert f"{local_backend.FINGERPRINT_LABEL}=fp" in call, name


def test_deploy_starts_from_a_clean_slate():
    backend, docker = _backend()
    backend.deploy_topology("disagg", "fp")
    first_create = next(
        i for i, c in enumerate(docker.calls) if c[:2] == ["network", "create"]
    )
    assert ["volume", "rm", "-f", "gwab-ab-disagg-kv"] in docker.calls[:first_create]
    assert [
        "network",
        "connect",
        "gwab-ab-disagg",
        local_backend.loadgen_name(backend.cfg),
    ] in docker.calls


def _address(cluster):
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    return endpoint["endpoint"]["address"]["socket_address"]["address"]


def test_envoy_config_matches_what_the_epp_requires():
    cfg = manifests.BenchConfig()
    config = json.loads(local_backend.envoy_config(cfg, "agg"))
    listener = config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    ext_proc = hcm["http_filters"][0]["typed_config"]
    mode = ext_proc["processing_mode"]
    assert (
        mode["request_body_mode"]
        == mode["response_body_mode"]
        == "FULL_DUPLEX_STREAMED"
    )
    assert ext_proc["send_body_without_waiting_for_header_response"] is True
    clusters = {c["name"]: c for c in config["static_resources"]["clusters"]}
    picked = clusters["picked_endpoint"]
    assert picked["type"] == "ORIGINAL_DST"
    assert (
        picked["original_dst_lb_config"]["http_header_name"]
        == "x-gateway-destination-endpoint"
    )
    assert _address(clusters["epp"]) == "gwab-ab-agg-epp"
    assert _address(clusters["frontend"]) == "gwab-ab-agg-frontend"


def test_envoy_routes_a_past_the_epp_and_b_through_it():
    cfg = manifests.BenchConfig()
    config = json.loads(local_backend.envoy_config(cfg, "disagg"))
    listener = config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    routes = {}
    for route in hcm["route_config"]["virtual_hosts"][0]["routes"]:
        (header,) = route["match"]["headers"]
        assert header["name"] == manifests.TARGET_HEADER
        routes[header["string_match"]["exact"]] = route
    # The same header values select the HTTPRoutes on Kubernetes.
    a, b = (
        routes[manifests.request_headers(cfg, "disagg", side)[manifests.TARGET_HEADER]]
        for side in manifests.SIDES
    )
    assert len(routes) == 2
    assert a["route"]["cluster"] == "frontend"
    assert a["typed_per_filter_config"][local_backend.EXT_PROC]["disabled"] is True
    assert b["route"]["cluster"] == "picked_endpoint"
    assert "typed_per_filter_config" not in b
    assert a["route"]["timeout"] == b["route"]["timeout"]


def test_urls_use_container_names():
    backend, _ = _backend()
    gateway = "http://gwab-ab-disagg-gateway:8080"
    # A and B share the gateway; the target header picks the route.
    assert backend.url("disagg", "frontend") == gateway
    assert backend.url("disagg", "epp") == gateway
    assert backend.url("agg", "direct") == "http://gwab-ab-agg-dir-frontend:8000"


def test_waits_until_every_worker_registered(monkeypatch):
    backend, docker = _backend("--ready-timeout", "6")
    monkeypatch.setattr(
        local_backend.retry.__globals__["time"], "sleep", lambda s: None
    )
    replies = [{"ab-agg-fe": 2, "ab-agg-epp": 1}, {"ab-agg-fe": 2, "ab-agg-epp": 2}]

    def run(args, **kwargs):
        if args[:2] == ["run", "--rm"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(replies.pop(0)), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(docker, "run", run)
    backend._wait_for_workers("agg", "gwab-agg-kv")
    assert replies == []


def test_exited_container_fails_fast(monkeypatch):
    backend, docker = _backend()

    def run(args, **kwargs):
        out = (
            "gwab-agg-epp exited\ngwab-agg-gateway running\n" if args[0] == "ps" else ""
        )
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(docker, "run", run)
    with pytest.raises(cli.KubeError, match="gwab-agg-epp"):
        backend._check_running("agg")


def test_exited_container_is_not_waited_out(monkeypatch):
    backend, docker = _backend("--ready-timeout", "3000")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        out = "gwab-agg-epp exited\n" if args[0] == "ps" else json.dumps({})
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(docker, "run", run)
    with pytest.raises(cli.KubeError, match="exited"):
        backend._wait_for_workers("agg", "gwab-agg-kv")
    assert len(calls) == 1  # failed on the first check, no retries


def test_every_container_gets_a_kubernetes_like_open_file_limit():
    backend, docker = _backend()
    backend.deploy_loadgen()
    backend.deploy_topology("agg", "fp")
    for name, call in _runs(docker).items():
        assert "nofile=1048576:1048576" in call, name


def test_loadgen_recreated_when_its_spec_changes(monkeypatch):
    backend, docker = _backend()
    seen = []

    def run(args, **kwargs):
        seen.append(list(args))
        out = "true oldhash\n" if args[0] == "inspect" else ""
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(docker, "run", run)
    backend.deploy_loadgen()
    assert ["rm", "-f", local_backend.loadgen_name(backend.cfg)] in seen
    created = next(a for a in seen if a[:2] == ["run", "-d"])
    assert any(a.startswith(f"{local_backend.LABEL}/spec=") for a in created)
    assert "/results:rw,mode=1777" in created


def test_runs_with_different_prefixes_do_not_touch_each_other():
    one, docker_one = _backend("--name-prefix", "u1")
    two, _ = _backend("--name-prefix", "u2")
    assert one.url("agg", "epp") != two.url("agg", "epp")
    assert local_backend.loadgen_name(one.cfg) != local_backend.loadgen_name(two.cfg)
    one.teardown_topology("agg")
    filters = [c[c.index("--filter") + 1] for c in docker_one.calls if "--filter" in c]
    assert filters == [f"label={local_backend.TOPOLOGY_LABEL}=u1/agg"]
    assert all("u2" not in " ".join(c) for c in docker_one.calls)


def test_teardown_refuses_to_keep_a_stale_discovery_store():
    docker = _RecordingDocker(fail=[(["volume", "rm"], "volume is in use")])
    backend, _ = _backend(docker=docker)
    # The volume is still there afterwards (inspect succeeds), so its model
    # cards would pass the readiness check of the next deployment.
    with pytest.raises(cli.KubeError, match="cannot remove volume gwab-ab-agg-kv"):
        backend.teardown_topology("agg")


def test_preflight_reports_missing_docker(monkeypatch):
    backend, docker = _backend()

    def missing(args, **kwargs):
        raise cli.KubeError("docker not found on PATH")

    monkeypatch.setattr(docker, "run", missing)
    assert backend.preflight() == ["docker not found on PATH"]


def test_identity_tracks_the_epp_binary_by_content(tmp_path):
    epp = tmp_path / "epp"
    epp.write_bytes(b"build one")
    backend, _ = _backend("--epp-binary", str(epp))
    first = backend.deployment_identity()["epp_binary_sha256"]
    epp.write_bytes(b"build two")  # rebuilt at the same path
    backend_again, _ = _backend("--epp-binary", str(epp))
    assert backend_again.deployment_identity()["epp_binary_sha256"] != first
    images = backend.deployment_identity()["image_ids"]
    assert set(images) == {
        backend.cfg.frontend_image,
        backend.cfg.worker_image,
        local_backend.DEFAULT_ENVOY_IMAGE,
    }


def test_deployed_fingerprint_needs_every_container_to_agree(monkeypatch):
    backend, docker = _backend()
    for out, want in (("fp\nfp\n", "fp"), ("fp\nold\n", None), ("", None)):
        monkeypatch.setattr(
            docker,
            "run",
            lambda args, out=out, **kw: subprocess.CompletedProcess(args, 0, out, ""),
        )
        assert backend.deployed_fingerprint("agg") == want


def _diagnose(monkeypatch, state, logs, topology="disagg"):
    backend, docker = _backend()

    def run(args, **kwargs):
        out = state if args[0] == "inspect" else logs
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(docker, "run", run)
    return backend.diagnose(topology, "epp")


def test_diagnose_tells_a_static_capable_epp_from_an_old_one(monkeypatch):
    static = "DYN_EPP_STATIC_ENDPOINT is set: skipping the Kubernetes pod reflector"
    crashed = _diagnose(monkeypatch, "exited", static + "\npanicked at src/epp.rs")
    assert "lacks DYN_EPP_STATIC_ENDPOINT" not in crashed
    old = _diagnose(monkeypatch, "exited", "POD_NAMESPACE is not set")
    assert "lacks DYN_EPP_STATIC_ENDPOINT" in old
    waiting = _diagnose(monkeypatch, "running", static)
    assert "has not discovered a prefill worker" in waiting
    assert "predates" not in waiting
