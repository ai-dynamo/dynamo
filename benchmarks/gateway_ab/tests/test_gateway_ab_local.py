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

    def __init__(self):
        self.calls = []

    def run(self, args, *, input=None, check=True, timeout=600):
        self.calls.append(list(args))
        out = ""
        if args[:2] == ["inspect", "--format"] and "IPAddress" in args[2]:
            out = "172.18.0.9\n"
        elif args[:2] == ["run", "--rm"]:
            # The model-card count: report every worker registered.
            out = json.dumps(
                {"ab-agg-fe": 2, "ab-agg-gw": 2, "ab-disagg-fe": 4, "ab-disagg-gw": 4}
            )
        elif args[:1] == ["ps"]:
            out = ""
        return subprocess.CompletedProcess(args, 0, out, "")


def _backend(*extra):
    args = cli.build_parser().parse_args(["all", "--backend", "local", *extra])
    bench = cli.Bench(args)
    docker = _RecordingDocker()
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
def test_both_sides_run_identical_workers(topology):
    backend, docker = _backend()
    backend.deploy_topology(topology)
    runs = _runs(docker)
    cfg = backend.cfg
    for ctype, replicas, mode in manifests.worker_roles(cfg, topology):
        for i in range(replicas):
            a = runs[f"gwab-{topology}-fe-{ctype}{i}"]
            b = runs[f"gwab-{topology}-gw-{ctype}{i}"]
            assert _command(a, cfg.worker_image) == _command(b, cfg.worker_image)
            assert _command(a, cfg.worker_image)[3:] == manifests.mocker_args(cfg, mode)
            assert _env(a)["DYN_NAMESPACE"] == f"ab-{topology}-fe"
            assert _env(b)["DYN_NAMESPACE"] == f"ab-{topology}-gw"


def test_epp_points_at_the_direct_frontend_without_kubernetes():
    backend, docker = _backend("--epp-binary", "/tmp/epp")
    backend.deploy_topology("agg")
    runs = _runs(docker)
    epp = runs["gwab-agg-epp"]
    env = _env(epp)
    assert env["DYN_EPP_STATIC_ENDPOINT"] == "172.18.0.9:8000"
    assert env["DYN_SECURE_SERVING"] == "false"
    assert env["DYN_NAMESPACE"] == "ab-agg-gw"
    assert env["DYN_KV_CACHE_BLOCK_SIZE"] == str(backend.cfg.block_size)
    assert "/tmp/epp:/epp-dev:ro" in epp and "/epp-dev" in epp
    direct = runs["gwab-agg-direct"]
    assert _command(direct, backend.cfg.frontend_image)[-2:] == [
        "--router-mode",
        "direct",
    ]
    assert _env(runs["gwab-agg-frontend"])["DYN_ROUTER_MODE"] == "kv"


def test_deploy_starts_from_a_clean_slate():
    backend, docker = _backend()
    backend.deploy_topology("disagg")
    first_create = next(
        i for i, c in enumerate(docker.calls) if c[:2] == ["network", "create"]
    )
    assert ["volume", "rm", "-f", "gwab-disagg-kv"] in docker.calls[:first_create]
    assert ["network", "connect", "gwab-disagg", local_backend.LOADGEN] in docker.calls


def test_envoy_config_matches_what_the_epp_requires():
    config = json.loads(local_backend.envoy_config("gwab-agg-epp"))
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
    epp_addr = clusters["epp"]["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    assert (
        epp_addr["endpoint"]["address"]["socket_address"]["address"] == "gwab-agg-epp"
    )


def test_urls_use_container_names():
    backend, _ = _backend()
    assert backend.url("agg", "frontend") == "http://gwab-agg-frontend:8000"
    assert backend.url("disagg", "gateway") == "http://gwab-disagg-gateway:8080"


def test_waits_until_every_worker_registered(monkeypatch):
    backend, docker = _backend("--ready-timeout", "6")
    monkeypatch.setattr(
        local_backend.retry.__globals__["time"], "sleep", lambda s: None
    )
    replies = [{"ab-agg-fe": 2, "ab-agg-gw": 1}, {"ab-agg-fe": 2, "ab-agg-gw": 2}]

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
    backend.deploy_topology("agg")
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
    assert ["rm", "-f", local_backend.LOADGEN] in seen
    created = next(a for a in seen if a[:2] == ["run", "-d"])
    assert any(a.startswith(f"{local_backend.LABEL}/spec=") for a in created)
    assert "/results:rw,mode=1777" in created
