# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from benchmarks.gateway_ab import cli, kube, loadgen, manifests, stats

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


def _bench(tmp_path, *extra, command="run"):
    args = cli.build_parser().parse_args(
        [command, "--context", "test-ctx", "--results", str(tmp_path), *extra]
    )
    bench = cli.Bench(args)
    # Measuring checks the deployment's fingerprint; pretend it matches.
    bench.backend.deployed_fingerprint = lambda topology: bench.fingerprint()
    return bench


def _result(errors=0, ttft=50.0):
    """What LoadGen.run returns for a run of 200 requests."""
    return {
        "records": 200,
        "errors": errors,
        "request_throughput": 1.0,
        "output_token_throughput": 1.0,
        "latency": {
            metric: dict.fromkeys(stats.LATENCY_STATS.values(), ttft)
            for metric in stats.LATENCY_METRICS
        },
    }


def test_a_and_b_requests_differ_only_in_the_route_header(tmp_path, monkeypatch):
    bench = _bench(
        tmp_path,
        "--concurrency",
        "4",
        "--rounds",
        "1",
        "--no-prime",
        "--gateway-url",
        "http://gw:80",
        "--with-direct",
    )
    commands = {}

    def fake_run(cmd, artifact_dir, timeout_s):
        commands[artifact_dir.split("/")[3]] = cmd
        return _result()

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    assert bench.measure_topology("agg", tmp_path) == 0
    a, b, direct = commands["frontend"], commands["epp"], commands["direct"]
    target = manifests.TARGET_HEADER
    assert len(a) == len(b)
    assert [(x, y) for x, y in zip(a, b) if x != y] == [
        ("/results/agg/frontend/c4/r0", "/results/agg/epp/c4/r0"),
        (f"{target}:gateway-ab/ab-agg-fe", f"{target}:gateway-ab/ab-agg-epp"),
    ]
    # Both through the gateway; only the direct arm reaches its Frontend itself.
    assert a[a.index("--url") + 1] == "http://gw:80"
    assert direct[direct.index("--url") + 1] == "http://ab-agg-dir-frontend:8000"
    # aiperf prepends the shared prefix, so --isl 1024 asks it for 512 more.
    assert a[a.index("--synthetic-input-tokens-mean") + 1] == "512"


def test_measure_alternates_order_and_shares_seeds(tmp_path, monkeypatch):
    bench = _bench(
        tmp_path, "--concurrency", "4,16", "--rounds", "2", "--topologies", "agg"
    )
    monkeypatch.setattr(bench, "url", lambda topology, side: f"http://{side}")
    calls = []

    def fake_run(cmd, artifact_dir, timeout_s):
        calls.append((artifact_dir, cmd[cmd.index("--random-seed") + 1]))
        return _result()

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    assert bench.measure_topology("agg", tmp_path) == 0
    # Both sides are primed once, unrecorded, before round 0.
    assert [d for d, _ in calls[:2]] == [
        "/results/agg/frontend/prime",
        "/results/agg/epp/prime",
    ]
    measured = calls[2:]
    # (round 0, c=4): A,B  (round 0, c=16): B,A  (round 1, c=4): B,A  (round 1, c=16): A,B
    assert [d.split("/")[3] for d, _ in measured] == [
        "frontend",
        "epp",
        "epp",
        "frontend",
        "epp",
        "frontend",
        "frontend",
        "epp",
    ]
    # Both sides of a pair replay one seed, and no two pairs share one, so a
    # later point never replays an earlier point's prompts.
    seeds = [seed for _, seed in measured]
    assert all(seeds[i] == seeds[i + 1] for i in range(0, len(seeds), 2))
    assert sorted(set(seeds)) == ["42", "43", "44", "45"]


def test_direct_arm_runs_next_to_a_with_the_same_seeds(tmp_path, monkeypatch):
    bench = _bench(
        tmp_path,
        "--concurrency",
        "4,16",
        "--rounds",
        "2",
        "--topologies",
        "agg",
        "--with-direct",
    )
    monkeypatch.setattr(bench, "url", lambda topology, side: f"http://{side}")
    calls = []

    def fake_run(cmd, artifact_dir, timeout_s):
        calls.append((artifact_dir.split("/")[3], cmd[cmd.index("--random-seed") + 1]))
        return _result()

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    assert bench.measure_topology("agg", tmp_path) == 0
    measured = calls[3:]
    # Each point runs every arm with one seed; A is always in the middle, so
    # it runs right next to both arms it is compared with.
    for i in range(0, len(measured), 3):
        arms = [side for side, _ in measured[i : i + 3]]
        assert arms[1] == "frontend"
        assert sorted(arms) == ["direct", "epp", "frontend"]
        assert len({seed for _, seed in measured[i : i + 3]}) == 1
    firsts = [measured[i][0] for i in range(0, len(measured), 3)]
    assert firsts == ["direct", "epp", "epp", "direct"]


def test_failed_run_is_recorded_not_replayed(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "2", "--no-prime")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    calls = []

    def first_session(cmd, artifact_dir, timeout_s):
        calls.append(artifact_dir)
        if artifact_dir == "/results/agg/frontend/c4/r1":
            raise cli.KubeError("boom")
        # One failed request makes B's round 0 invalid.
        return _result(errors=1 if artifact_dir == "/results/agg/epp/c4/r0" else 0)

    monkeypatch.setattr(bench.loadgen, "run", first_session)
    assert bench.measure_topology("agg", tmp_path) == 2
    # Each run was tried once: a replay would find the prompts its first
    # attempt sent already cached, which its pair partner never had.
    assert len(calls) == len(set(calls)) == 4
    assert (tmp_path / "runs/agg/frontend/c4/r1.error").exists()

    calls.clear()
    monkeypatch.setattr(
        bench.loadgen, "run", lambda cmd, d, t: calls.append(d) or _result()
    )
    assert bench.measure_topology("agg", tmp_path) == 0
    # A resume measures both incomplete rounds on every arm, so no pair mixes
    # runs from two sessions.
    assert sorted(calls) == [
        "/results/agg/epp/c4/r0",
        "/results/agg/epp/c4/r1",
        "/results/agg/frontend/c4/r0",
        "/results/agg/frontend/c4/r1",
    ]
    assert not (tmp_path / "runs/agg/frontend/c4/r1.error").exists()


@pytest.mark.parametrize("command", ["all", "run"])
def test_one_failing_topology_does_not_stop_the_others(tmp_path, monkeypatch, command):
    bench = _bench(tmp_path, "--rounds", "1", "--concurrency", "4", command=command)
    for name in (
        "deploy_loadgen",
        "deploy_topology",
        "teardown_topology",
        "teardown_all",
    ):
        monkeypatch.setattr(bench, name, lambda *args: None)
    monkeypatch.setattr(bench, "preflight", lambda: [])
    monkeypatch.setattr(bench, "write_meta", lambda results: None)
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    monkeypatch.setattr(bench.loadgen, "run", lambda cmd, d, t: _result())
    # agg never came up.
    fingerprint = bench.fingerprint()
    bench.backend.deployed_fingerprint = lambda topology: (
        None if topology == "agg" else fingerprint
    )
    assert cli.COMMANDS[command](bench) == 1
    assert (tmp_path / "runs/disagg/epp/c4/r0.json").exists()
    assert "## disagg" in (tmp_path / "report.md").read_text()


def test_resume_refuses_other_settings_but_not_an_empty_directory(
    tmp_path, monkeypatch
):
    def bench(*extra):
        b = _bench(tmp_path, *extra)
        monkeypatch.setattr(b.backend.kube, "get_json", lambda *a, **k: {"items": []})
        monkeypatch.setattr(b.loadgen, "aiperf_version", lambda: "0.0")
        return b

    bench("--isl", "1024").write_meta(tmp_path)
    # Nothing measured yet, so a start with other flags takes the directory.
    bench("--isl", "2048").write_meta(tmp_path)
    run = tmp_path / "runs/agg/epp/c4/r0.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}")
    bench("--isl", "2048", "--rounds", "5").write_meta(tmp_path)
    # Extending the rounds is recorded, so the report names the real count.
    assert json.loads((tmp_path / "meta.json").read_text())["rounds"] == 5
    for other in (
        ("--isl", "1024"),
        ("--isl", "2048", "--gateway-url", "http://gw"),
        ("--isl", "2048", "--no-prime"),
    ):
        with pytest.raises(cli.RefusedError):
            bench(*other).write_meta(tmp_path)
    monkeypatch.setattr(cli, "HARNESS_VERSION", cli.HARNESS_VERSION + 1)
    with pytest.raises(cli.RefusedError, match="harness version"):
        bench("--isl", "2048").write_meta(tmp_path)


def test_run_refuses_a_deployment_made_with_other_flags(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "1")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    monkeypatch.setattr(bench.loadgen, "run", lambda cmd, d, t: _result())
    bench.backend.deployed_fingerprint = lambda topology: "deployed-elsewhere"
    with pytest.raises(cli.KubeError, match="other settings"):
        bench.measure_topology("agg", tmp_path)
    bench.backend.deployed_fingerprint = lambda topology: None
    with pytest.raises(cli.KubeError, match="not deployed"):
        bench.measure_topology("agg", tmp_path)


def test_kubernetes_deploy_stamps_the_fingerprint_run_checks(tmp_path, monkeypatch):
    bench = _bench(tmp_path, command="deploy")
    del bench.backend.deployed_fingerprint  # the real lookup, not the stand-in
    backend = bench.backend
    applied = {}
    monkeypatch.setattr(backend.kube, "exists", lambda args, **k: False)
    monkeypatch.setattr(
        backend.kube,
        "apply",
        lambda objs: applied.update({o["metadata"]["name"]: o for o in objs}),
    )
    monkeypatch.setattr(
        backend.kube, "get_json", lambda args, **k: applied.get(args[1])
    )
    monkeypatch.setattr(backend, "_check_settled", lambda name: None)
    backend.deploy_topology("agg", bench.fingerprint())
    assert backend.deployed_fingerprint("agg") == bench.fingerprint()
    assert _bench(tmp_path, "--agg-workers", "3").fingerprint() != bench.fingerprint()


def test_all_refuses_an_incompatible_resume_before_deploying(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"identity": {}}))
    run = tmp_path / "runs/agg/epp/c4/r0.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}")
    bench = _bench(tmp_path, command="all")
    deployed = []
    monkeypatch.setattr(bench, "preflight", lambda: [])
    monkeypatch.setattr(bench, "deploy_loadgen", lambda: deployed.append("loadgen"))
    with pytest.raises(cli.RefusedError):
        cli.cmd_all(bench)
    assert deployed == []


def test_all_tears_down_when_the_loadgen_fails(tmp_path, monkeypatch):
    bench = _bench(tmp_path, command="all")
    torn_down = []
    monkeypatch.setattr(bench, "preflight", lambda: [])

    def fail():
        raise cli.KubeError("loadgen never became Ready")

    monkeypatch.setattr(bench, "deploy_loadgen", fail)
    monkeypatch.setattr(bench, "teardown_all", lambda: torn_down.append("all"))
    with pytest.raises(cli.KubeError):
        cli.cmd_all(bench)
    assert torn_down == ["all"]


def test_report_does_not_pass_a_gate_it_could_not_evaluate(tmp_path):
    for side, ttft in (("frontend", 50.0), ("epp", 80.0)):
        path = tmp_path / "runs" / "agg" / side / "c8" / "r0.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_result(ttft=ttft)))
    bench = _bench(tmp_path, "--max-ttft-regression", "10", command="report")
    # One round per side cannot evaluate the gate, so the run is not a pass.
    assert cli.cmd_report(bench, tmp_path) == 1
    # summary.json stays strict JSON although one-round intervals are NaN.
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text
    json.loads(text)


@pytest.mark.parametrize(
    "argv",
    [
        ["--topologies", "agg,hybrid"],
        ["--concurrency", "0"],
        ["--request-rate", "inf"],
        ["--request-rate", "14,14"],
        ["--max-ttft-regression", "nan"],
        ["--isl", "256"],
        ["--concurrency", "8", "--request-rate", "3"],
        ["--backend", "local", "--context", "x"],
        ["--epp-binary", "/x"],
    ],
)
def test_bad_flags_are_rejected_before_anything_runs(argv):
    with pytest.raises(SystemExit):
        cli.main(["run", *argv])


_SUMMARY = {
    "aiperf_version": "0.12.0",
    "request_count": {"avg": 200},
    "request_throughput": {"avg": 14.0},
    "output_token_throughput": {"avg": 1700.0},
    "time_to_first_token": {"avg": 40.0, "p50": 39.0, "p90": 42.0, "p99": 50.0},
    "inter_token_latency": {"avg": 7.0, "p50": 7.0, "p90": 7.5, "p99": 8.0},
    "request_latency": {"avg": 950.0, "p50": 940.0, "p90": 990.0, "p99": 1000.0},
    "error_summary": [],
}


def _is_pgrep(command):
    return command[:2] == ["sh", "-c"] and "pgrep" in command[2]


def _is_launch(command):
    return command[:2] == ["sh", "-c"] and "nohup" in command[2]


class _FakeKube:
    def __init__(self, rc_replies, pgrep_replies=(), exec_fails=False, summary=None):
        self.rc_replies = list(rc_replies)
        # pgrep's status: 0 while aiperf processes remain, 1 once they are gone.
        self.pgrep_replies = list(pgrep_replies)
        # Every pgrep exec fails the way an unreachable pod does.
        self.exec_fails = exec_fails
        self.summary = json.dumps(_SUMMARY) if summary is None else summary
        self.commands = []

    def exec(self, pod, command, *, input=None, check=True, timeout=None):
        self.commands.append(command)
        if command[0] == "cat" and command[1].endswith(".rc"):
            rc, out, err = self.rc_replies.pop(0)
            return subprocess.CompletedProcess(command, rc, out, err)
        if command[0] == "cat":
            return subprocess.CompletedProcess(command, 0, self.summary, "")
        if _is_pgrep(command):
            if self.exec_fails:
                return subprocess.CompletedProcess(
                    command, 1, "", "error: unable to upgrade connection"
                )
            rc = self.pgrep_replies.pop(0) if self.pgrep_replies else 1
            return subprocess.CompletedProcess(command, 0, f"{rc}\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")


def _run(kube):
    return loadgen.LoadGen(kube, "pod").run(["aiperf", "profile"], "/results/a", 60)


def test_loadgen_waits_out_exec_failures_and_reads_aiperf_summary(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    restarting = (1, "", "error: error dialing backend: connection refused")
    kube = _FakeKube(
        [
            (1, "", "cat: x: No such file or directory"),
            *[restarting] * 4,
            (0, "0\n", ""),
        ],
        pgrep_replies=[0, 0, 1],
    )
    out = _run(kube)
    assert out["records"] == 200 and out["errors"] == 0
    assert out["latency"]["time_to_first_token"]["p90"] == 42.0
    # aiperf renames its processes, so the pattern must match those too, and
    # the launch waits until pgrep finds none.
    assert kube.commands[0] == ["pkill", "-9", "-f", "^[a]iperf |[a]iperf profile"]
    assert all(_is_pgrep(c) for c in kube.commands[1:4])
    assert _is_launch(kube.commands[4])


def test_loadgen_gives_up_when_the_pod_is_gone(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube([(1, "", 'Error from server (NotFound): pods "pod" not found')])
    with pytest.raises(cli.KubeError, match="is gone"):
        _run(kube)


@pytest.mark.parametrize("exec_fails", [True, False])
def test_loadgen_never_launches_over_aiperf_it_cannot_confirm_stopped(
    monkeypatch, exec_fails
):
    # An exec failing with exit 1 must not read as pgrep finding nothing, and
    # an aiperf that survives pkill must not get a second run on top of it.
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube([], pgrep_replies=[0] * 30, exec_fails=exec_fails)
    with pytest.raises(cli.KubeError, match="cannot confirm"):
        _run(kube)
    assert not any(_is_launch(c) for c in kube.commands)


def test_launch_backgrounds_only_the_run(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube([(0, "0\n", "")])
    _run(kube)
    launch = next(c[2] for c in kube.commands if _is_launch(c))
    setup, _, rest = launch.partition("{ nohup")
    # The setup runs in the foreground; the trailing & sits inside the braces.
    # No lone `&` (background) in the setup, only `&&` chaining.
    assert setup.replace("&&", "").count("&") == 0
    assert rest.rstrip().endswith("& }")


def test_unreadable_summary_fails_the_run_not_the_harness(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    with pytest.raises(cli.KubeError, match="cannot parse"):
        _run(_FakeKube([(0, "0\n", "")], summary=""))


def _kube_reply(monkeypatch, k, rc, stderr):
    monkeypatch.setattr(
        k,
        "kubectl",
        lambda args, **kw: subprocess.CompletedProcess(args, rc, "{}", stderr),
    )


def test_kube_lookup_reports_only_not_found_as_absent(monkeypatch):
    k = kube.Kube("ctx", "ns")
    _kube_reply(monkeypatch, k, 1, 'Error from server (NotFound): pods "x" not found')
    assert k.get_json(["pod", "x"]) is None and not k.exists(["pod", "x"])
    _kube_reply(
        monkeypatch, k, 1, 'error: the server doesn\'t have a resource type "gw"'
    )
    assert k.get_json(["gw", "x"]) is None
    # A timeout is not absence: it must not skip a recreate or end a wait.
    _kube_reply(monkeypatch, k, 1, "Unable to connect to the server: i/o timeout")
    with pytest.raises(cli.KubeError, match="i/o timeout"):
        k.exists(["dynamographdeployment", "x"])
    _kube_reply(
        monkeypatch, k, 1, 'Error from server (Forbidden): crd "x" is forbidden'
    )
    with pytest.raises(kube.ForbiddenError):
        k.exists(["crd", "x"], namespaced=False)


def test_preflight_tolerates_namespace_scoped_credentials(tmp_path, monkeypatch):
    def problems(*extra):
        bench = _bench(tmp_path, "--hf-secret", "my-hf-token", *extra)
        k = bench.backend.kube
        monkeypatch.setattr(
            k,
            "kubectl",
            lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""),
        )

        def get_json(args, namespaced=True):
            # CRDs are cluster-scoped and the Gateway is in another namespace.
            if args[0] in ("crd", "-n"):
                raise kube.ForbiddenError("Error from server (Forbidden)")
            return None  # the Secret

        monkeypatch.setattr(k, "get_json", get_json)
        return bench.preflight()

    secret = "the Secret named by --hf-secret was not found in gateway-ab"
    assert problems() == [
        "not allowed to read Gateway agentgateway-system/inference-gateway; "
        "pass --gateway-url",
        secret,
    ]
    assert problems("--gateway-url", "http://gw") == [secret]


def test_gateway_url_uses_the_port_of_an_http_listener(tmp_path, monkeypatch):
    bench = _bench(tmp_path)
    gateway = {
        "spec": {"listeners": [{"name": "http", "port": 80, "protocol": "HTTP"}]}
    }
    # Istio's generated Service lists its status port first.
    services = {
        "items": [
            {
                "metadata": {"name": "inference-gateway-istio"},
                "spec": {"ports": [{"port": 15021}, {"port": 80}]},
            }
        ]
    }
    monkeypatch.setattr(
        bench.backend.kube,
        "get_json",
        lambda args, **k: gateway if "gateway" in args else services,
    )
    assert bench.backend.url("agg", "epp") == (
        "http://inference-gateway-istio.agentgateway-system.svc.cluster.local:80"
    )


def _dgd(generation=2, observed=2, ready_gen=2, phase="Completed", **comp):
    component = {"replicas": 2, "readyReplicas": 2, "updatedReplicas": 2}
    component.update(comp)
    return {
        "metadata": {"generation": generation},
        "status": {
            "observedGeneration": observed,
            "conditions": [
                {"type": "Ready", "status": "True", "observedGeneration": ready_gen}
            ],
            "rollingUpdate": {"phase": phase},
            "components": {"decode": component},
        },
    }


@pytest.mark.parametrize(
    "dgd,pod,settled",
    [
        (_dgd(), {}, True),
        (_dgd(observed=1), {}, False),
        (_dgd(ready_gen=1), {}, False),
        (_dgd(phase="InProgress"), {}, False),
        (_dgd(readyReplicas=1), {}, False),
        (_dgd(), {"deletionTimestamp": "now"}, False),
    ],
)
def test_check_settled_waits_for_the_current_generation(
    tmp_path, monkeypatch, dgd, pod, settled
):
    bench = _bench(tmp_path)
    pods = {"items": [{"metadata": pod}]}
    monkeypatch.setattr(
        bench.backend.kube,
        "get_json",
        lambda args, **k: dgd if args[0] == "dynamographdeployment" else pods,
    )
    if settled:
        bench.backend._check_settled("x")
    else:
        with pytest.raises(cli.KubeError):
            bench.backend._check_settled("x")


def test_crash_looping_pod_fails_the_wait_at_once(tmp_path, monkeypatch):
    bench = _bench(tmp_path)
    waiting = {"reason": "CrashLoopBackOff", "message": "back-off 5m0s"}
    pod = {
        "metadata": {"name": "ab-agg-epp-decode-0"},
        "status": {
            "containerStatuses": [{"name": "main", "state": {"waiting": waiting}}]
        },
    }
    monkeypatch.setattr(
        bench.backend.kube, "get_json", lambda args, **k: {"items": [pod]}
    )
    with pytest.raises(kube.FatalError, match="ab-agg-epp-decode-0.*CrashLoopBackOff"):
        bench.backend._check_settled("x")


def test_loadgen_pod_recreated_when_spec_changes(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--hf-secret", "hf")
    old = manifests.build_loadgen_pod(manifests.BenchConfig())  # no secret
    old["status"] = {"phase": "Running"}
    calls = []
    monkeypatch.setattr(bench.backend.kube, "get_json", lambda args, **k: old)
    monkeypatch.setattr(
        bench.backend.kube, "kubectl", lambda args, **k: calls.append(args)
    )
    monkeypatch.setattr(
        bench.backend.kube, "apply", lambda objs: calls.append(["apply"])
    )
    monkeypatch.setattr(bench.backend.kube, "wait", lambda *a: None)
    bench.backend.deploy_loadgen()
    assert calls[0][:2] == ["delete", "pod"] and calls[1] == ["apply"]

    calls.clear()
    same = manifests.build_loadgen_pod(bench.cfg)
    same["status"] = {"phase": "Running"}
    monkeypatch.setattr(bench.backend.kube, "get_json", lambda args, **k: same)
    bench.backend.deploy_loadgen()
    assert calls == []
