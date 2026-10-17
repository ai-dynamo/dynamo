# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.gateway_ab import cli, loadgen, manifests

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


def _bench(tmp_path, *extra):
    args = cli.build_parser().parse_args(
        ["run", "--context", "test-ctx", "--results", str(tmp_path), *extra]
    )
    return cli.Bench(args)


def _fake_result(records):
    return {
        "records": records,
        "errors": 0,
        "request_throughput": 1.0,
        "output_token_throughput": 1.0,
        "samples": {
            "time_to_first_token": [1.0] * records,
            "inter_token_latency": [1.0] * records,
            "request_latency": [1.0] * records,
        },
    }


def test_aiperf_command_differs_only_in_url():
    w = loadgen.Workload()
    common = dict(
        model="m",
        headers={"X-A": "1"},
        workload=w,
        point=loadgen.LoadPoint("c", 8),
        seed=1,
        artifact_dir="/results/x",
    )
    a = loadgen.aiperf_command(url="http://fe:8000", **common)
    b = loadgen.aiperf_command(url="http://gw:80", **common)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert diff == [("http://fe:8000", "http://gw:80")]
    assert "--header" in a and "X-A:1" in a
    assert a[a.index("--prefix-prompt-length") + 1] == "512"


def test_prefix_flags_omitted_when_disabled():
    w = loadgen.Workload(prefix_length=0)
    cmd = loadgen.aiperf_command(
        model="m",
        url="u",
        headers={},
        workload=w,
        point=loadgen.LoadPoint("c", 1),
        seed=0,
        artifact_dir="d",
    )
    assert "--prefix-prompt-length" not in cmd


def test_request_count_defaults():
    w = loadgen.Workload()
    c4, c64 = loadgen.LoadPoint("c", 4), loadgen.LoadPoint("c", 64)
    assert w.requests_for(c4) == 200
    assert w.requests_for(c64) == 640
    assert w.warmup_for(c4) == 10
    assert w.requests_for(loadgen.LoadPoint("rps", 20)) == 800
    assert w.warmup_for(loadgen.LoadPoint("rps", 20)) == 100
    assert loadgen.Workload(request_count=7).requests_for(c64) == 7


def test_rate_mode_replaces_concurrency():
    w = loadgen.Workload(request_rates=(2.5, 10))
    assert [p.label for p in w.points()] == ["rps2.5", "rps10"]
    cmd = loadgen.aiperf_command(
        model="m",
        url="u",
        headers={},
        workload=w,
        point=w.points()[0],
        seed=0,
        artifact_dir="d",
    )
    assert "--concurrency" not in cmd
    assert cmd[cmd.index("--request-rate") + 1] == "2.5"
    assert cmd[cmd.index("--arrival-pattern") + 1] == "poisson"


def test_load_point_labels_round_trip():
    for label in ("c8", "rps12.5", "rps3"):
        assert loadgen.LoadPoint.parse(label).label == label
    assert loadgen.LoadPoint("c", 8) < loadgen.LoadPoint("c", 32)
    assert loadgen.LoadPoint.parse("c8").describe() == "c=8"
    assert loadgen.LoadPoint.parse("rps2.5").describe() == "2.5 req/s"
    for bad in ("prime", "c", "x8", "c-1"):
        with pytest.raises(ValueError):
            loadgen.LoadPoint.parse(bad)


def test_measure_alternates_order_and_resumes(tmp_path, monkeypatch):
    bench = _bench(
        tmp_path, "--concurrency", "4,16", "--rounds", "2", "--topologies", "agg"
    )
    monkeypatch.setattr(bench, "url", lambda topology, side: f"http://{side}")
    calls = []

    def fake_run(cmd, artifact_dir, timeout_s):
        calls.append(artifact_dir)
        conc = float(cmd[cmd.index("--concurrency") + 1])
        return _fake_result(bench.workload.requests_for(loadgen.LoadPoint("c", conc)))

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    assert bench.measure_topology("agg", tmp_path) == 0
    # Both sides are primed once, unrecorded, before round 0.
    assert calls[:2] == ["/results/agg/frontend/prime", "/results/agg/gateway/prime"]
    assert not (tmp_path / "runs/agg/frontend/prime").exists()
    sides = [c.split("/")[3] for c in calls[2:]]
    # (round 0, c=4): A,B  (round 0, c=16): B,A  (round 1, c=4): B,A  (round 1, c=16): A,B
    assert sides == [
        "frontend",
        "gateway",
        "gateway",
        "frontend",
        "gateway",
        "frontend",
        "frontend",
        "gateway",
    ]
    saved = json.loads((tmp_path / "runs/agg/gateway/c16/r1.json").read_text())
    assert saved["expected_requests"] == 200 and saved["seed"] == 43

    calls.clear()
    assert bench.measure_topology("agg", tmp_path) == 0
    assert calls == []


def test_failed_point_is_retried_then_recorded(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "1")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")

    def always_fail(cmd, artifact_dir, timeout_s):
        raise cli.KubeError("boom")

    monkeypatch.setattr(bench.loadgen, "run", always_fail)
    # A failed priming run is logged, not fatal; each measured point is
    # attempted twice and then recorded as missing.
    assert bench.measure_topology("agg", tmp_path) == 2
    assert (tmp_path / "runs/agg/frontend/c4/r0.error").exists()
    assert not (tmp_path / "runs/agg/frontend/c4/r0.json").exists()


def test_no_prime(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "1", "--no-prime")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    calls = []

    def fake_run(cmd, artifact_dir, timeout_s):
        calls.append(artifact_dir)
        return _fake_result(200)

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    bench.measure_topology("disagg", tmp_path)
    assert not any(c.endswith("/prime") for c in calls)
    assert len(calls) == 2


def test_csv_parsing_rejects_unknown_topology():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--topologies", "agg,hybrid"])


def test_all_reports_remaining_topologies_after_a_failure(tmp_path, monkeypatch):
    args = cli.build_parser().parse_args(
        [
            "all",
            "--context",
            "test-ctx",
            "--results",
            str(tmp_path),
            "--rounds",
            "1",
            "--concurrency",
            "4",
        ]
    )
    bench = cli.Bench(args)
    torn_down = []
    monkeypatch.setattr(bench, "preflight", lambda: [])
    monkeypatch.setattr(bench, "deploy_loadgen", lambda: None)
    monkeypatch.setattr(bench, "write_meta", lambda results: None)
    monkeypatch.setattr(bench, "teardown_topology", torn_down.append)
    monkeypatch.setattr(bench, "teardown_all", lambda: None)
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")

    def deploy(topology):
        if topology == "agg":
            raise cli.KubeError("agg would not come up")

    monkeypatch.setattr(bench, "deploy_topology", deploy)
    monkeypatch.setattr(bench.loadgen, "run", lambda cmd, d, t: _fake_result(200))

    assert cli.cmd_all(bench) == 1
    assert torn_down == ["agg", "disagg"]
    assert (tmp_path / "runs/disagg/gateway/c4/r0.json").exists()
    assert "## disagg" in (tmp_path / "report.md").read_text()


def test_load_mode_selection(tmp_path):
    default = _bench(tmp_path)
    assert [p.label for p in default.workload.points()] == ["rps6", "rps14"]
    closed = _bench(tmp_path, "--concurrency", "8,32")
    assert [p.label for p in closed.workload.points()] == ["c8", "c32"]
    rates = _bench(tmp_path, "--request-rate", "3,9.5")
    assert [p.label for p in rates.workload.points()] == ["rps3", "rps9.5"]
    with pytest.raises(SystemExit):
        cli.main(["run", "--context", "x", "--concurrency", "8", "--request-rate", "3"])


def _dgd(
    generation=2, observed=2, ready="True", ready_gen=2, phase="Completed", **comp
):
    component = {"replicas": 2, "readyReplicas": 2, "updatedReplicas": 2}
    component.update(comp)
    return {
        "metadata": {"generation": generation},
        "status": {
            "observedGeneration": observed,
            "conditions": [
                {"type": "Ready", "status": ready, "observedGeneration": ready_gen}
            ],
            "rollingUpdate": {"phase": phase},
            "components": {"decode": component},
        },
    }


@pytest.mark.parametrize(
    "dgd,terminating,settled",
    [
        (_dgd(), False, True),
        (_dgd(observed=1), False, False),
        (_dgd(ready="False"), False, False),
        (_dgd(ready_gen=1), False, False),
        (_dgd(phase="InProgress"), False, False),
        (_dgd(updatedReplicas=1), False, False),
        (_dgd(readyReplicas=1), False, False),
        (_dgd(), True, False),
    ],
)
def test_check_settled(tmp_path, monkeypatch, dgd, terminating, settled):
    bench = _bench(tmp_path)
    pod = {"metadata": {"deletionTimestamp": "now"} if terminating else {}}

    def get_json(args, namespaced=True):
        return dgd if args[0] == "dynamographdeployment" else {"items": [pod]}

    monkeypatch.setattr(bench.kube, "get_json", get_json)
    if settled:
        bench._check_settled("x")
    else:
        with pytest.raises(cli.KubeError):
            bench._check_settled("x")


def test_invalid_run_is_retried_and_remeasured_on_resume(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "1", "--no-prime")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    bad = dict(_fake_result(200), errors=3)
    replies = {"frontend": [bad, bad], "gateway": [bad, _fake_result(200)]}

    def fake_run(cmd, artifact_dir, timeout_s):
        return replies[artifact_dir.split("/")[3]].pop(0)

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    assert bench.measure_topology("agg", tmp_path) == 1
    # The invalid result is kept for the report but does not count as done.
    assert (
        json.loads((tmp_path / "runs/agg/frontend/c4/r0.json").read_text())["errors"]
        == 3
    )
    assert (tmp_path / "runs/agg/frontend/c4/r0.error").exists()

    calls = []
    monkeypatch.setattr(
        bench.loadgen, "run", lambda cmd, d, t: calls.append(d) or _fake_result(200)
    )
    assert bench.measure_topology("agg", tmp_path) == 0
    assert calls == ["/results/agg/frontend/c4/r0"]


def test_resume_refuses_a_changed_workload(tmp_path, monkeypatch):
    def fake_meta_sources(bench):
        monkeypatch.setattr(bench.kube, "get_json", lambda *a, **k: {"items": []})
        monkeypatch.setattr(bench.loadgen, "aiperf_version", lambda: "0.0")

    first = _bench(tmp_path, "--isl", "1024")
    fake_meta_sources(first)
    first.write_meta(tmp_path)
    more_rounds = _bench(tmp_path, "--isl", "1024", "--rounds", "5")
    fake_meta_sources(more_rounds)
    more_rounds.write_meta(tmp_path)  # extending rounds is allowed
    changed = _bench(tmp_path, "--isl", "2048")
    fake_meta_sources(changed)
    with pytest.raises(cli.KubeError, match="different workload"):
        changed.write_meta(tmp_path)


class _FakeKube:
    def __init__(self, cat_replies, pgrep_replies=None):
        self.cat_replies = list(cat_replies)
        # pgrep exits 0 while aiperf processes remain, 1 once they are gone.
        self.pgrep_replies = list(pgrep_replies or [])
        self.commands = []

    def exec(self, pod, command, *, input=None, check=True, timeout=None):
        import subprocess

        self.commands.append(command)
        if command[0] == "cat":
            rc, out, err = self.cat_replies.pop(0)
            return subprocess.CompletedProcess(command, rc, out, err)
        if command[0] == "pgrep":
            rc = self.pgrep_replies.pop(0) if self.pgrep_replies else 1
            return subprocess.CompletedProcess(command, rc, "", "")
        if command[0] == "python3":
            return subprocess.CompletedProcess(command, 0, json.dumps({"ok": 1}), "")
        return subprocess.CompletedProcess(command, 0, "", "")


def test_loadgen_kills_strays_and_polls_until_done(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube(
        [(1, "", "cat: x: No such file or directory"), (0, "0\n", "")],
        pgrep_replies=[0, 0, 1],
    )
    out = loadgen.LoadGen(kube, "pod").run(["aiperf", "profile"], "/results/a", 60)
    assert out == {"ok": 1}
    # aiperf renames its processes, so the pattern must match those too, and
    # the launch waits until pgrep finds none.
    assert kube.commands[0] == ["pkill", "-9", "-f", "^aiperf |aiperf profile"]
    assert [c[0] for c in kube.commands[1:4]] == ["pgrep", "pgrep", "pgrep"]
    assert kube.commands[4][:2] == ["sh", "-c"]


def test_loadgen_refuses_to_launch_over_unkillable_aiperf(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube([], pgrep_replies=[0] * 30)
    with pytest.raises(cli.KubeError, match="did not exit"):
        loadgen.LoadGen(kube, "pod").run(["aiperf", "profile"], "/results/a", 60)
    assert not any(c[:2] == ["sh", "-c"] for c in kube.commands)


def test_loadgen_gives_up_when_the_pod_is_gone(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    gone = (1, "", 'Error from server (NotFound): pods "pod" not found')
    kube = _FakeKube([gone, gone, gone])
    with pytest.raises(cli.KubeError, match="cannot reach loadgen pod"):
        loadgen.LoadGen(kube, "pod").run(["aiperf", "profile"], "/results/a", 3600)


def test_load_point_labels_never_use_exponents():
    big = loadgen.LoadPoint("c", 1_000_000)
    assert big.label == "c1000000"
    assert loadgen.LoadPoint.parse(big.label) == big
    assert loadgen.LoadPoint("rps", 0.25).label == "rps0.25"


@pytest.mark.parametrize(
    "argv",
    [
        ["--concurrency", "0"],
        ["--concurrency", "8,-1"],
        ["--request-rate", "0"],
        ["--rounds", "0"],
    ],
)
def test_non_positive_values_are_rejected(argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", *argv])


def test_success_clears_stale_failure_marker(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "1", "--no-prime")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    marker = tmp_path / "runs/agg/frontend/c4/r0.error"
    marker.parent.mkdir(parents=True)
    marker.write_text("failed earlier\n")
    monkeypatch.setattr(bench.loadgen, "run", lambda cmd, d, t: _fake_result(200))
    assert bench.measure_topology("agg", tmp_path) == 0
    assert not marker.exists()


def test_loadgen_pod_recreated_when_spec_changes(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--hf-secret", "hf")
    old = manifests.build_loadgen_pod(manifests.BenchConfig())  # no secret
    old["status"] = {"phase": "Running"}
    calls = []
    monkeypatch.setattr(bench.kube, "get_json", lambda args, **k: old)
    monkeypatch.setattr(bench.kube, "kubectl", lambda args, **k: calls.append(args))
    monkeypatch.setattr(bench.kube, "apply", lambda objs: calls.append(["apply"]))
    monkeypatch.setattr(bench.kube, "wait", lambda *a: None)
    bench.deploy_loadgen()
    assert calls[0][:2] == ["delete", "pod"] and calls[1] == ["apply"]

    calls.clear()
    same = manifests.build_loadgen_pod(bench.cfg)
    same["status"] = {"phase": "Running"}
    monkeypatch.setattr(bench.kube, "get_json", lambda args, **k: same)
    bench.deploy_loadgen()
    assert calls == []
