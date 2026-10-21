# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import signal
import subprocess

import pytest

from benchmarks.gateway_ab import cli, kube, loadgen, manifests

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
    # aiperf prepends the prefix, so --isl 1024 asks it for 512 more tokens.
    assert a[a.index("--synthetic-input-tokens-mean") + 1] == "512"


def test_isl_must_exceed_the_shared_prefix():
    with pytest.raises(ValueError, match="prefix-length"):
        loadgen.Workload(isl=512, prefix_length=512)
    assert loadgen.Workload(isl=512, prefix_length=512, num_prefixes=0)
    with pytest.raises(SystemExit):
        cli.main(["run", "--context", "x", "--isl", "256"])


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
    assert cmd[cmd.index("--synthetic-input-tokens-mean") + 1] == "1024"


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
    assert calls[:2] == ["/results/agg/frontend/prime", "/results/agg/epp/prime"]
    assert not (tmp_path / "runs/agg/frontend/prime").exists()
    sides = [c.split("/")[3] for c in calls[2:]]
    # (round 0, c=4): A,B  (round 0, c=16): B,A  (round 1, c=4): B,A  (round 1, c=16): A,B
    assert sides == [
        "frontend",
        "epp",
        "epp",
        "frontend",
        "epp",
        "frontend",
        "frontend",
        "epp",
    ]
    saved = json.loads((tmp_path / "runs/agg/epp/c16/r1.json").read_text())
    assert saved["expected_requests"] == 200
    # One seed per round and load point, the same for both sides: a later
    # point must not replay an earlier point's prompts.
    seeds = {
        (side, label, rnd): json.loads(
            (tmp_path / f"runs/agg/{side}/{label}/r{rnd}.json").read_text()
        )["seed"]
        for side in manifests.SIDES
        for label in ("c4", "c16")
        for rnd in range(2)
    }
    for label in ("c4", "c16"):
        for rnd in range(2):
            assert seeds[("frontend", label, rnd)] == seeds[("epp", label, rnd)]
    assert sorted(
        seeds[("frontend", lb, r)] for lb in ("c4", "c16") for r in range(2)
    ) == [
        42,
        43,
        44,
        45,
    ]

    calls.clear()
    assert bench.measure_topology("agg", tmp_path) == 0
    assert calls == []


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
        return _fake_result(200)

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    assert bench.measure_topology("agg", tmp_path) == 0
    assert [side for side, _ in calls[:3]] == ["direct", "frontend", "epp"]
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
        return _fake_result(200)

    monkeypatch.setattr(bench.loadgen, "run", fake_run)
    bench.measure_topology("agg", tmp_path)
    a, b, direct = commands["frontend"], commands["epp"], commands["direct"]
    target = manifests.TARGET_HEADER
    assert len(a) == len(b)
    assert [(x, y) for x, y in zip(a, b) if x != y] == [
        ("/results/agg/frontend/c4/r0", "/results/agg/epp/c4/r0"),
        (f"{target}:ab-agg-fe", f"{target}:ab-agg-epp"),
    ]
    # Both through the gateway; only the direct arm reaches its Frontend itself.
    assert a[a.index("--url") + 1] == "http://gw:80"
    assert direct[direct.index("--url") + 1] == "http://ab-agg-dir-frontend:8000"


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
    bench.backend.deployed_fingerprint = lambda topology: bench.fingerprint()
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
    assert (tmp_path / "runs/disagg/epp/c4/r0.json").exists()
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

    monkeypatch.setattr(bench.backend.kube, "get_json", get_json)
    if settled:
        bench.backend._check_settled("x")
    else:
        with pytest.raises(cli.KubeError):
            bench.backend._check_settled("x")


def test_invalid_run_is_retried_and_remeasured_on_resume(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "1", "--no-prime")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    bad = dict(_fake_result(200), errors=3)
    replies = {"frontend": [bad, bad], "epp": [bad, _fake_result(200)]}

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
        monkeypatch.setattr(
            bench.backend.kube, "get_json", lambda *a, **k: {"items": []}
        )
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
    # Not only the workload: the target and the deployment count too.
    elsewhere = _bench(tmp_path, "--isl", "1024", "--gateway-url", "http://gw")
    fake_meta_sources(elsewhere)
    with pytest.raises(cli.KubeError, match="different workload"):
        elsewhere.write_meta(tmp_path)
    # Extending the rounds is recorded, so the report names the real count.
    assert json.loads((tmp_path / "meta.json").read_text())["rounds"] == 5


def test_resume_refuses_a_run_from_an_older_harness(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"workload": {}}))
    bench = _bench(tmp_path)
    with pytest.raises(cli.KubeError, match="harness version"):
        bench.check_resume(tmp_path)


def test_run_refuses_a_deployment_made_with_other_flags(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--concurrency", "4", "--rounds", "1")
    monkeypatch.setattr(bench, "url", lambda topology, side: "http://x")
    monkeypatch.setattr(bench.loadgen, "run", lambda cmd, d, t: _fake_result(200))
    bench.backend.deployed_fingerprint = lambda topology: "deployed-elsewhere"
    with pytest.raises(cli.KubeError, match="other settings"):
        bench.measure_topology("agg", tmp_path)
    bench.backend.deployed_fingerprint = lambda topology: None
    with pytest.raises(cli.KubeError, match="not deployed"):
        bench.measure_topology("agg", tmp_path)


def test_fingerprint_follows_the_deployment_settings(tmp_path):
    base = _bench(tmp_path).fingerprint()
    assert _bench(tmp_path, "--isl", "2048").fingerprint() == base
    assert _bench(tmp_path, "--agg-workers", "3").fingerprint() != base
    assert _bench(tmp_path, "--frontend-image", "x:y").fingerprint() != base


def test_all_refuses_an_incompatible_resume_before_deploying(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"identity": {}}))
    bench = _bench(tmp_path, command="all")
    deployed = []
    monkeypatch.setattr(bench, "preflight", lambda: [])
    monkeypatch.setattr(bench, "deploy_loadgen", lambda: deployed.append("loadgen"))
    with pytest.raises(cli.KubeError, match="different workload"):
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


def test_sigterm_unwinds_so_teardown_runs():
    previous = cli._exit_on_sigterm()
    try:
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as exc:
            handler(signal.SIGTERM, None)
        assert exc.value.code == 128 + signal.SIGTERM
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize(
    "argv",
    [
        ["--backend", "local", "--context", "x"],
        ["--backend", "local", "--hf-secret", "hf"],
        ["--backend", "local", "--router-cpu", "4"],
        ["--epp-binary", "/tmp/epp"],
        ["--envoy-image", "envoy:x"],
    ],
)
def test_flags_for_the_other_backend_are_rejected(argv):
    with pytest.raises(SystemExit):
        cli.main(["run", *argv])


def test_report_does_not_pass_a_gate_it_could_not_evaluate(tmp_path):
    for side, ttft in (("frontend", 50.0), ("epp", 80.0)):
        path = tmp_path / "runs" / "agg" / side / "c8" / "r0.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                dict(
                    _fake_result(10),
                    samples={
                        "time_to_first_token": [ttft] * 10,
                        "inter_token_latency": [5.0] * 10,
                        "request_latency": [500.0] * 10,
                    },
                )
            )
        )
    bench = _bench(tmp_path, "--max-ttft-regression", "10", command="report")
    # One round per side cannot evaluate the gate, so the run is not a pass.
    assert cli.cmd_report(bench, tmp_path) == 1
    # summary.json stays strict JSON although one-round CVs are NaN.
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text
    json.loads(text)


def _is_pgrep(command):
    return command[:2] == ["sh", "-c"] and "pgrep" in command[2]


def _is_launch(command):
    return command[:2] == ["sh", "-c"] and "nohup" in command[2]


class _FakeKube:
    def __init__(self, cat_replies, pgrep_replies=None, exec_fails=False):
        self.cat_replies = list(cat_replies)
        # pgrep's status: 0 while aiperf processes remain, 1 once they are gone.
        self.pgrep_replies = list(pgrep_replies or [])
        # Every pgrep exec fails the way an unreachable pod does.
        self.exec_fails = exec_fails
        self.commands = []

    def exec(self, pod, command, *, input=None, check=True, timeout=None):
        self.commands.append(command)
        if command[0] == "cat":
            rc, out, err = self.cat_replies.pop(0)
            return subprocess.CompletedProcess(command, rc, out, err)
        if _is_pgrep(command):
            if self.exec_fails:
                return subprocess.CompletedProcess(
                    command, 1, "", "error: unable to upgrade connection"
                )
            rc = self.pgrep_replies.pop(0) if self.pgrep_replies else 1
            return subprocess.CompletedProcess(command, 0, f"{rc}\n", "")
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
    assert kube.commands[0] == ["pkill", "-9", "-f", "^[a]iperf |[a]iperf profile"]
    assert all(_is_pgrep(c) for c in kube.commands[1:4])
    assert _is_launch(kube.commands[4])


def test_failed_exec_is_not_taken_for_no_aiperf_left(monkeypatch):
    """An exec failing with exit 1 must not read as pgrep finding nothing."""
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube([], exec_fails=True)
    with pytest.raises(cli.KubeError, match="cannot confirm"):
        loadgen.LoadGen(kube, "pod").run(["aiperf", "profile"], "/results/a", 60)
    assert not any(_is_launch(c) for c in kube.commands)


def test_kill_pattern_does_not_match_its_own_shell():
    import re

    pattern = re.compile(loadgen._AIPERF_PROCS)
    assert pattern.search("aiperf system_controller")
    assert pattern.search("sh -c aiperf profile --model m")
    check = f"sh -c pgrep -f '{loadgen._AIPERF_PROCS}' >/dev/null; echo $?"
    assert not pattern.search(check)


def test_loadgen_refuses_to_launch_over_unkillable_aiperf(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube([], pgrep_replies=[0] * 30)
    with pytest.raises(cli.KubeError, match="cannot confirm"):
        loadgen.LoadGen(kube, "pod").run(["aiperf", "profile"], "/results/a", 60)
    assert not any(_is_launch(c) for c in kube.commands)


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


def test_launch_backgrounds_only_the_run(monkeypatch):
    monkeypatch.setattr(loadgen.time, "sleep", lambda s: None)
    kube = _FakeKube([(0, "0\n", "")])
    loadgen.LoadGen(kube, "pod").run(["aiperf", "profile"], "/results/a", 60)
    launch = next(c[2] for c in kube.commands if _is_launch(c))
    setup, _, rest = launch.partition("{ nohup")
    # The setup runs in the foreground; the trailing & sits inside the braces.
    # No lone `&` (background) in the setup, only `&&` chaining.
    assert setup.replace("&&", "").count("&") == 0
    assert rest.rstrip().endswith("& }")


def test_preflight_names_the_flag_not_the_secret(tmp_path, monkeypatch):
    bench = _bench(tmp_path, "--hf-secret", "my-hf-token")
    kube = bench.backend.kube
    monkeypatch.setattr(kube, "kubectl", lambda *a, **k: "")
    monkeypatch.setattr(kube, "exists", lambda args, **k: args[0] != "secret")
    monkeypatch.setattr(
        kube,
        "get_json",
        lambda *a, **k: {
            "status": {"conditions": [{"type": "Programmed", "status": "True"}]}
        },
    )
    problems = bench.preflight()
    assert problems == [
        f"the Secret named by --hf-secret was not found in {kube.namespace}"
    ]
    assert not any("my-hf-token" in p for p in problems)


@pytest.mark.parametrize(
    "argv",
    [
        ["--requests", "-5"],
        ["--warmup-requests", "-1"],
        ["--run-timeout", "0"],
        ["--isl", "0"],
        ["--speedup-ratio", "0"],
        ["--agg-workers", "0"],
        ["--block-size", "0"],
    ],
)
def test_invalid_numbers_are_rejected_before_anything_runs(argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", *argv])


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
