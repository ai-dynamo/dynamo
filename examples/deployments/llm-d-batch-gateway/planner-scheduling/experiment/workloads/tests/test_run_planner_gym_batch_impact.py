# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import run_planner_gym_batch_impact as campaign

UTC = timezone.utc

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]


def _parsed_args(
    tmp_path: Path,
    *,
    arm: str = "stock",
    extra: list[str] | None = None,
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"custom_id":"request-0"}\n', encoding="utf-8")
    trace = tmp_path / "online-trace.jsonl"
    trace.write_text(
        '{"timestamp":0,"input_length":32,"output_length":8,"hash_ids":[1]}\n',
        encoding="utf-8",
    )
    planner_gym_root = tmp_path / "planner-gym"
    planner_gym_root.mkdir(exist_ok=True)
    planner_gym_script = planner_gym_root / "planner_gym.py"
    planner_gym_script.write_text("# test fixture\n", encoding="utf-8")
    argv = [
        "--arm",
        arm,
        "--experiment-root",
        str(campaign.EXPERIMENT_ROOT),
        "--dynamo-repo-root",
        str(campaign.DYNAMO_REPO_ROOT),
        "--planner-gym-root",
        str(planner_gym_root),
        "--planner-gym-script",
        str(planner_gym_script),
        "--planner-gym-python",
        sys.executable,
        "--aiperf-executable",
        sys.executable,
        "--dataset",
        str(dataset),
        "--trace",
        str(trace),
        "--endpoint-url",
        "http://frontend.test:8000",
        "--batch-base-url",
        "http://gateway.test:8001",
        "--model",
        "Qwen/Qwen3-0.6B",
    ]
    if extra:
        argv.extend(extra)
    return campaign.parse_args(argv)


@pytest.mark.unit
def test_make_run_id_is_utc_and_suffix_is_injectable() -> None:
    timestamp = datetime(2026, 9, 18, 1, 2, 3, tzinfo=UTC)

    assert campaign.make_run_id(timestamp, "a1b2c3") == (
        "20260918T010203Z-planner-gym-batch-impact-a1b2c3"
    )


@pytest.mark.unit
def test_match_documents_expand_to_exactly_one_real_cell(tmp_path: Path) -> None:
    args = _parsed_args(
        tmp_path,
        arm="online-only",
        extra=[
            "--trace-name",
            "short-online",
            "--trace-block-size",
            "512",
            "--seed",
            "7",
            "--max-requests",
            "23",
            "--arrival-speedup",
            "2",
        ],
    )

    endpoint_catalog, match_config = campaign._match_documents(args)

    assert args.planner_gym_python == sys.executable
    assert args.pod_name_regex == campaign.CAMPAIGN_POD_NAME_REGEX
    evidence_args = campaign._online_only_evidence_args(args)
    assert evidence_args.run_kind == "baseline"
    assert evidence_args.pod_name_regex == campaign.CAMPAIGN_POD_NAME_REGEX
    assert endpoint_catalog["endpoints"] == [
        {
            "name": "batch-impact-endpoint",
            "url": "http://frontend.test:8000",
            "model": "Qwen/Qwen3-0.6B",
            "endpoint_type": "chat",
            "description": "Existing endpoint measured by the online-only arm.",
        }
    ]
    assert match_config["backend"]["type"] == "real"
    assert match_config["backend"]["aiperf"]["extra_args"] == [
        "--isl-block-size",
        "512",
        "--record-processor-service-count",
        "1",
    ]
    assert len(match_config["backend"]["autoscalers"]) == 1
    assert match_config["evaluations"]["traces"] == [
        {
            "name": "short-online",
            "path": str(args.trace),
            "block_size": 512,
            "presorted": True,
        }
    ]
    assert match_config["evaluations"]["defaults"] == {
        "seed": 7,
        "max_requests": 23,
        "arrival_speedup": 2.0,
    }
    assert len(match_config["slo_profiles"]) == 1
    assert match_config["execution"] == {
        "repetitions": 1,
        "fail_fast": True,
        "max_runs": 1,
    }


@pytest.mark.unit
def test_live_defaults_preserve_full_trace_and_persistent_backlog(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "online-trace.jsonl"
    trace.write_text(
        '{"timestamp":0,"input_length":32,"output_length":8,"hash_ids":[1]}\n',
        encoding="utf-8",
    )

    args = campaign.parse_args(
        [
            "--arm",
            "stock",
            "--trace",
            str(trace),
            "--endpoint-url",
            "http://frontend.test:8000",
            "--model",
            "Qwen/Qwen3-0.6B",
        ]
    )
    _endpoint_catalog, match_config = campaign._match_documents(args)

    assert args.aiperf_executable == str(campaign.DEFAULT_AIPERF_EXECUTABLE)
    assert args.max_requests is None
    assert match_config["evaluations"]["defaults"]["max_requests"] is None
    assert args.batch_size == 4000
    assert args.expected_gate_type == ["prometheus-query"]


@pytest.mark.unit
def test_stock_batch_command_forwards_external_contract(tmp_path: Path) -> None:
    args = _parsed_args(
        tmp_path,
        extra=[
            "--namespace",
            "sachalm",
            "--context",
            "cluster-a",
            "--expected-gate-type",
            "constant",
            "--expected-gate-type",
            "always-open",
            "--expected-worker-pool-id",
            "batch-pool",
            "--metrics-url",
            "async=http://metrics.test:9090/metrics",
        ],
    )
    run_dir = tmp_path / "outer-run"

    command = campaign._batch_command(args, run_dir)

    assert command[:2] == [args.batch_python, str(args.batch_harness)]
    assert command[command.index("--run-kind") + 1] == "baseline"
    assert command[command.index("--namespace") + 1] == "sachalm"
    assert command[command.index("--tenant") + 1] == "planner-poc-baseline"
    assert command[command.index("--context") + 1] == "cluster-a"
    assert command.count("--expected-gate-type") == 2
    assert "always-open" in command
    assert command[command.index("--expected-worker-pool-id") + 1] == "batch-pool"
    assert command[command.index("--abort-cleanup-timeout-seconds") + 1] == "120.0"
    assert "async=http://metrics.test:9090/metrics" in command
    assert "kubectl" not in command


@pytest.mark.unit
def test_planner_native_defaults_and_command_are_explicit(tmp_path: Path) -> None:
    args = _parsed_args(
        tmp_path,
        arm="planner-native",
        extra=[
            "--native-planner-configmap",
            "batch-planner-config",
            "--metrics-url",
            "frontend=http://frontend.test:8000/metrics",
            "--metrics-url",
            "async=http://async.test:9090/metrics",
        ],
    )

    campaign.validate_args(args)
    command = campaign._batch_command(args, tmp_path / "run")

    assert args.expected_gate_type == ["redis-leased-rate"]
    assert args.native_planner_pod_name_regex == (r"^qwen3-0-6b-batch-planner-")
    assert command[command.index("--run-kind") + 1] == "planner-native"
    assert command[command.index("--native-planner-configmap") + 1] == (
        "batch-planner-config"
    )
    assert command[command.index("--native-planner-decision-log-regex") + 1] == (
        campaign.baseline_harness.NATIVE_PLANNER_DEFAULT_DECISION_LOG_REGEX
    )


@pytest.mark.unit
def test_embedded_url_credentials_are_rejected_before_traffic(tmp_path: Path) -> None:
    args = _parsed_args(
        tmp_path,
        extra=[
            "--endpoint-url",
            "http://user:secret@frontend.test:8000/v1?token=secret",
        ],
    )

    with pytest.raises(campaign.CampaignError, match="must not embed"):
        campaign.validate_args(args)


@pytest.mark.unit
def test_reportable_metric_endpoints_are_required_before_traffic(
    tmp_path: Path,
) -> None:
    missing = _parsed_args(
        tmp_path,
        arm="online-only",
        extra=["--metrics-url", "frontend=http://frontend.test:8000/metrics"],
    )

    with pytest.raises(campaign.CampaignError, match=r"missing: async"):
        campaign.validate_args(missing)

    complete = _parsed_args(
        tmp_path,
        arm="online-only",
        extra=[
            "--metrics-url",
            "frontend=http://frontend.test:8000/metrics",
            "--metrics-url",
            "async=http://async.test:9090/metrics",
        ],
    )
    campaign.validate_args(complete)

    unexpected = _parsed_args(
        tmp_path,
        arm="online-only",
        extra=[
            "--metrics-url",
            "frontend=http://frontend.test:8000/metrics",
            "--metrics-url",
            "async=http://async.test:9090/metrics",
            "--metrics-url",
            "gateway=http://gateway.test:8001/metrics",
        ],
    )
    with pytest.raises(campaign.CampaignError, match=r"unexpected: gateway"):
        campaign.validate_args(unexpected)


@pytest.mark.unit
def test_duplicate_reportable_metric_endpoint_is_rejected(tmp_path: Path) -> None:
    args = _parsed_args(
        tmp_path,
        arm="online-only",
        extra=[
            "--metrics-url",
            "frontend=http://frontend.test:8000/metrics",
            "--metrics-url",
            "frontend=http://other-frontend.test:8000/metrics",
            "--metrics-url",
            "async=http://async.test:9090/metrics",
        ],
    )

    with pytest.raises(campaign.CampaignError, match="duplicate metric endpoint"):
        campaign.validate_args(args)


@pytest.mark.unit
def test_child_logs_redact_credentials_and_url_queries(tmp_path: Path) -> None:
    fake_hf_token = "hf_" + ("x" * 30)
    source = io.StringIO(
        f"HF_TOKEN={fake_hf_token} "
        "url=https://user:password@example.test/path?token=secret\n"
        "  - name: API_TOKEN\n"
        "    value: multiline-secret-value\n"
        "  - name: VISIBLE_SETTING\n"
        "    value: visible\n"
    )
    log_path = tmp_path / "child.log"
    errors: list[str] = []

    campaign._copy_stream(source, log_path, errors)

    logged = log_path.read_text(encoding="utf-8")
    assert errors == []
    assert fake_hf_token not in logged
    assert "password" not in logged
    assert "token=secret" not in logged
    assert "multiline-secret-value" not in logged
    assert "    value: visible" in logged
    assert "<redacted>" in logged


@pytest.mark.integration
def test_planner_process_starts_only_after_batch_dispatch_is_active(
    tmp_path: Path,
) -> None:
    order_path = tmp_path / "order.txt"
    release_path = tmp_path / "planner-started"
    batch_script = tmp_path / "fake_batch.py"
    planner_script = tmp_path / "fake_planner.py"
    batch_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        "order = Path(sys.argv[1])\n"
        "release = Path(sys.argv[2])\n"
        "order.write_text('created\\n', encoding='utf-8')\n"
        "print('created batch batch-test', flush=True)\n"
        'print(\'baseline-harness-event {"event":"batch-created",\'\n'
        '      \'"batch_id":"batch-test",\'\n'
        '      \'"observed_at":"2026-09-18T00:00:00.000Z"}\', flush=True)\n'
        "time.sleep(0.1)\n"
        "with order.open('a', encoding='utf-8') as output:\n"
        "    output.write('active\\n')\n"
        'print(\'baseline-harness-event {"event":"batch-active",\'\n'
        '      \'"batch_id":"batch-test",\'\n'
        '      \'"observed_at":"2026-09-18T00:00:01.000Z",\'\n'
        '      \'"status":"in_progress"}\', flush=True)\n'
        "time.sleep(0.1)\n"
        "with order.open('a', encoding='utf-8') as output:\n"
        "    output.write('dispatch-active\\n')\n"
        'print(\'baseline-harness-event {"event":"batch-dispatch-active",\'\n'
        '      \'"batch_id":"batch-test",\'\n'
        '      \'"observed_at":"2026-09-18T00:00:02.000Z",\'\n'
        '      \'"status":"in_progress",\'\n'
        '      \'"completed":1,"failed":0,"total":10}\', flush=True)\n'
        "deadline = time.monotonic() + 5\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "with order.open('a', encoding='utf-8') as output:\n"
        "    output.write('batch-finished\\n')\n",
        encoding="utf-8",
    )
    planner_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "order = Path(sys.argv[1])\n"
        "release = Path(sys.argv[2])\n"
        "with order.open('a', encoding='utf-8') as output:\n"
        "    output.write('planner\\n')\n"
        "release.write_text('started\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs"
    batch = campaign._start_process(
        "batch",
        [sys.executable, str(batch_script), str(order_path), str(release_path)],
        cwd=tmp_path,
        log_dir=log_dir,
        watch_batch_marker=True,
    )

    dispatch = campaign._wait_for_batch_dispatch_active(batch, 5)
    planner = campaign._start_process(
        "planner",
        [sys.executable, str(planner_script), str(order_path), str(release_path)],
        cwd=tmp_path,
        log_dir=log_dir,
    )
    batch_exit, planner_exit = campaign._wait_concurrent(batch, planner, 5)

    assert dispatch.batch_id == "batch-test"
    assert dispatch.created_observed_at <= dispatch.in_progress_observed_at
    assert dispatch.in_progress_status == "in_progress"
    assert dispatch.dispatch_active_status == "in_progress"
    assert dispatch.dispatch_active_at == "2026-09-18T00:00:02.000Z"
    assert (dispatch.completed, dispatch.failed, dispatch.total) == (1, 0, 10)
    assert (batch_exit, planner_exit) == (0, 0)
    assert order_path.read_text(encoding="utf-8").splitlines() == [
        "created",
        "active",
        "dispatch-active",
        "planner",
        "batch-finished",
    ]


@pytest.mark.integration
def test_batch_dispatch_barrier_fails_closed_before_starting_online(
    tmp_path: Path,
) -> None:
    batch_script = tmp_path / "fake_batch.py"
    batch_script.write_text(
        "import time\n"
        "print('created batch batch-never-active', flush=True)\n"
        'print(\'baseline-harness-event {"event":"batch-created",\'\n'
        '      \'"batch_id":"batch-never-active",\'\n'
        '      \'"observed_at":"2026-09-18T00:00:00.000Z"}\', flush=True)\n'
        'print(\'baseline-harness-event {"event":"batch-active",\'\n'
        '      \'"batch_id":"batch-never-active",\'\n'
        '      \'"observed_at":"2026-09-18T00:00:01.000Z",\'\n'
        '      \'"status":"in_progress"}\', flush=True)\n'
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    batch = campaign._start_process(
        "batch",
        [sys.executable, str(batch_script)],
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        watch_batch_marker=True,
        termination_grace_seconds=0.1,
    )

    with pytest.raises(campaign.CampaignError, match="online load was not started"):
        campaign._wait_for_batch_dispatch_active(batch, 0.2)

    assert batch.marker_state["batch_id"] == "batch-never-active"
    assert batch.process.poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
@pytest.mark.integration
def test_terminate_process_kills_the_owned_process_group(tmp_path: Path) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    parent_script = tmp_path / "process_tree.py"
    parent_script.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    handle = campaign._start_process(
        "tree",
        [sys.executable, str(parent_script), str(descendant_pid_path)],
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
    )
    descendant_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while not descendant_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert descendant_pid_path.exists()
        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        assert handle.process_group_id == handle.process.pid
        assert os.getpgid(descendant_pid) == handle.process_group_id

        campaign._terminate_process(handle)

        assert handle.process.poll() is not None
        assert not handle.stdout_thread.is_alive()
        assert not handle.stderr_thread.is_alive()
        with pytest.raises(ProcessLookupError):
            os.kill(descendant_pid, 0)
    finally:
        campaign._cleanup_processes([handle])
        if descendant_pid is not None:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
@pytest.mark.integration
def test_wait_one_cleans_up_after_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = campaign._start_process(
        "interrupted",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
    )
    original_wait = handle.process.wait
    interrupted = False

    def interrupt_once(timeout: float | None = None) -> int:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("test interrupt")
        return original_wait(timeout=timeout)

    monkeypatch.setattr(handle.process, "wait", interrupt_once)

    with pytest.raises(KeyboardInterrupt, match="test interrupt"):
        campaign._wait_one(handle, 5)

    assert handle.process.poll() is not None
    assert not campaign._process_tree_exists(handle)
    assert not handle.stdout_thread.is_alive()
    assert not handle.stderr_thread.is_alive()
