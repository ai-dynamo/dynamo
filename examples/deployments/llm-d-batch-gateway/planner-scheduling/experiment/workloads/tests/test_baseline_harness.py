# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import baseline_harness as harness
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_make_run_id_is_utc_and_suffix_is_injectable() -> None:
    timestamp = datetime(2026, 8, 28, 12, 34, 56, tzinfo=UTC)

    assert harness.make_run_id(timestamp, "abcdef") == (
        "20260828T123456Z-baseline-abcdef"
    )
    assert (
        harness.make_run_id(timestamp, "abcdef", kind="planner-controlled")
        == "20260828T123456Z-planner-controlled-abcdef"
    )
    assert (
        harness.make_run_id(timestamp, "abcdef", kind="planner-native")
        == "20260828T123456Z-planner-native-abcdef"
    )


def test_effective_request_total_uses_declared_bootstrap_then_gateway() -> None:
    batch = {"metadata": {"planner_request_count": "100"}}

    assert harness.effective_request_total(batch, 0, 100) == (
        100,
        "planner_request_count",
    )
    assert harness.effective_request_total(batch, 100, 100) == (100, "gateway")

    with pytest.raises(harness.HarnessError, match="expected 100"):
        harness.effective_request_total(batch, 99, 100)
    with pytest.raises(harness.HarnessError, match="planner_request_count"):
        harness.effective_request_total({"metadata": None}, 0, 100)


def test_planner_controlled_runs_require_controller_pairing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")

    unpaired = harness.parse_args(
        ["--dataset", str(dataset), "--run-kind", "planner-controlled"]
    )
    with pytest.raises(harness.HarnessError, match="required"):
        harness.validate_args(unpaired)

    paired = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--run-kind",
            "planner-controlled",
            "--paired-controller-run-id",
            "20260828T183813Z-planner-loop-15424f",
        ]
    )
    harness.validate_args(paired)

    malformed = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--run-kind",
            "planner-controlled",
            "--paired-controller-run-id",
            "not-a-controller-run",
        ]
    )
    with pytest.raises(harness.HarnessError, match="invalid run ID"):
        harness.validate_args(malformed)

    baseline = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--paired-controller-run-id",
            "20260828T183813Z-planner-loop-15424f",
        ]
    )
    with pytest.raises(harness.HarnessError, match="only with"):
        harness.validate_args(baseline)


def test_planner_native_runs_have_distinct_control_plane_contract(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")

    missing_config = harness.parse_args(
        ["--dataset", str(dataset), "--run-kind", "planner-native"]
    )
    with pytest.raises(harness.HarnessError, match="configmap is required"):
        harness.validate_args(missing_config)

    native = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--run-kind",
            "planner-native",
            "--native-planner-configmap",
            "qwen3-0-6b-batch-planner-config",
            "--native-planner-pod-name-regex",
            r"qwen3-0-6b-batch-planner.*planner",
        ]
    )
    harness.validate_args(native)

    assert native.expected_gate_type == ["redis-leased-rate"]
    assert native.paired_controller_run_id is None
    assert native.native_planner_decision_log_regex == (r"Batch scheduling decision:")
    assert native.native_planner_min_decision_logs == 2
    assert harness.control_plane_metadata(native) == {
        "mode": "native-planner",
        "standalone_controller_run_id": None,
        "native_planner": {
            "pod_name_regex": r"qwen3-0-6b-batch-planner.*planner",
            "configmap": "qwen3-0-6b-batch-planner-config",
            "decision_log_regex": r"Batch scheduling decision:",
            "minimum_decision_logs": 2,
        },
    }

    paired_native = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--run-kind",
            "planner-native",
            "--native-planner-configmap",
            "qwen3-0-6b-batch-planner-config",
            "--paired-controller-run-id",
            "20260828T183813Z-planner-loop-15424f",
        ]
    )
    with pytest.raises(harness.HarnessError, match="only with"):
        harness.validate_args(paired_native)

    wrong_gate = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--run-kind",
            "planner-native",
            "--native-planner-configmap",
            "qwen3-0-6b-batch-planner-config",
            "--expected-gate-type",
            "constant",
        ]
    )
    with pytest.raises(harness.HarnessError, match="redis-leased-rate"):
        harness.validate_args(wrong_gate)

    empty_log_match = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--run-kind",
            "planner-native",
            "--native-planner-configmap",
            "qwen3-0-6b-batch-planner-config",
            "--native-planner-decision-log-regex",
            "",
        ]
    )
    with pytest.raises(harness.HarnessError, match="cannot be empty"):
        harness.validate_args(empty_log_match)


def test_native_planner_options_are_rejected_for_baseline(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    args = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--native-planner-configmap",
            "qwen3-0-6b-batch-planner-config",
        ]
    )

    with pytest.raises(harness.HarnessError, match="only with"):
        harness.validate_args(args)


def test_native_planner_evidence_requires_identity_config_and_recurring_ticks(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    args = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--run-kind",
            "planner-native",
            "--native-planner-configmap",
            "qwen3-0-6b-batch-planner-config",
        ]
    )
    harness.validate_args(args)
    evidence = {
        "native_planner": {
            "matched_pods": ["qwen3-0-6b-batch-planner-abc-planner-def"],
            "running_pods": ["qwen3-0-6b-batch-planner-abc-planner-def"],
            "expected_configmap_mounted": True,
            "expected_configmap_captured": True,
            "decision_log_match_count": 2,
        }
    }

    harness.validate_native_planner_evidence(args, evidence, require_decisions=False)
    harness.validate_native_planner_evidence(args, evidence, require_decisions=True)

    evidence["native_planner"]["decision_log_match_count"] = 1
    with pytest.raises(harness.HarnessError, match="observed=1, required=2"):
        harness.validate_native_planner_evidence(args, evidence, require_decisions=True)


def test_prepare_workload_is_deterministic_and_does_not_modify_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gsm8k.jsonl"
    rows = [
        {
            "custom_id": f"source-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "source-model",
                "messages": [{"role": "user", "content": f"question {index}"}],
                "temperature": 0.9,
                "max_tokens": 17,
            },
        }
        for index in range(5)
    ]
    _write_dataset(source, rows)
    source_before = source.read_bytes()

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    controlled = tmp_path / "controlled.jsonl"
    first_summary = harness.normalize_workload(
        source=source,
        destination=first,
        manifest_path=tmp_path / "first-manifest.json",
        batch_size=3,
        start_index=1,
        model="target-model",
        max_tokens=128,
        temperature=0.0,
    )
    second_summary = harness.normalize_workload(
        source=source,
        destination=second,
        manifest_path=tmp_path / "second-manifest.json",
        batch_size=3,
        start_index=1,
        model="target-model",
        max_tokens=128,
        temperature=0.0,
    )
    harness.normalize_workload(
        source=source,
        destination=controlled,
        manifest_path=tmp_path / "controlled-manifest.json",
        batch_size=3,
        start_index=1,
        model="target-model",
        max_tokens=128,
        temperature=0.0,
        run_kind="planner-controlled",
    )

    assert source.read_bytes() == source_before
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == controlled.read_bytes()
    assert first_summary["source_sha256"] == second_summary["source_sha256"]
    assert first_summary["output_sha256"] == second_summary["output_sha256"]
    normalized = [json.loads(line) for line in first.read_text().splitlines()]
    assert [row["custom_id"] for row in normalized] == [
        "gsm8k-baseline-000002",
        "gsm8k-baseline-000003",
        "gsm8k-baseline-000004",
    ]
    assert all(row["body"]["model"] == "target-model" for row in normalized)
    assert all(row["body"]["max_tokens"] == 128 for row in normalized)
    assert all(row["body"]["temperature"] == 0.0 for row in normalized)
    assert all(row["body"]["stream"] is False for row in normalized)


def test_prepare_workload_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.jsonl"
    row = {
        "custom_id": "same",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"messages": [{"role": "user", "content": "hello"}]},
    }
    _write_dataset(source, [row, row])

    with pytest.raises(harness.HarnessError, match="duplicate custom_id"):
        harness.normalize_workload(
            source=source,
            destination=tmp_path / "out.jsonl",
            manifest_path=tmp_path / "manifest.json",
            batch_size=2,
            start_index=0,
            model="model",
            max_tokens=8,
            temperature=0.0,
        )


def test_prepare_workload_labels_native_planner_requests_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gsm8k.jsonl"
    _write_dataset(
        source,
        [
            {
                "custom_id": "source-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"messages": [{"role": "user", "content": "question"}]},
            }
        ],
    )
    destination = tmp_path / "batch-input.jsonl"
    manifest_path = tmp_path / "workload-manifest.json"

    manifest = harness.normalize_workload(
        source=source,
        destination=destination,
        manifest_path=manifest_path,
        batch_size=1,
        start_index=0,
        model="model",
        max_tokens=8,
        temperature=0.0,
        run_kind="planner-native",
    )

    request = json.loads(destination.read_text(encoding="utf-8"))
    assert request["custom_id"] == "gsm8k-planner-native-000001"
    assert manifest["run_kind"] == "planner-native"
    assert manifest["custom_id_prefix"] == "gsm8k-planner-native-"


def test_capture_command_preserves_nonzero_exit_and_streams(tmp_path: Path) -> None:
    result = harness.capture_command(
        tmp_path,
        "probe",
        [
            sys.executable,
            "-c",
            "import sys; print('visible'); print('failure', file=sys.stderr); sys.exit(7)",
        ],
    )

    assert result.exit_code == 7
    assert (tmp_path / "probe.stdout").read_text().strip() == "visible"
    assert (tmp_path / "probe.stderr").read_text().strip() == "failure"
    assert json.loads((tmp_path / "probe.command.json").read_text())["exit_code"] == 7


def test_redaction_removes_hugging_face_and_bearer_credentials() -> None:
    fake_hf_token = "hf_" + ("x" * 24)
    secret_text = f"HF_TOKEN={fake_hf_token} Authorization: Bearer abc.def-123"

    redacted = harness.redact_text(secret_text)

    assert fake_hf_token not in redacted
    assert "abc.def-123" not in redacted
    assert redacted.count("<redacted>") >= 2


@pytest.mark.parametrize(
    "argv,retained",
    [
        (
            [
                "harness",
                "--batch-base-url",
                "https://alice:batch-pass@example.test/v1?token=a=b#fragment",
            ],
            "example.test/v1",
        ),
        (
            [
                "harness",
                "--online-base-url=https://bob:online-pass@example.test/v1?key=value",
            ],
            "example.test/v1",
        ),
        (
            [
                "harness",
                "--metrics-url",
                "gateway=https://carol:metrics-pass@example.test/metrics?key=value",
            ],
            "gateway=https://example.test/metrics",
        ),
        (
            [
                "harness",
                "--metrics-url=gateway=https://dave:metrics-pass@example.test/metrics?key=value",
            ],
            "gateway=https://example.test/metrics",
        ),
    ],
)
def test_sanitized_command_redacts_split_and_inline_url_flags(
    argv: list[str], retained: str
) -> None:
    rendered = harness.sanitized_command(argv)

    assert retained in rendered
    for secret in (
        "alice",
        "bob",
        "carol",
        "dave",
        "batch-pass",
        "online-pass",
        "metrics-pass",
        "token=a=b",
        "key=value",
        "fragment",
    ):
        assert secret not in rendered


def test_redact_text_sanitizes_credentials_inside_arbitrary_urls() -> None:
    value = (
        "request failed at "
        "https://url-user:url-pass@example.test/private?access_token=secret#fragment "
        "while reading redis://redis-user:redis-pass@redis.test:6379/0?token=other"
    )

    redacted = harness.redact_text(value)

    assert "https://example.test/private?<redacted>" in redacted
    assert "url-user" not in redacted
    assert "url-pass" not in redacted
    assert "access_token" not in redacted
    assert "secret" not in redacted
    assert "redis://redis.test:6379/0?<redacted>" in redacted
    assert "redis-user" not in redacted
    assert "redis-pass" not in redacted
    assert "token=other" not in redacted
    assert "fragment" not in redacted


def test_capture_command_redacts_url_credentials_from_argv_artifact(
    tmp_path: Path,
) -> None:
    secret_url = "https://artifact-user:artifact-pass@example.test/v1?token=secret"

    harness.capture_command(
        tmp_path,
        "url-argv",
        [sys.executable, "-c", "pass", secret_url],
    )
    artifact = (tmp_path / "url-argv.command.json").read_text(encoding="utf-8")

    assert "https://example.test/v1?<redacted>" in artifact
    assert "artifact-user" not in artifact
    assert "artifact-pass" not in artifact
    assert "token=secret" not in artifact


def test_git_state_resolves_nested_seed_and_requires_real_revision(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "renamed-checkout"
    nested = repo / "nested" / "path"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )

    state = harness.git_state(nested, tmp_path / "artifacts")

    assert state["repo_root"] == str(repo.resolve())
    assert len(state["revision"]) == 40
    assert state["revision"] != "unknown"
    assert state["dirty"] is False


def test_git_state_fails_when_head_is_unknown(tmp_path: Path) -> None:
    repo = tmp_path / "empty-repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)

    with pytest.raises(harness.HarnessError, match="revision capture failed"):
        harness.git_state(repo, tmp_path / "artifacts")


def test_git_state_fails_when_status_capture_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40

    def fake_capture(
        _directory: Path,
        name: str,
        argv: list[str],
        *,
        timeout_seconds: float = 30,
    ) -> harness.CapturedCommand:
        del timeout_seconds
        if name == "repo-root":
            return harness.CapturedCommand(argv, 0, f"{tmp_path}\n", "")
        if name == "revision":
            return harness.CapturedCommand(argv, 0, f"{revision}\n", "")
        return harness.CapturedCommand(argv, 1, "", "status failed")

    monkeypatch.setattr(harness, "capture_command", fake_capture)

    with pytest.raises(harness.HarnessError, match="status capture failed"):
        harness.git_state(tmp_path, tmp_path / "artifacts")


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.test/metrics",
        "https://example.test/metrics?token=value",
        "ftp://example.test/metrics",
    ],
)
def test_parse_metrics_urls_rejects_secret_bearing_or_invalid_urls(url: str) -> None:
    with pytest.raises(harness.HarnessError):
        harness.parse_metrics_urls([f"gateway={url}"])


def test_validate_gate_requires_constant_by_default() -> None:
    constant = {
        "active_async_pods": ["async-dispatch-0"],
        "active_gate_evidence": [
            {
                "pod": "async-dispatch-0",
                "container": "llm-d-async",
                "gate_type": "constant",
                "source": "container-args",
            }
        ],
    }
    harness.validate_gate(constant, ["constant"], False)

    with pytest.raises(harness.HarnessError, match="expected every"):
        harness.validate_gate(
            {
                "active_async_pods": ["async-dispatch-0"],
                "active_gate_evidence": [
                    {
                        "pod": "async-dispatch-0",
                        "container": "llm-d-async",
                        "gate_type": "fast",
                        "source": "container-args",
                    }
                ],
            },
            ["constant"],
            False,
        )


def test_gate_discovery_uses_only_live_async_args_or_exact_mounted_key(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    async_pod = {
        "metadata": {
            "name": "async-dispatch-llm-d-async-0",
            "labels": {"app.kubernetes.io/name": "llm-d-async"},
        },
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--transport-config",
                        (
                            '{"queues":[{"worker_pool_id":"dynamo-batch",'
                            '"queue_name":"batch"}]}'
                        ),
                        "--pool-config-file=/etc/llm-d-async/worker-pools.json",
                    ],
                    "volumeMounts": [
                        {"name": "active-config", "mountPath": "/etc/llm-d-async"}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "active-config",
                    "configMap": {"name": "active-async-config"},
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "llm-d-async", "ready": True}],
        },
    }
    configmaps = {
        "active-async-config": {
            "data": {
                "worker-pools.json": (
                    '[{"id":"dynamo-batch","gate_type":"wait-on-refuse",'
                    '"gate_params":{"gate":{"gate_type":"redis-leased-rate",'
                    '"gate_params":{"unused":"gate_type=constant"}}}},'
                    '{"id":"other-pool","gate_type":"constant"}]'
                ),
                "stale.json": '{"gate_type":"constant"}',
            }
        },
        "unmounted-stale-config": {
            "data": {"worker-pools.json": '{"gate_type":"constant"}'}
        },
    }

    records = evidence._active_gate_evidence([async_pod], configmaps)

    assert {record["gate_type"] for record in records} == {"redis-leased-rate"}
    harness.validate_gate(
        {
            "active_async_pods": ["async-dispatch-llm-d-async-0"],
            "active_gate_evidence": records,
        },
        ["redis-leased-rate"],
        False,
    )


def test_pool_gate_discovery_does_not_accept_an_unrelated_pool(tmp_path: Path) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    config = (
        '[{"id":"dynamo-batch","gate_type":"constant",'
        '"gate_params":{"unused":"gate_type=redis-leased-rate"}},'
        '{"id":"other-pool","gate_type":"redis-leased-rate"}]'
    )

    assert evidence._configured_pool_gate_types(config) == ["constant"]


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--transport=redis-sortedset",
            "--transport-config",
            '{"batch_size":8,"queues":[{"gate_type":"constant"}]}',
        ],
        [
            "--transport=redis-sortedset",
            '--transport-config={"queues":[{"gate_type":"constant"}]}',
        ],
    ],
)
def test_transport_config_gate_discovery_supports_chart_argv(
    tmp_path: Path, arguments: list[str]
) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))

    assert evidence._transport_config_gate_types(arguments) == ["constant"]


def test_legacy_stock_gate_discovery_matches_pinned_chart_argv(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))

    assert evidence._direct_gate_types(
        [
            "--message-queue-impl=redis-sortedset",
            "--redis.ss.gate-type=constant",
        ]
    ) == ["constant"]


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--message-queue-impl=redis-sortedset",
            "--redis.ss.gate-type=constant",
            "--redis.ss.gate-type=redis-leased-rate",
        ],
        [
            "--message-queue-impl=redis-pubsub",
            "--redis.ss.gate-type=constant",
        ],
        [
            "--message-queue-impl=redis-sortedset",
            '--redis.ss.queues-config=[{"gate_type":"redis-leased-rate"}]',
            "--redis.ss.gate-type=constant",
        ],
    ],
)
def test_legacy_gate_discovery_rejects_ambiguous_or_inactive_flags(
    tmp_path: Path, arguments: list[str]
) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))

    assert evidence._direct_gate_types(arguments) == []


def test_active_gate_discovery_accepts_stock_single_transport_queue(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
        run_kind="baseline",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {
            "name": "async-dispatch-llm-d-async-0",
            "labels": {"app.kubernetes.io/name": "llm-d-async"},
        },
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--transport=redis-sortedset",
                        "--transport-config",
                        (
                            '{"batch_size":8,"queues":['
                            '{"gate_type":"constant","gate_params":{}}]}'
                        ),
                    ],
                }
            ]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "llm-d-async", "ready": True}],
        },
    }

    records = evidence._active_gate_evidence([pod], {})

    assert records == [
        {
            "pod": "async-dispatch-llm-d-async-0",
            "container": "llm-d-async",
            "gate_type": "constant",
            "source": "transport-config-queue",
            "pool_id": "default",
        }
    ]


def test_planner_pool_gate_requires_transport_queue_binding(tmp_path: Path) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
        run_kind="planner-native",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {"name": "async-dispatch-llm-d-async-0"},
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--transport-config",
                        (
                            '{"queues":[{"worker_pool_id":"other-pool",'
                            '"queue_name":"batch"}]}'
                        ),
                        "--pool-config-file=/etc/llm-d-async/worker-pools.json",
                    ],
                    "volumeMounts": [
                        {"name": "pool-config", "mountPath": "/etc/llm-d-async"}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "pool-config",
                    "configMap": {"name": "async-pool-config"},
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "llm-d-async", "ready": True}],
        },
    }
    configmaps = {
        "async-pool-config": {
            "data": {
                "worker-pools.json": (
                    '[{"id":"dynamo-batch","gate_type":"redis-leased-rate"}]'
                )
            }
        }
    }

    assert evidence._active_gate_evidence([pod], configmaps) == []

    pod["spec"]["containers"][0]["args"][1] = (
        '{"queues":[{"worker_pool_id":"dynamo-batch","queue_name":"batch"}]}'
    )
    records = evidence._active_gate_evidence([pod], configmaps)
    assert {record["gate_type"] for record in records} == {"redis-leased-rate"}
    assert {record["pool_id"] for record in records} == {"dynamo-batch"}


def test_planner_gate_discovery_matches_v090_rendered_legacy_argv(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
        run_kind="planner-native",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {
            "name": "async-dispatch-llm-d-async-0",
            "labels": {"app.kubernetes.io/name": "llm-d-async"},
        },
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--message-queue-impl=redis-sortedset",
                        "--redis.ss.queues-config",
                        (
                            '[{"id":"dynamo-batch","queue_name":"batch",'
                            '"worker_pool_id":"dynamo-batch"}]'
                        ),
                        "--pool-config-file=/etc/llm-d-async/config/worker-pools.json",
                    ],
                    "volumeMounts": [
                        {
                            "name": "ap-config",
                            "mountPath": "/etc/llm-d-async/config",
                            "readOnly": True,
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "ap-config",
                    "configMap": {"name": "async-dispatch-llm-d-async-config"},
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "llm-d-async", "ready": True}],
        },
    }
    configmaps = {
        "async-dispatch-llm-d-async-config": {
            "data": {
                "worker-pools.json": (
                    '[{"id":"dynamo-batch","gate_type":"wait-on-refuse",'
                    '"gate_params":{"gate":{"gate_type":"redis-leased-rate",'
                    '"gate_params":{"control_key":"drain-limit"}}}}]'
                )
            }
        }
    }

    records = evidence._active_gate_evidence([pod], configmaps)

    assert records == [
        {
            "pod": "async-dispatch-llm-d-async-0",
            "container": "llm-d-async",
            "gate_type": "redis-leased-rate",
            "source": "mounted-configmap-key",
            "pool_id": "dynamo-batch",
            "configmap": "async-dispatch-llm-d-async-config",
            "key": "worker-pools.json",
            "path": "/etc/llm-d-async/config/worker-pools.json",
        }
    ]


def test_planner_gate_requires_exact_mounted_pool_gate(tmp_path: Path) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
        run_kind="planner-native",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {"name": "async-dispatch-llm-d-async-0"},
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--transport-config",
                        (
                            '{"queues":[{"worker_pool_id":"dynamo-batch",'
                            '"gate_type":"redis-leased-rate"}]}'
                        ),
                        "--redis.ss.gate-type=redis-leased-rate",
                    ],
                }
            ]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "llm-d-async", "ready": True}],
        },
    }

    # In Planner mode, neither a queue gate nor a legacy direct gate proves the
    # worker-pool admission boundary used by ActionWait dispatch.
    assert evidence._active_gate_evidence([pod], {}) == []

    pod["spec"]["containers"][0].update(
        {
            "args": [
                "--transport-config",
                (
                    '{"queues":[{"worker_pool_id":"dynamo-batch",'
                    '"gate_type":"redis-leased-rate"}]}'
                ),
                "--pool-config-file=/etc/llm-d-async/worker-pools.json",
            ],
            "volumeMounts": [{"name": "pool-config", "mountPath": "/etc/llm-d-async"}],
        }
    )
    pod["spec"]["volumes"] = [
        {"name": "pool-config", "configMap": {"name": "async-pool-config"}}
    ]
    configmaps = {
        "async-pool-config": {
            "data": {
                "worker-pools.json": ('[{"id":"dynamo-batch","gate_type":"constant"}]')
            }
        }
    }

    records = evidence._active_gate_evidence([pod], configmaps)
    assert {record["gate_type"] for record in records} == {"constant"}
    with pytest.raises(harness.HarnessError, match="expected every"):
        harness.validate_gate(
            {
                "active_async_pods": ["async-dispatch-llm-d-async-0"],
                "active_gate_evidence": records,
            },
            ["redis-leased-rate"],
            False,
        )


def test_new_transport_config_takes_precedence_over_legacy_gate_args(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
        run_kind="baseline",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {"name": "async-dispatch-llm-d-async-0"},
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--transport=redis-sortedset",
                        "--transport-config",
                        '{"queues":[{"queue_name":"batch"}]}',
                        "--redis.ss.gate-type=constant",
                    ],
                }
            ]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "llm-d-async", "ready": True}],
        },
    }

    assert evidence._active_gate_evidence([pod], {}) == []


def test_transport_config_rejects_multiple_queues(tmp_path: Path) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    arguments = [
        "--transport-config",
        ('{"queues":[{"gate_type":"constant"},{"gate_type":"redis-leased-rate"}]}'),
    ]

    assert evidence._transport_config_gate_types(arguments) == []


def test_gate_discovery_reads_exact_mounted_transport_config(tmp_path: Path) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
        run_kind="baseline",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {"name": "llm-d-async-0"},
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--transport-config-file",
                        "/etc/llm-d-async/transport.json",
                    ],
                    "volumeMounts": [
                        {"name": "transport", "mountPath": "/etc/llm-d-async"}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "transport",
                    "configMap": {"name": "active-transport"},
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "llm-d-async", "ready": True}],
        },
    }
    configmaps = {
        "active-transport": {
            "data": {
                "transport.json": '{"queues":[{"gate_type":"constant"}]}',
                "stale.json": '{"queues":[{"gate_type":"redis-leased-rate"}]}',
            }
        }
    }

    records = evidence._active_gate_evidence([pod], configmaps)

    assert len(records) == 1
    assert records[0]["gate_type"] == "constant"
    assert records[0]["source"] == "mounted-transport-config-key"
    assert records[0]["key"] == "transport.json"


def test_gate_discovery_ignores_sidecar_gate_text(tmp_path: Path) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {
            "name": "async-dispatch-llm-d-async-0",
            "labels": {"app.kubernetes.io/name": "llm-d-async"},
        },
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": [
                        "--message-queue-impl=redis-sortedset",
                        "--redis.ss.gate-type=constant",
                    ],
                },
                {
                    "name": "metrics-sidecar",
                    "image": "example.test/sidecar:latest",
                    "args": ["--gate-type=redis-leased-rate"],
                },
            ]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {"name": "llm-d-async", "ready": True},
                {"name": "metrics-sidecar", "ready": True},
            ],
        },
    }

    records = evidence._active_gate_evidence([pod], {})

    assert {record["gate_type"] for record in records} == {"constant"}
    with pytest.raises(harness.HarnessError, match="expected every"):
        harness.validate_gate(
            {
                "active_async_pods": ["async-dispatch-llm-d-async-0"],
                "active_gate_evidence": records,
            },
            ["redis-leased-rate"],
            False,
        )


def test_gate_discovery_requires_running_ready_async_container(tmp_path: Path) -> None:
    args = SimpleNamespace(
        pod_name_regex=r".*",
        native_planner_pod_name_regex=None,
        native_planner_decision_log_regex=None,
        expected_worker_pool_id="dynamo-batch",
    )
    evidence = harness.KubernetesEvidence(args, tmp_path, datetime.now(UTC))
    pod = {
        "metadata": {"name": "async-dispatch-llm-d-async-0"},
        "spec": {
            "containers": [
                {
                    "name": "llm-d-async",
                    "command": ["/llm-d-async"],
                    "args": ["--gate-type=redis-leased-rate"],
                }
            ]
        },
        "status": {
            "phase": "Pending",
            "containerStatuses": [{"name": "llm-d-async", "ready": False}],
        },
    }

    assert evidence._active_async_pod_names([pod]) == []
    assert evidence._active_gate_evidence([pod], {}) == []


def test_gate_validation_rejects_any_live_async_pod_without_expected_gate() -> None:
    records = [
        {"pod": "async-0", "gate_type": "redis-leased-rate"},
        {"pod": "async-1", "gate_type": "constant"},
    ]

    with pytest.raises(harness.HarnessError, match="async-1"):
        harness.validate_gate(
            {
                "active_async_pods": ["async-0", "async-1"],
                "active_gate_evidence": records,
            },
            ["redis-leased-rate"],
            False,
        )


class _FakeOnlineResponse:
    def __init__(self, content_type: str, lines: list[bytes]) -> None:
        self.status = 200
        self.headers = {"Content-Type": content_type}
        self.lines = lines

    def __enter__(self) -> _FakeOnlineResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.lines)


@pytest.mark.parametrize(
    "content_type,lines",
    [
        ("text/event-stream", []),
        ("text/event-stream", [b"data: not-json\n", b"data: [DONE]\n"]),
        (
            "text/event-stream",
            [
                b"data: not-json\n",
                b'data: {"choices":[{"delta":{"content":"ONLINE"}}]}\n',
                b"data: [DONE]\n",
            ],
        ),
        (
            "text/event-stream",
            [
                b"data: [DONE]\n",
                b'data: {"choices":[{"delta":{"content":"ONLINE"}}]}\n',
            ],
        ),
        (
            "application/json",
            [
                b'data: {"choices":[{"delta":{"content":"ONLINE"}}]}\n',
                b"data: [DONE]\n",
            ],
        ),
    ],
)
def test_online_request_rejects_empty_malformed_or_non_sse_2xx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    lines: list[bytes],
) -> None:
    args = SimpleNamespace(
        online_rate=1.0,
        online_duration_seconds=1.0,
        model="model",
        online_max_tokens=8,
        online_base_url="http://online.test",
        request_timeout_seconds=5.0,
        tenant="tenant",
    )
    runner = harness.OnlineLoadRunner(args, tmp_path)
    monkeypatch.setattr(
        harness.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeOnlineResponse(content_type, lines),
    )
    started = harness.time.monotonic()

    runner._request(0, started, started)

    assert runner.results[0]["ok"] is False
    assert runner.results[0]["error_type"] == "invalid_sse_response"


def test_online_request_accepts_complete_sse_with_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        online_rate=1.0,
        online_duration_seconds=1.0,
        model="model",
        online_max_tokens=8,
        online_base_url="http://online.test",
        request_timeout_seconds=5.0,
        tenant="tenant",
    )
    runner = harness.OnlineLoadRunner(args, tmp_path)
    response = _FakeOnlineResponse(
        "text/event-stream; charset=utf-8",
        [
            b'data: {"choices":[{"delta":{"content":"ONLINE"}}]}\n',
            b"data: [DONE]\n",
        ],
    )
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *_args, **_kwargs: response
    )
    started = harness.time.monotonic()

    runner._request(0, started, started)

    assert runner.results[0]["ok"] is True
    assert runner.results[0]["parsed_event_count"] == 1
    assert runner.results[0]["content_seen"] is True
    assert runner.results[0]["done_seen"] is True


class _FakeMetricsResponse:
    status = 200

    def __enter__(self) -> _FakeMetricsResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b"metric_total 1\n"


def test_metrics_sampler_counts_only_success_and_surfaces_endpoint_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request, *, timeout: float):  # type: ignore[no-untyped-def]
        del timeout
        if request.full_url.endswith("/failed"):
            raise harness.urllib.error.URLError("endpoint unavailable")
        return _FakeMetricsResponse()

    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    sampler = harness.MetricsSampler(
        tmp_path,
        [
            ("healthy", "http://metrics.test/healthy"),
            ("failed", "http://metrics.test/failed"),
        ],
        60.0,
    )
    sampler._snapshot()
    completed_thread = threading.Thread(target=lambda: None)
    completed_thread.start()
    completed_thread.join()
    sampler.thread = completed_thread

    summary = sampler.stop()

    assert summary["samples"] == 1
    assert summary["failed_samples"] == 1
    assert summary["error"] is not None
    assert summary["endpoints"] == [
        {
            "name": "healthy",
            "url": "http://metrics.test/healthy",
            "successful_samples": 1,
            "failed_samples": 0,
        },
        {
            "name": "failed",
            "url": "http://metrics.test/failed",
            "successful_samples": 0,
            "failed_samples": 1,
        },
    ]


def test_metrics_sampler_all_endpoint_failure_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise harness.urllib.error.URLError("all endpoints unavailable")

    monkeypatch.setattr(harness.urllib.request, "urlopen", fail_urlopen)
    sampler = harness.MetricsSampler(
        tmp_path,
        [("failed", "http://metrics.test/failed")],
        60.0,
    )
    sampler._snapshot()
    completed_thread = threading.Thread(target=lambda: None)
    completed_thread.start()
    completed_thread.join()
    sampler.thread = completed_thread

    summary = sampler.stop()

    assert summary["samples"] == 0
    assert summary["failed_samples"] == 1
    assert "configured metric sample" in summary["error"]


def test_metrics_sampler_zero_collection_is_an_error(tmp_path: Path) -> None:
    sampler = harness.MetricsSampler(
        tmp_path,
        [("never-sampled", "http://metrics.test/never")],
        60.0,
    )
    completed_thread = threading.Thread(target=lambda: None)
    completed_thread.start()
    completed_thread.join()
    sampler.thread = completed_thread

    summary = sampler.stop()

    assert summary["samples"] == 0
    assert summary["failed_samples"] == 0
    assert "no successful samples" in summary["error"]


def test_create_batch_declares_trusted_request_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = harness.BatchClient("http://gateway.test", "tenant", 5.0)
    captured: dict[str, object] = {}

    def fake_json_request(
        method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        captured.update(method=method, path=path, payload=payload)
        return {"id": "batch-1"}

    monkeypatch.setattr(client, "json_request", fake_json_request)

    assert client.create_batch("file-1", "24h", 100) == {"id": "batch-1"}
    assert captured == {
        "method": "POST",
        "path": "/v1/batches",
        "payload": {
            "input_file_id": "file-1",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"planner_request_count": "100"},
        },
    }


@pytest.mark.parametrize("request_count", [0, -1, True, 1.5, "1"])
def test_create_batch_rejects_invalid_request_count(request_count: object) -> None:
    client = harness.BatchClient("http://gateway.test", "tenant", 5.0)

    with pytest.raises(harness.HarnessError, match="positive integer"):
        client.create_batch("file-1", "24h", request_count)  # type: ignore[arg-type]


def test_terminal_validation_rejects_duplicate_result_ids(tmp_path: Path) -> None:
    class FakeBatchClient:
        def download_file(self, _file_id: str) -> bytes:
            return (
                json.dumps({"custom_id": "duplicate"})
                + "\n"
                + json.dumps({"custom_id": "duplicate"})
                + "\n"
            ).encode()

    terminal = {
        "status": "completed",
        "request_counts": {"total": 2, "completed": 2, "failed": 0},
        "output_file_id": "output-id",
    }

    with pytest.raises(harness.HarnessError, match="duplicate custom_id"):
        harness.validate_terminal_results(
            tmp_path,
            terminal,
            {"duplicate", "expected-other"},
            FakeBatchClient(),  # type: ignore[arg-type]
            False,
        )


def test_terminal_validation_rejects_same_count_wrong_custom_id_set(
    tmp_path: Path,
) -> None:
    class FakeBatchClient:
        def download_file(self, _file_id: str) -> bytes:
            return (
                json.dumps({"custom_id": "submitted-1"})
                + "\n"
                + json.dumps({"custom_id": "wrong-2"})
                + "\n"
            ).encode()

    terminal = {
        "status": "completed",
        "request_counts": {"total": 2, "completed": 2, "failed": 0},
        "output_file_id": "output-id",
    }

    with pytest.raises(
        harness.HarnessError, match="missing_count=1.*unexpected_count=1"
    ):
        harness.validate_terminal_results(
            tmp_path,
            terminal,
            {"submitted-1", "submitted-2"},
            FakeBatchClient(),  # type: ignore[arg-type]
            False,
        )


def test_terminal_validation_sanitizes_downloaded_result_text(tmp_path: Path) -> None:
    fake_hf_token = "hf_" + ("z" * 24)
    bearer_secret = "downloaded-bearer-secret"
    redis_password = "downloaded-redis-password"

    class FakeBatchClient:
        def download_file(self, _file_id: str) -> bytes:
            return (
                json.dumps(
                    {
                        "custom_id": "submitted-1",
                        "response": {
                            "body": {
                                "message": (
                                    f"HF_TOKEN={fake_hf_token} "
                                    f"Bearer {bearer_secret} "
                                    "redis://redis-user:"
                                    f"{redis_password}@redis.test:6379/0?token=secret"
                                )
                            }
                        },
                    }
                )
                + "\n"
            ).encode()

    terminal = {
        "status": "completed",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "output_file_id": "output-id",
    }

    result = harness.validate_terminal_results(
        tmp_path,
        terminal,
        {"submitted-1"},
        FakeBatchClient(),  # type: ignore[arg-type]
        False,
    )

    assert result["valid"] is True
    rendered = (tmp_path / "batch-output.jsonl").read_text()
    assert json.loads(rendered)["custom_id"] == "submitted-1"
    assert fake_hf_token not in rendered
    assert bearer_secret not in rendered
    assert redis_password not in rendered
    assert "redis-user" not in rendered
    assert "redis://redis.test:6379/0?<redacted>" in rendered


def test_terminal_validation_rejects_non_utf8_without_persisting(
    tmp_path: Path,
) -> None:
    class FakeBatchClient:
        def download_file(self, _file_id: str) -> bytes:
            return b"\xff\xfe"

    terminal = {
        "status": "completed",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "output_file_id": "output-id",
    }

    with pytest.raises(harness.HarnessError, match="not valid UTF-8"):
        harness.validate_terminal_results(
            tmp_path,
            terminal,
            {"submitted-1"},
            FakeBatchClient(),  # type: ignore[arg-type]
            False,
        )

    assert not (tmp_path / "batch-output.jsonl").exists()


def test_shell_wrapper_preserves_harness_failure_exit(tmp_path: Path) -> None:
    workloads_dir = Path(harness.__file__).resolve().parent
    completed = subprocess.run(
        [
            str(workloads_dir / "run_baseline.sh"),
            "--preflight-only",
            "--skip-cluster-preflight",
            "--skip-api-preflight",
            "--dataset",
            str(tmp_path / "missing.jsonl"),
            "--experiment-root",
            str(tmp_path / "experiment"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 1
    assert "does not exist" in completed.stderr
