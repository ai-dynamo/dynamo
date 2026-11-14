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

import contextlib
import http.client
import io
import json
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import baseline_harness as harness
import pytest

UTC = timezone.utc

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


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--timeout-seconds", "nan"),
        ("--abort-cleanup-timeout-seconds", "nan"),
        ("--poll-interval-seconds", "inf"),
        ("--online-rate", "-inf"),
        ("--online-duration-seconds", "inf"),
        ("--temperature", "nan"),
    ],
)
def test_cli_rejects_nonfinite_float_arguments(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        harness.parse_args([option, value])


def test_online_schedule_has_explicit_total_request_cap(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    args = harness.parse_args(
        [
            "--dataset",
            str(dataset),
            "--online-rate",
            "100",
            "--online-duration-seconds",
            "11",
            "--online-max-requests",
            "1000",
        ]
    )

    with pytest.raises(harness.HarnessError, match="online-max-requests=1000"):
        harness.validate_args(args)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--batch-size", "10001"], "batch-size cannot exceed"),
        (
            ["--start-index", "99999", "--batch-size", "2"],
            "bounded scan limit",
        ),
        (
            ["--timeout-seconds", "100001", "--poll-interval-seconds", "1"],
            "progress records",
        ),
        (
            [
                "--metrics-url",
                "frontend=http://metrics.test",
                "--timeout-seconds",
                "10001",
                "--metrics-interval-seconds",
                "1",
            ],
            "snapshots per endpoint",
        ),
        (
            [
                "--metrics-url",
                "frontend=http://metrics.test",
                "--timeout-seconds",
                "1",
                "--metrics-interval-seconds",
                "1",
                "--online-rate",
                "0.001",
                "--online-duration-seconds",
                "100000000",
            ],
            "snapshots per endpoint",
        ),
        (["--online-max-requests", "100001"], "cannot exceed"),
        (["--online-max-inflight", "257"], "cannot exceed"),
    ],
)
def test_validate_args_bounds_evidence_growth(
    tmp_path: Path, options: list[str], message: str
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    args = harness.parse_args(["--dataset", str(dataset), *options])

    with pytest.raises(harness.HarnessError, match=message):
        harness.validate_args(args)


def test_normalize_workload_rejects_oversized_source_record(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("x" * (harness.MAX_DATASET_RECORD_BYTES + 1), encoding="utf-8")

    with pytest.raises(harness.HarnessError, match="record limit"):
        harness.normalize_workload(
            source,
            tmp_path / "batch.jsonl",
            tmp_path / "manifest.json",
            batch_size=1,
            start_index=0,
            model="model",
            max_tokens=1,
            temperature=0.0,
        )


def test_normalize_workload_rejects_oversized_custom_id(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_dataset(
        source,
        [
            {
                "custom_id": "x" * (harness.MAX_CUSTOM_ID_BYTES + 1),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"messages": [{"role": "user", "content": "hi"}]},
            }
        ],
    )

    with pytest.raises(harness.HarnessError, match="custom_id exceeds"):
        harness.normalize_workload(
            source,
            tmp_path / "batch.jsonl",
            tmp_path / "manifest.json",
            batch_size=1,
            start_index=0,
            model="model",
            max_tokens=1,
            temperature=0.0,
        )


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


def test_redaction_handles_long_yaml_env_block_linearly() -> None:
    secret = "yaml-secret-that-is-not-token-shaped"
    value = "".join(
        [
            "env:\n",
            "  - name: HF_TOKEN\n",
            *(f"    extension-{index}: value\n" for index in range(20_000)),
            f"    value: {secret}\n",
            "  - name: SAFE_VALUE\n",
            "    value: visible\n",
        ]
    )

    redacted = harness.redact_text(value)

    assert secret not in redacted
    assert "    value: <redacted>\n" in redacted
    assert "    value: visible\n" in redacted


def test_redaction_handles_long_non_url_scheme_text_linearly() -> None:
    value = "x" * 100_000

    assert harness.redact_text(value) == value


def test_structured_redaction_covers_generic_secret_names(tmp_path: Path) -> None:
    sentinel_values = ["key-material", "token-material", "password-material"]
    output = tmp_path / "evidence.json"

    harness.write_json(
        output,
        {
            "api_key": sentinel_values[0],
            "nested": {"access_token": sentinel_values[1]},
            "env": [
                {"name": "DATABASE_PASSWORD", "value": sentinel_values[2]},
                {"name": "SAFE_VALUE", "value": "visible"},
            ],
            "completion_tokens": 17,
        },
    )

    raw = output.read_text()
    assert all(sentinel not in raw for sentinel in sentinel_values)
    parsed = json.loads(raw)
    assert parsed["api_key"] == "<redacted>"
    assert parsed["nested"]["access_token"] == "<redacted>"
    assert parsed["env"][0]["value"] == "<redacted>"
    assert parsed["env"][1]["value"] == "visible"
    assert parsed["completion_tokens"] == 17


def test_write_json_redacts_credential_headers_at_the_storage_boundary(
    tmp_path: Path,
) -> None:
    header_fixtures = {
        "Authorization": "Bearer authorization-secret",
        "X-API-Key": "api-key-secret",
        "X-HF-Token": "hf_header-secret-value",
    }
    output = tmp_path / "request-metadata.json"

    harness.write_json(
        output,
        {
            "request": {
                "headers": {
                    **header_fixtures,
                    "Content-Type": "application/json",
                }
            }
        },
    )

    raw = output.read_text(encoding="utf-8")
    assert all(fixture not in raw for fixture in header_fixtures.values())
    parsed = json.loads(raw)
    assert parsed["request"]["headers"] == {
        "Authorization": "<redacted>",
        "Content-Type": "application/json",
        "X-API-Key": "<redacted>",
        "X-HF-Token": "<redacted>",
    }


def test_write_json_redacts_mapping_keys_and_round_trips_redacted_strings(
    tmp_path: Path,
) -> None:
    hf_credential = "hf_" + ("k" * 24)
    redaction_sentinels = {
        "assignment-secret",
        "quoted secret",
        "bearer-secret",
        "url-user",
        "url-password",
        "query-secret",
    }
    output = tmp_path / "redacted-strings.json"

    harness.write_json(
        output,
        {
            f"evidence-{hf_credential}": "safe value",
            "strings": {
                "assignment": "HF_TOKEN=assignment-secret",
                "quoted_assignment": 'API_KEY="quoted secret"',
                "bearer": "Bearer bearer-secret",
                "url": (
                    "https://url-user:url-password@example.test/path"
                    "?token=query-secret"
                ),
            },
        },
    )

    raw = output.read_text(encoding="utf-8")
    assert hf_credential not in raw
    assert all(sentinel not in raw for sentinel in redaction_sentinels)
    parsed = json.loads(raw)
    assert parsed["evidence-<redacted-hugging-face-credential>"] == "safe value"
    assert parsed["strings"] == {
        "assignment": "HF_TOKEN=<redacted>",
        "quoted_assignment": 'API_KEY="<redacted>"',
        "bearer": "Bearer <redacted>",
        "url": "https://example.test/path?<redacted>",
    }


def test_generic_assignment_redaction_is_line_bounded() -> None:
    secret = "generic-secret-value"
    value = f"API_KEY={secret}\n" + ("x" * 100_000)

    redacted = harness.redact_text(value)

    assert secret not in redacted
    assert redacted.startswith("API_KEY=<redacted>\n")


def test_raw_redaction_covers_prefixed_env_yaml_json_and_logs() -> None:
    secrets = ["openai-secret", "database-secret", "client-secret", "yaml-secret"]
    value = "\n".join(
        [
            f"OPENAI_API_KEY={secrets[0]}",
            f"DATABASE_PASSWORD: {secrets[1]}",
            f'{{"name":"MY_CLIENT_SECRET","value":"{secrets[2]}"}}',
            "  - name: SERVICE_ACCESS_TOKEN",
            f"    value: {secrets[3]}",
        ]
    )

    redacted = harness.redact_text(value)

    assert all(secret not in redacted for secret in secrets)
    assert redacted.count("<redacted>") >= 4


def test_capture_command_bounds_retained_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness, "MAX_CAPTURE_STREAM_BYTES", 128)

    result = harness.capture_command(
        tmp_path,
        "large-output",
        [sys.executable, "-c", "print('x' * 10000)"],
    )

    assert result.exit_code == 0
    assert harness.TRUNCATION_MARKER.strip() in result.stdout
    assert (tmp_path / "large-output.stdout").stat().st_size < 256


def test_tee_bounds_artifact_while_preserving_console() -> None:
    console = io.StringIO()
    artifact = io.StringIO()
    tee = harness.TeeTextIO(console, artifact, artifact_byte_limit=16)

    tee.write("visible-" + ("x" * 100))

    assert console.getvalue() == "visible-" + ("x" * 100)
    assert harness.TRUNCATION_MARKER.strip() in artifact.getvalue()
    assert len(artifact.getvalue()) < 80


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


@pytest.mark.parametrize("runtime", ["docker", "containerd", "cri-o"])
def test_redact_text_preserves_local_runtime_image_ids(runtime: str) -> None:
    digest = "0123456789abcdef" * 4
    image_id = f"{runtime}://sha256:{digest}"

    assert harness.redact_text(image_id) == image_id


@pytest.mark.parametrize(
    "value",
    [
        f"docker://sha256:{'a' * 64}?token=secret-query",
        f"containerd://sha256:{'b' * 64}#secret-fragment",
        f"cri-o://sha256:{'c' * 63}",
        f"podman://sha256:{'d' * 64}",
    ],
)
def test_redact_text_does_not_allowlist_runtime_image_id_lookalikes(
    value: str,
) -> None:
    assert harness.redact_text(value) == "<redacted-url>"


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


def test_redact_text_sanitizes_url_after_nonletter_scheme_characters() -> None:
    value = "9+.-https://url-user:url-pass@example.test/private?token=secret"

    assert harness.redact_text(value) == "9+.-https://example.test/private?<redacted>"


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
        run_kind="baseline",
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
        run_kind="baseline",
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
        run_kind="baseline",
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
        run_kind="baseline",
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
        run_kind="baseline",
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
        run_kind="baseline",
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

    pod["spec"]["containers"][0]["args"][
        1
    ] = '{"queues":[{"worker_pool_id":"dynamo-batch","queue_name":"batch"}]}'
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
        run_kind="baseline",
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
        run_kind="baseline",
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
        run_kind="baseline",
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


class _TricklingResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self) -> None:
        self.closed = False

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def __iter__(self):  # type: ignore[no-untyped-def]
        while not self.closed:
            harness.time.sleep(0.005)
            yield b": heartbeat\n"

    def read1(self, _size: int) -> bytes:
        harness.time.sleep(0.005)
        return b"" if self.closed else b"x"


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

    assert runner.last_result is not None
    assert runner.last_result["ok"] is False
    assert runner.last_result["error_type"] == "invalid_sse_response"


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

    assert runner.last_result is not None
    assert runner.last_result["ok"] is True
    assert runner.last_result["parsed_event_count"] == 1
    assert runner.last_result["content_seen"] is True
    assert runner.last_result["done_seen"] is True


def test_online_request_absolute_deadline_stops_sse_heartbeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        online_rate=1.0,
        online_duration_seconds=1.0,
        model="model",
        online_max_tokens=8,
        online_base_url="http://online.test",
        request_timeout_seconds=0.03,
        tenant="tenant",
    )
    runner = harness.OnlineLoadRunner(args, tmp_path)
    response = _TricklingResponse()
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *_args, **_kwargs: response
    )
    started = harness.time.monotonic()

    runner._request(0, started, started)

    assert runner.last_result is not None
    assert runner.last_result["ok"] is False
    assert runner.last_result["error_type"] == "TimeoutError"
    assert response.closed


def test_online_request_deadline_stops_no_newline_byte_trickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        online_rate=1.0,
        online_duration_seconds=1.0,
        model="model",
        online_max_tokens=8,
        online_base_url="http://online.test",
        request_timeout_seconds=0.05,
        tenant="tenant",
    )
    runner = harness.OnlineLoadRunner(args, tmp_path)
    client_socket, server_socket = socket.socketpair()
    server_socket.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
    response = http.client.HTTPResponse(client_socket)
    response.begin()
    producer_stopped = threading.Event()

    def trickle_without_newline() -> None:
        while not producer_stopped.wait(0.005):
            try:
                server_socket.sendall(b":")
            except OSError:
                return

    producer = threading.Thread(target=trickle_without_newline, daemon=True)
    producer.start()
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *_args, **_kwargs: response
    )
    started = harness.time.monotonic()
    try:
        runner._request(0, started, started)
        assert harness.time.monotonic() - started < 0.3
        assert runner.last_result is not None
        assert runner.last_result["ok"] is False
        assert runner.last_result["error_type"] == "TimeoutError"
    finally:
        producer_stopped.set()
        with contextlib.suppress(OSError):
            server_socket.shutdown(socket.SHUT_RDWR)
        server_socket.close()
        client_socket.close()
        producer.join(timeout=0.2)


def test_http_size_limit_does_not_block_on_reader_owned_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_socket, server_socket = socket.socketpair()
    server_socket.sendall(b"HTTP/1.1 200 OK\r\n\r\n123456789")
    response = http.client.HTTPResponse(client_socket)
    response.begin()
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *_args, **_kwargs: response
    )
    request = harness.urllib.request.Request("http://gateway.test", method="GET")
    started = harness.time.monotonic()
    try:
        with pytest.raises(harness.HarnessError, match="evidence limit"):
            with harness.open_url_with_deadline(request, timeout_seconds=1.0) as (
                opened,
                deadline,
                expired,
            ):
                harness.read_http_body(
                    opened,
                    deadline=deadline,
                    expired=expired,
                    max_bytes=8,
                )
        assert harness.time.monotonic() - started < 0.3
    finally:
        with contextlib.suppress(OSError):
            server_socket.shutdown(socket.SHUT_RDWR)
        server_socket.close()
        client_socket.close()


def test_json_http_client_absolute_deadline_stops_trickling_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _TricklingResponse()
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *_args, **_kwargs: response
    )
    client = harness.JsonHttpClient("http://gateway.test", "tenant", 0.03)

    with pytest.raises(harness.HarnessError, match="exceeded .* deadline"):
        client.request("GET", "/v1/batches")

    assert response.closed


def test_json_http_client_deadline_covers_open_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _TricklingResponse()
    release_headers = threading.Event()
    opener_started = threading.Event()

    def trickle_headers(*_args: object, **_kwargs: object) -> _TricklingResponse:
        opener_started.set()
        release_headers.wait()
        return response

    monkeypatch.setattr(harness.urllib.request, "urlopen", trickle_headers)
    client = harness.JsonHttpClient("http://gateway.test", "tenant", 0.03)
    started = harness.time.monotonic()
    try:
        with pytest.raises(harness.HarnessError, match="exceeded .* deadline"):
            client.request("GET", "/v1/batches")
        assert opener_started.is_set()
        assert harness.time.monotonic() - started < 0.25
    finally:
        release_headers.set()

    deadline = harness.time.monotonic() + 0.25
    while not response.closed and harness.time.monotonic() < deadline:
        harness.time.sleep(0.005)
    assert response.closed


class _FakeMetricsResponse:
    status = 200

    def __init__(self) -> None:
        self._read = False

    def __enter__(self) -> _FakeMetricsResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
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


def test_metrics_sampler_stop_does_not_count_inflight_cancellation_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second_request_started = threading.Event()
    release_second_request = threading.Event()
    request_count = 0

    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeMetricsResponse:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return _FakeMetricsResponse()
        second_request_started.set()
        release_second_request.wait(timeout=1.0)
        return _FakeMetricsResponse()

    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    sampler = harness.MetricsSampler(
        tmp_path,
        [("healthy", "http://metrics.test/healthy")],
        0.01,
    )
    sampler.start()
    assert second_request_started.wait(timeout=1.0)

    try:
        summary = sampler.stop()
    finally:
        release_second_request.set()

    assert summary["samples"] == 1
    assert summary["failed_samples"] == 0
    assert summary["error"] is None
    assert summary["endpoints"] == [
        {
            "name": "healthy",
            "url": "http://metrics.test/healthy",
            "successful_samples": 1,
            "failed_samples": 0,
        }
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


def test_cancel_batch_uses_supported_idempotent_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = harness.BatchClient("http://gateway.test", "tenant", 5.0)
    captured: dict[str, object] = {}

    def fake_json_request(
        method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        captured.update(method=method, path=path, payload=payload)
        return {"id": "batch-1", "status": "cancelling"}

    monkeypatch.setattr(client, "json_request", fake_json_request)

    assert client.cancel_batch("batch-1")["status"] == "cancelling"
    assert captured == {
        "method": "POST",
        "path": "/v1/batches/batch-1/cancel",
        "payload": None,
    }


def test_cancel_batch_to_terminal_records_cleanup_evidence(tmp_path: Path) -> None:
    class FakeBatchClient:
        def __init__(self) -> None:
            self.get_calls = 0
            self.cancel_calls = 0

        def get_batch(self, _batch_id: str) -> dict[str, object]:
            self.get_calls += 1
            status = "in_progress" if self.get_calls == 1 else "cancelled"
            return {
                "id": "batch-1",
                "status": status,
                "request_counts": {"total": 10, "completed": 2, "failed": 0},
            }

        def cancel_batch(self, _batch_id: str) -> dict[str, object]:
            self.cancel_calls += 1
            return {
                "id": "batch-1",
                "status": "cancelling",
                "request_counts": {"total": 10, "completed": 2, "failed": 0},
            }

    client = FakeBatchClient()
    artifact_dir = tmp_path / "abort-cleanup"

    summary = harness.cancel_batch_to_terminal(
        client,  # type: ignore[arg-type]
        "batch-1",
        artifact_dir,
        poll_interval_seconds=0.001,
        timeout_seconds=1.0,
        source="test",
    )

    assert client.cancel_calls == 1
    assert summary["succeeded"] is True
    assert summary["initial_status"] == "in_progress"
    assert summary["terminal_status"] == "cancelled"
    assert (artifact_dir / "cancel-response.json").is_file()
    persisted = json.loads((artifact_dir / "cancel-summary.json").read_text())
    assert persisted["terminal_status"] == "cancelled"
    progress = [
        json.loads(line)
        for line in (artifact_dir / "cancel-progress.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in progress] == [
        "in_progress",
        "cancelling",
        "cancelled",
    ]


def test_cancel_batch_to_terminal_does_not_recancel_terminal_job(
    tmp_path: Path,
) -> None:
    class FakeBatchClient:
        def get_batch(self, _batch_id: str) -> dict[str, object]:
            return {
                "id": "batch-1",
                "status": "completed",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            }

        def cancel_batch(self, _batch_id: str) -> dict[str, object]:
            raise AssertionError("terminal jobs must not receive a cancel request")

    summary = harness.cancel_batch_to_terminal(
        FakeBatchClient(),  # type: ignore[arg-type]
        "batch-1",
        tmp_path,
        poll_interval_seconds=0.001,
        timeout_seconds=1.0,
        source="test",
    )

    assert summary["succeeded"] is True
    assert summary["cancel_requested"] is False
    assert summary["terminal_status"] == "completed"


def test_cancel_batch_to_terminal_records_failure_when_job_stays_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeBatchClient:
        def get_batch(self, _batch_id: str) -> dict[str, object]:
            return {
                "id": "batch-1",
                "status": "in_progress",
                "request_counts": {"total": 1, "completed": 0, "failed": 0},
            }

        def cancel_batch(self, _batch_id: str) -> dict[str, object]:
            return {
                "id": "batch-1",
                "status": "cancelling",
                "request_counts": {"total": 1, "completed": 0, "failed": 0},
            }

    clock = [0.0]
    monkeypatch.setattr(harness.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        harness.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    summary = harness.cancel_batch_to_terminal(
        FakeBatchClient(),  # type: ignore[arg-type]
        "batch-1",
        tmp_path,
        poll_interval_seconds=0.001,
        timeout_seconds=0.005,
        source="test",
    )

    assert summary["succeeded"] is False
    assert summary["cancel_requested"] is True
    assert summary["terminal_status"] is None
    assert "did not become terminal" in summary["error"]
    assert json.loads((tmp_path / "cancel-summary.json").read_text()) == summary


def test_execute_cancels_created_batch_when_created_artifact_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        metrics_url=[],
        dataset=dataset,
        batch_size=1,
        start_index=0,
        model="Qwen/Qwen3-0.6B",
        max_tokens=8,
        temperature=0.0,
        run_kind="baseline",
        skip_cluster_preflight=True,
        batch_base_url="http://gateway.test",
        tenant="tenant",
        request_timeout_seconds=1.0,
        skip_api_preflight=True,
        preflight_only=False,
        metrics_interval_seconds=1.0,
        completion_window="24h",
        poll_interval_seconds=0.001,
        timeout_seconds=10.0,
        allow_request_failures=False,
        abort_cleanup_timeout_seconds=1.0,
    )

    class FakeContext:
        def __init__(self) -> None:
            self.run_id = "test-run"
            self.run_dir = tmp_path
            self.started = datetime.now(UTC)
            self.metadata: dict[str, object] = {"inputs": {}, "batch": {}}
            self.notes: list[str] = []

        def set_batch(self, **fields: object) -> None:
            batch = self.metadata["batch"]
            assert isinstance(batch, dict)
            batch.update(fields)

        def add_note(self, note: str) -> None:
            self.notes.append(note)

        def write_metadata(self) -> None:
            return None

    class FakeBatchClient:
        def upload(self, _path: Path) -> dict[str, object]:
            return {"id": "file-1"}

        def create_batch(
            self, _file_id: str, _window: str, _request_count: int
        ) -> dict[str, object]:
            return {"id": "batch-created-before-write"}

    class FakeMetricsSampler:
        def start(self) -> None:
            return None

        def stop(self) -> dict[str, object]:
            return {"enabled": False, "samples": 0, "error": None}

    class FakeOnlineLoadRunner:
        stopped = False

        def start(self) -> None:
            raise AssertionError("online load must not start")

        def stop(self) -> None:
            self.stopped = True

        def finish(self) -> dict[str, object]:
            return {"enabled": False}

    client = FakeBatchClient()
    online = FakeOnlineLoadRunner()
    cleanup_calls: list[tuple[str, str]] = []
    original_write_json = harness.write_json

    def fail_created_artifact(path: Path, value: object) -> None:
        if path.name == "batch-created.json":
            raise OSError("simulated artifact write failure")
        original_write_json(path, value)

    def record_cleanup(
        _client: object,
        batch_id: str,
        _artifact_dir: Path,
        *,
        poll_interval_seconds: float,
        timeout_seconds: float,
        source: str,
    ) -> dict[str, object]:
        assert poll_interval_seconds == args.poll_interval_seconds
        assert timeout_seconds == args.abort_cleanup_timeout_seconds
        cleanup_calls.append((batch_id, source))
        return {
            "triggered": True,
            "source": source,
            "batch_id": batch_id,
            "succeeded": True,
            "terminal_status": "cancelled",
        }

    monkeypatch.setattr(harness, "validate_args", lambda _args: None)
    monkeypatch.setattr(harness, "local_tool_versions", lambda _run_dir: {})
    monkeypatch.setattr(
        harness,
        "normalize_workload",
        lambda *_args, **_kwargs: {
            "id_mapping": [{"submitted_custom_id": "request-0"}],
            "output_sha256": "sha256",
        },
    )
    monkeypatch.setattr(harness, "BatchClient", lambda *_args: client)
    monkeypatch.setattr(harness, "MetricsSampler", lambda *_args: FakeMetricsSampler())
    monkeypatch.setattr(harness, "OnlineLoadRunner", lambda *_args: online)
    monkeypatch.setattr(harness, "write_json", fail_created_artifact)
    monkeypatch.setattr(harness, "cancel_batch_to_terminal", record_cleanup)

    context = FakeContext()
    with pytest.raises(OSError, match="simulated artifact write failure"):
        harness.execute(args, context)  # type: ignore[arg-type]

    assert cleanup_calls == [
        ("batch-created-before-write", "baseline-harness-exception")
    ]
    assert online.stopped is True
    assert context.metadata["batch"] == {
        "abort_cleanup": {
            "triggered": True,
            "source": "baseline-harness-exception",
            "batch_id": "batch-created-before-write",
            "succeeded": True,
            "terminal_status": "cancelled",
        }
    }


def test_poll_batch_emits_dispatch_barrier_after_positive_progress_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batches = iter(
        [
            {
                "status": "validating",
                "metadata": {"planner_request_count": "1"},
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            },
            {
                "status": "in_progress",
                "metadata": {"planner_request_count": "1"},
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            },
            {
                "status": "in_progress",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            },
            {
                "status": "completed",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            },
        ]
    )

    class FakeBatchClient:
        def get_batch(self, _batch_id: str) -> dict[str, object]:
            return next(batches)

    monkeypatch.setattr(harness.time, "sleep", lambda _seconds: None)

    terminal = harness.poll_batch(
        FakeBatchClient(),  # type: ignore[arg-type]
        "batch-1",
        tmp_path / "progress.jsonl",
        expected_total=1,
        poll_interval_seconds=1.0,
        timeout_seconds=5.0,
    )

    emitted = [
        json.loads(line[len(harness.HARNESS_EVENT_PREFIX) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(harness.HARNESS_EVENT_PREFIX)
    ]
    assert terminal["status"] == "completed"
    assert [event["event"] for event in emitted] == [
        "batch-active",
        "batch-dispatch-active",
        "batch-terminal",
    ]
    assert emitted[0]["status"] == "in_progress"
    assert emitted[0]["completed"] == 0
    assert emitted[1]["status"] == "in_progress"
    assert emitted[1]["completed"] == 1
    assert emitted[1]["failed"] == 0
    assert emitted[1]["total"] == 1


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
    api_key = "downloaded-generic-api-key"
    database_password = "downloaded-database-password"

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
                            },
                            "api_key": api_key,
                            "nested": {"database_password": database_password},
                            "env": {"name": "MY_CLIENT_SECRET", "value": "env-secret"},
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
    assert api_key not in rendered
    assert database_password not in rendered
    assert "env-secret" not in rendered
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
