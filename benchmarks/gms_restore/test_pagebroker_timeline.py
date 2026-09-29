# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for aggregate timing and evidence isolation."""

import json

import pytest
from pagebroker_timeline import parse_case, write_html

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def case(tmp_path):
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    save(
        "timing.json",
        {
            "create_epoch": 1000,
            "ready_epoch": 1010,
            "capture_id": "capture",
            "resident_gms_generation": "current",
            "resident_gms_ready_verified_epoch": 999,
        },
    )
    save(
        "plan.json",
        {
            "capture_id": "capture",
            "ranks": [
                {
                    "rank": r,
                    "destination_uuid": f"GPU-{r}",
                    "manifest_sha256": f"manifest-{r}",
                    "allocations": [
                        {
                            "allocation_id": f"allocation-{r}",
                            "aligned_size": size,
                            "shard": "shards/shared-name.bin",
                            "offset": 0,
                        }
                    ],
                }
                for r, size in [(0, 100), (1, 200)]
            ],
        },
    )
    save(
        "resident-gms-before.json",
        {
            "metadata": {
                "name": "independent-gms",
                "ownerReferences": [
                    {"kind": "DaemonSet", "name": "gms", "uid": "owner"}
                ],
            }
        },
    )
    save("operator-created-pod.json", {"spec": {"containers": [{"name": "main"}]}})
    save(
        "resident-gms-ready.json",
        {
            "generation": "current",
            "online": [
                {
                    "rank": r,
                    "uuid": f"GPU-{r}",
                    "generation": "current",
                    "daemon_started_epoch": 997,
                    "online_epoch": 998,
                    "weights_server_nonce": "nonce",
                    "payload_bytes_read": 0,
                    "weight_allocations": 0,
                }
                for r in (0, 1)
            ],
        },
    )
    reports, events = [], []
    for rank, size, first, last in [(0, 100, 1001, 1003), (1, 200, 1002, 1005)]:
        identity = {
            "capture_id": "capture",
            "generation": "current",
            "rank": rank,
            "destination_uuid": f"GPU-{rank}",
        }
        phases = [
            ("lease_start", 1000.1, {}),
            ("allocation_import_complete", 1000.2, {}),
            ("ring_queue_start", first - 0.25, {}),
            ("ring_acquired", first, {"queue_seconds": 0.25}),
            ("read_copy_start", first, {"o_direct": True}),
            ("read_copy_complete", last, {}),
            ("copies_drained", last, {}),
            ("committed", last + 0.1, {}),
        ]
        for phase, at, extra in phases:
            value = {
                **identity,
                "event": "gms_load_phase",
                "phase": phase,
                "epoch_ns": int(at * 1e9),
                **extra,
            }
            if phase in {
                "ring_queue_start",
                "ring_acquired",
                "read_copy_start",
                "read_copy_complete",
            }:
                value.update(shard="shards/shared-name.bin", bytes=size)
            events.append(value)
        reports.append(
            {
                "report": json.dumps(
                    {
                        **identity,
                        "event": "gms_weights_loaded",
                        "manifest_sha256": f"manifest-{rank}",
                        "server_nonce": "nonce",
                        "committed_epoch_ns": int((last + 0.1) * 1e9),
                    }
                )
            }
        )
    save("publications.json", reports)
    save("inference.json", {"text": "coherent answer"})
    # Same file from two ranks means two distinct payloads. An earlier resident
    # generation and duplicate kubectl log lines must not inflate transferred bytes.
    foreign = dict(events[3], generation="previous", epoch_ns=1001000000000)
    (tmp_path / "pagebroker.txt").write_text(
        "\n".join(
            "kubectl-prefix " + json.dumps(e) for e in [foreign, *events, events[-2]]
        )
    )
    return tmp_path


def test_aggregate_uses_earliest_read_and_last_copy(case):
    result = parse_case(case)
    assert result["metrics"]["payload_window_s"] == 4
    assert result["metrics"]["completed_bytes"] == 300
    assert result["metrics"]["payload_effective_GB_per_s"] == 75 / 1e9
    assert result["metrics"]["gms_queue_seconds_sum"] == 0.5
    assert result["validation"]["unique_rank_shard_payloads"] == 2
    for key in (
        "all_payloads_o_direct",
        "payload_set_matches_plan",
        "completed_bytes_match_plan",
        "completed_payloads_match_starts",
        "native_rank_uuid_matches_plan",
        "terminal_report_provenance_matches_plan",
    ):
        assert result["validation"][key], key
    assert result["ownership"]["gms_owned_by_daemonset"]
    assert result["ownership"]["all_rank_servers_online_before_request"]
    assert all(s["end"] <= 0 for r in result["prewarm_rows"] for s in r["segments"])


def test_missing_rank_completion_is_not_complete(case):
    lines = (case / "pagebroker.txt").read_text().splitlines()
    (case / "pagebroker.txt").write_text(
        "\n".join(
            line
            for line in lines
            if not (
                '"rank": 1' in line
                and (
                    '"phase": "read_copy_complete"' in line
                    or '"phase": "committed"' in line
                )
            )
        )
    )
    result = parse_case(case)
    assert not result["validation"]["completed_bytes_match_plan"]
    assert not result["validation"]["completed_payloads_match_starts"]
    assert not result["validation"]["committed_rank_set_matches_plan"]


def test_html_escapes_recorded_strings(case):
    result = parse_case(case)
    result["case"] = "</script><script>alert(1)</script>"
    output = case / "chart.html"
    write_html(output, [result])
    text = output.read_text()
    assert "</script><script>alert" not in text
    assert "\\u003c/script>" in text


def test_artifact_discovery_follows_dgd_creation(case):
    timing = json.loads((case / "timing.json").read_text())
    timing["dgd_create_return_epoch"] = 1000.3
    (case / "timing.json").write_text(json.dumps(timing))
    (case / "coordinator.txt").write_text(
        "\n".join(
            json.dumps({"event": event, "epoch": at, "generation": "current"})
            for event, at in [
                ("manifest_discovery_start", 1000.4),
                ("manifest_discovery_complete", 1000.6),
                ("pagebroker_request_start", 1000.7),
            ]
        )
    )
    result = parse_case(case)
    assert result["validation"]["artifact_discovery_after_dgd_created"]
    assert result["validation"]["pagebroker_dispatch_after_manifest_discovery"]
    assert result["metrics"]["manifest_discovery_s"] == pytest.approx(0.2)
    timing["dgd_create_return_epoch"] = 1000.5
    (case / "timing.json").write_text(json.dumps(timing))
    assert not parse_case(case)["validation"]["artifact_discovery_after_dgd_created"]


def test_generic_service_identity_is_separate_from_dgd_load(case):
    timing = json.loads((case / "timing.json").read_text())
    timing["resident_gms_service_generation"] = "prestarted-service"
    timing["dgd_create_return_epoch"] = 1000.3
    (case / "timing.json").write_text(json.dumps(timing))
    (case / "dgd-created.json").write_text(json.dumps({"metadata": {"uid": "current"}}))
    (case / "snapshot-selection.json").write_text(
        json.dumps(
            {
                "dgd_uid": "current",
                "discovery_started_epoch": 1000.4,
                "discovery_completed_epoch": 1000.5,
            }
        )
    )
    ready = json.loads((case / "resident-gms-ready.json").read_text())
    ready.pop("generation")
    ready["service_generation"] = "prestarted-service"
    for record in ready["online"]:
        record.pop("generation")
        record.update(
            service_generation="prestarted-service",
            cuda_context_current=False,
            cuda_primary_context_active=False,
            loader_lanes_ready=0,
            pinned_bytes=0,
        )
    (case / "resident-gms-ready.json").write_text(json.dumps(ready))
    result = parse_case(case)
    assert result["generation"] == "current"
    assert result["service_generation"] == "prestarted-service"
    for key in (
        "online_service_generation_matches",
        "online_rank_uuids_match_plan",
        "no_capture_selected_at_service_ready",
        "all_rank_servers_pristine_before_request",
        "no_loader_context_or_ring_before_request",
    ):
        assert result["ownership"][key], key
    assert result["validation"]["load_generation_matches_created_dgd_uid"]
    assert result["validation"]["snapshot_selection_matches_created_dgd_uid"]
    assert result["validation"]["snapshot_api_selection_after_dgd_created"]
    ready["online"][0]["capture_id"] = "selected-too-early"
    (case / "resident-gms-ready.json").write_text(json.dumps(ready))
    assert not parse_case(case)["ownership"]["no_capture_selected_at_service_ready"]
