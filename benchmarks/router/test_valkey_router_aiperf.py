# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.router import valkey_router_aiperf as harness  # noqa: E402
from benchmarks.router.valkey_aiperf import common as harness_common  # noqa: E402
from benchmarks.router.valkey_aiperf import loadgen as harness_loadgen  # noqa: E402
from benchmarks.router.valkey_aiperf import validation as harness_validation  # noqa: E402
from benchmarks.router.valkey_aiperf import valkey as harness_valkey  # noqa: E402
from benchmarks.router import valkey_frontend_scale as scale  # noqa: E402
from benchmarks.router.valkey_scale import common as scale_common  # noqa: E402


def test_split_harness_resolves_repository_root() -> None:
    assert harness_common.REPO == REPO
    assert harness_common.DEFAULT_AIPERF == REPO / "dynamo/bin/aiperf"


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        event_plane="nats",
        frontend_count=3,
        logical_mocker_workers=4,
        mocker_processes=4,
        mocker_data_parallel_size=8,
        concurrency=4096,
        frontend_cpus=None,
        mocker_cpus=None,
        valkey_cpus=None,
        aiperf_cpus=None,
        requests=10,
        valkey_gc_interval_ms=60_000,
        valkey_gc_inspection_budget=256,
        kill_valkey_primary=False,
    )


def _valid_result(arm: str, *, input_sha256: str = "a" * 64) -> dict:
    result = {
        "sample_index": 1,
        "run": 1,
        "arm": arm,
        "status": "ok",
        "request_plane": "tcp",
        "offered_load": {
            "mode": "closed_loop",
            "concurrency": 4096,
            "request_rate_rps": "inf",
        },
        "event_plane": "nats",
        "topology": {
            "frontend_processes": 3,
            "mocker_processes": 4,
            "logical_mocker_workers": 4,
            "data_parallel_ranks_per_worker": 8,
            "routing_ranks": 32,
            "configured_cpu_affinity": {
                "frontend": None,
                "mocker": None,
                "valkey": None,
                "aiperf": None,
            },
            "mocker_process_layout": [{}, {}, {}, {}],
            "discovered_worker_ids": [1, 2, 3, 4],
            "discovered_worker_identity_count": 4,
        },
        "aiperf": {"timed_out": False, "returncode": 0},
        "aiperf_metrics": {
            "summary": {
                "request_count": {"avg": 10},
                "error_request_count": None,
                "request_throughput": {"avg": 123.0},
            },
            "records": {
                "completed_profiling_records": 10,
                "cancelled_profiling_records": 0,
                "errored_profiling_records": 0,
                "malformed_records": 0,
            },
        },
        "aiperf_input_sha256": input_sha256,
        "valkey_authoritative_admission": False,
        "valkey_gc_interval_ms": 60_000 if arm == "valkey_ha" else None,
        "valkey_gc_inspection_budget": 256 if arm == "valkey_ha" else None,
    }
    if arm == "valkey_ha":
        result["valkey_client_pressure"] = {
            "probe_inclusive": True,
            "sample_interval_seconds": 0.05,
            "sample_rounds": 2,
            "successful_reads": 4,
            "read_errors": 0,
            "ports": {
                "15001": {
                    "peak_connected_clients": 9,
                    "peak_blocked_clients": 2,
                    "maxclients": 10000,
                },
                "15002": {
                    "peak_connected_clients": 2,
                    "peak_blocked_clients": 0,
                    "maxclients": 10000,
                },
            },
        }
    return result


def _healthy_valkey_state(*, failed_calls: int = 0) -> dict:
    stats = f"calls=1,usec=1,usec_per_call=1.00,rejected_calls=0,failed_calls={failed_calls}"
    return {
        "primary_replication": {
            "role": "master",
            "connected_slaves": "1",
            "min_slaves_good_slaves": "1",
            "master_replid": "same",
            "slave0": "state=online,offset=1,lag=0",
        },
        "replica_replication": {
            "role": "slave",
            "master_link_status": "up",
            "master_sync_in_progress": "0",
            "master_replid": "same",
        },
        "primary_commandstats": {
            "dynkv.apply_owned": stats,
            "dynkv.register_worker_ranks": stats,
            "dynkv.select_reserve": stats,
            "dynkv.release": stats,
            "wait": stats,
        },
        "replica_commandstats": {
            "dynkv.apply_owned_at": stats,
            "dynkv.worker_lease_apply": stats,
            "dynkv.admit_apply": stats,
        },
    }


def _healthy_promoted_valkey_state(*, failed_calls: int = 0) -> dict:
    stats = f"calls=1,usec=1,usec_per_call=1.00,rejected_calls=0,failed_calls={failed_calls}"
    return {
        "replication": {"role": "master", "connected_slaves": "0"},
        "commandstats": {
            "dynkv.apply_owned": stats,
            "dynkv.select_reserve": stats,
            "dynkv.release": stats,
            "wait": stats,
        },
        "stats": [1, 32, 0],
        "lifecycle_stats": [4, 32, 0],
        "admission_stats": 0,
    }


def test_valkey_state_records_client_pressure(monkeypatch) -> None:
    responses = {
        (15001, "clients"): "connected_clients:403\r\nmaxclients:10000\r\n",
        (15002, "clients"): "connected_clients:2\r\nmaxclients:10000\r\n",
        (15001, "replication"): "role:master\r\n",
        (15002, "replication"): "role:slave\r\n",
        (15001, "commandstats"): "cmdstat_dynkv.stats:calls=1\r\n",
        (15002, "commandstats"): "cmdstat_dynkv.stats:calls=1\r\n",
    }
    monkeypatch.setattr(
        harness_valkey,
        "valkey_info",
        lambda port, section: responses[(port, section)],
    )
    monkeypatch.setattr(
        harness_valkey,
        "valkey_integer_array_command",
        lambda *args: [1, 2, 3],
    )
    monkeypatch.setattr(
        harness_valkey,
        "valkey_integer_command",
        lambda *args: 0,
    )

    state = harness_valkey.valkey_state(15001, 15002)

    assert state["primary_clients"] == {
        "connected_clients": "403",
        "maxclients": "10000",
    }
    assert state["replica_clients"] == {
        "connected_clients": "2",
        "maxclients": "10000",
    }
    singleton = harness_valkey.valkey_singleton_state(15002, "index")
    assert singleton["clients"] == state["replica_clients"]


def test_valkey_client_pressure_sampler_records_peak() -> None:
    stop = threading.Event()
    replies = iter(
        (
            "connected_clients:3\r\nblocked_clients:0\r\nmaxclients:10000\r\n",
            "connected_clients:9\r\nblocked_clients:2\r\nmaxclients:10000\r\n",
        )
    )

    def read_info(port: int, section: str) -> str:
        assert port == 15001
        assert section == "clients"
        reply = next(replies)
        if "connected_clients:9" in reply:
            stop.set()
        return reply

    pressure = harness_valkey.sample_valkey_client_pressure(
        stop,
        [15001],
        interval_seconds=0,
        info_reader=read_info,
    )

    assert pressure == {
        "probe_inclusive": True,
        "sample_interval_seconds": 0,
        "sample_rounds": 2,
        "successful_reads": 2,
        "read_errors": 0,
        "ports": {
            "15001": {
                "peak_connected_clients": 9,
                "peak_blocked_clients": 2,
                "maxclients": 10000,
            }
        },
    }


def test_three_arm_schedule_uses_all_permutations_then_cyclic() -> None:
    arms = ("inprocess", "inprocess_immediate", "valkey_ha")
    schedule = harness.build_arm_schedule(arms, 9)
    orders = [
        tuple(sample["arm"] for sample in schedule if sample["run"] == run)
        for run in range(1, 10)
    ]

    assert len(set(orders[:6])) == 6
    assert orders[6:] == [
        arms,
        ("inprocess_immediate", "valkey_ha", "inprocess"),
        ("valkey_ha", "inprocess", "inprocess_immediate"),
    ]
    positions = Counter((sample["arm"], sample["ordinal"]) for sample in schedule)
    assert set(positions.values()) == {3}


def test_matched_arm_selects_only_policy_equivalent_pair() -> None:
    args = SimpleNamespace(arm="matched", valkey_authoritative_admission=True)
    assert harness.arms_for_args(args) == ("inprocess_immediate", "valkey_ha")


def test_two_hundred_logical_workers_partition_across_processes() -> None:
    layout = harness_common.mocker_process_layout(200, 20, 8)

    assert len(layout) == 20
    assert {shard["logical_worker_count"] for shard in layout} == {10}
    assert layout[0]["logical_worker_start"] == 0
    assert layout[-1]["logical_worker_end_exclusive"] == 200
    assert layout[-1]["routing_target_ordinal_end_exclusive"] == 1600


def test_router_environment_scrub_covers_policy_and_worker_lease() -> None:
    required = {
        "DYN_ROUTER_LOAD_AWARE",
        "DYN_ROUTER_PREFILL_LOAD_SCALE",
        "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
        "DYN_OVERLAP_SCORE_WEIGHT",
        "DYN_ROUTER_SNAPSHOT_THRESHOLD",
        "DYN_ROUTER_RESET_STATES",
        "DYN_ROUTER_TTL_SECS",
        "DYN_ROUTER_EVENT_THREADS",
        "DYN_ROUTER_QUEUE_POLICY",
        "DYN_USE_REMOTE_INDEXER",
        "DYN_ROUTER_VALKEY_WORKER_LEASE_MS",
        "DYN_ROUTER_VALKEY_GC_INTERVAL_MS",
        "DYN_ROUTER_VALKEY_GC_INSPECTION_BUDGET",
    }
    assert required <= set(harness_common.ROUTER_ENV_TO_CLEAR)


def test_gc_cli_defaults_and_scale_forwarding_are_locked_down(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["valkey_router_aiperf.py"])
    args = harness.parse_args()

    assert args.valkey_gc_interval_ms == 60_000
    assert args.valkey_gc_inspection_budget == 256
    assert "--valkey-gc-interval-ms" in scale_common.PROTECTED_HARNESS_ARGUMENTS
    assert "--valkey-gc-inspection-budget" in scale_common.PROTECTED_HARNESS_ARGUMENTS


def test_valkey_index_key_is_component_scoped_and_percent_encoded() -> None:
    assert harness_common.valkey_index_key("tenant:a", "model/x", "scope%one", 16) == (
        "dynamo:kv-router:tenant%3Aa:component-model%2Fx:"
        "scope-scope%25one:block-size-16"
    )


def test_release_provenance_is_required_and_changes_are_rejected() -> None:
    release = {
        "dynamo_core": {
            "rust_build_profile": "release",
            "build_git_revision": "abc",
            "build_git_dirty": False,
        },
        "artifact": "a",
        "git": {"revision": "abc", "dirty": False},
    }
    assert harness.release_core_provenance_error(release) is None
    assert harness.provenance_change_errors(release, release, release) == []

    debug = {"dynamo_core": {"rust_build_profile": "debug"}, "artifact": "a"}
    assert "release" in harness.release_core_provenance_error(debug)
    changed = deepcopy(release)
    changed["artifact"] = "b"
    assert harness.provenance_change_errors(release, release, changed)
    dirty_only = deepcopy(release)
    dirty_only["git"]["dirty"] = True
    assert harness.provenance_change_errors(release, dirty_only, dirty_only) == []

    wrong_source = deepcopy(release)
    wrong_source["dynamo_core"]["build_git_revision"] = "def"
    assert "does not match" in harness.release_core_provenance_error(wrong_source)

    dirty_core = deepcopy(release)
    dirty_core["dynamo_core"]["build_git_dirty"] = True
    assert "dirty" in harness.release_core_provenance_error(dirty_core)

    unknown_source = deepcopy(release)
    unknown_source["dynamo_core"].pop("build_git_revision")
    assert "source revision" in harness.release_core_provenance_error(unknown_source)


def test_arm_validation_rejects_malformed_records() -> None:
    result = _valid_result("inprocess")
    assert harness_validation.validate_arm_result(result, _args()) == []

    result["aiperf_metrics"]["records"]["malformed_records"] = 1
    errors = harness_validation.validate_arm_result(result, _args())
    assert "aiperf malformed_records is not zero" in errors


def test_valkey_health_rejects_failed_commands_and_admission_leaks() -> None:
    assert (
        harness_valkey.valkey_ha_validation_errors(
            _healthy_valkey_state(), authoritative_admission=True
        )
        == []
    )
    errors = harness_valkey.valkey_ha_validation_errors(
        _healthy_valkey_state(failed_calls=1), authoritative_admission=True
    )
    assert any("failed_calls=1" in error for error in errors)

    missing_replication = _healthy_valkey_state()
    del missing_replication["replica_commandstats"]["dynkv.apply_owned_at"]
    errors = harness_valkey.valkey_ha_validation_errors(
        missing_replication, authoritative_admission=True
    )
    assert any("replica commandstats" in error for error in errors)

    result = _valid_result("valkey_ha")
    result["topology"].update(
        {
            "valkey_client_endpoint_count": 1,
            "valkey_data_node_count": 2,
            "valkey_required_replica_acks": 1,
        }
    )
    result.update(
        {
            "valkey_expected_registered_ranks": 32,
            "valkey_registered_ranks": 32,
            "valkey_final_admission_stats": {"primary": 1, "replica": 1},
            "valkey_final_state": _healthy_valkey_state(),
            "valkey_authoritative_admission": True,
        }
    )
    assert any(
        "admission reservations leaked" in error
        for error in harness_validation.validate_arm_result(result, _args())
    )

    result = _valid_result("valkey_ha")
    result["valkey_client_pressure"]["ports"] = {}
    errors = harness_validation.validate_arm_result(result, _args())
    assert any("client-pressure" in error for error in errors)


def test_valkey_primary_kill_validation_requires_quorum_promotion() -> None:
    args = _args()
    args.kill_valkey_primary = True
    result = _valid_result("valkey_ha")
    result["topology"].update(
        {
            "valkey_client_endpoint_count": 2,
            "valkey_data_node_count": 2,
            "valkey_sentinel_count": 3,
            "valkey_required_replica_acks": 1,
        }
    )
    result["aiperf"]["fault_injection"] = {
        "status": "promoted",
        "promotion_seconds": 0.75,
    }
    result.update(
        {
            "valkey_expected_registered_ranks": 32,
            "valkey_registered_ranks": 32,
            "valkey_final_admission_stats": {"promoted": 0},
            "valkey_final_state": _healthy_promoted_valkey_state(),
            "valkey_authoritative_admission": True,
        }
    )

    assert harness_validation.validate_arm_result(result, args) == []

    result["aiperf"]["fault_injection"]["status"] = "promotion_timeout"
    errors = harness_validation.validate_arm_result(result, args)
    assert any("fault injection did not promote" in error for error in errors)


def test_promoted_singleton_validation_rejects_lost_ranks() -> None:
    state = _healthy_promoted_valkey_state()
    state["stats"][1] = 31

    errors = harness_valkey.valkey_singleton_validation_errors(
        state,
        expected_ranks=32,
        authoritative_admission=True,
    )

    assert any("registered ranks=31" in error for error in errors)


def test_fault_trigger_counts_only_completed_profiling_records() -> None:
    complete = {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1,
            "request_end_ns": 2,
            "was_cancelled": False,
        },
        "error": None,
    }

    assert harness_loadgen.is_completed_profiling_record(complete)
    for mutation in (
        {"metadata": {**complete["metadata"], "benchmark_phase": "warmup"}},
        {"metadata": {**complete["metadata"], "was_cancelled": True}},
        {"metadata": {**complete["metadata"], "request_end_ns": 0}},
        {"error": {"message": "failed"}},
    ):
        record = deepcopy(complete)
        record.update(mutation)
        assert not harness_loadgen.is_completed_profiling_record(record)


def test_valkey_teardown_log_scan_counts_owner_fencing_failures(tmp_path: Path) -> None:
    log = tmp_path / "mocker" / "dynamo-mocker.log.txt"
    log.parent.mkdir()
    log.write_text(
        "healthy line\n"
        "Failed to publish event error=DYNKV_STALE_WORKER_OWNER\n"
        "Direct Valkey KV metadata integrity fault\n"
        "Valkey lifecycle GC tick failed; serving and lease heartbeat continue\n"
        "Failed to unregister mocker Valkey lease\n"
    )

    result = harness_valkey.scan_valkey_teardown_logs(tmp_path)

    # One line can intentionally match both the generic publish failure and
    # the more specific owner-fencing marker.
    assert result["failure_count"] == 5
    assert result["counts"]["DYNKV_STALE_WORKER_OWNER"] == 1
    assert result["counts"]["Failed to publish event"] == 1
    assert result["counts"]["Failed to unregister mocker Valkey lease"] == 1
    assert result["counts"]["Direct Valkey KV metadata integrity fault"] == 1
    assert result["counts"]["Valkey lifecycle GC tick failed"] == 1


def test_scale_rejects_different_generated_input_dataset() -> None:
    first = {
        "valid": True,
        "validation_errors": [],
        "aiperf_input_sha256": "a" * 64,
    }
    same = {
        "valid": True,
        "validation_errors": [],
        "aiperf_input_sha256": "a" * 64,
    }
    changed = {
        "valid": True,
        "validation_errors": [],
        "aiperf_input_sha256": "b" * 64,
    }

    scale.require_consistent_input_dataset(same, [first])
    scale.require_consistent_input_dataset(changed, [first])

    assert same["valid"] is True
    assert changed["valid"] is False
    assert "SHA-256 differs" in changed["validation_errors"][0]


def test_scale_rejects_changed_child_provenance() -> None:
    first = {
        "valid": True,
        "validation_errors": [],
        "child_provenance": {"benchmark_harness": {"sha256": "a" * 64}},
    }
    changed = {
        "valid": True,
        "validation_errors": [],
        "child_provenance": {"benchmark_harness": {"sha256": "b" * 64}},
    }

    scale.require_consistent_child_provenance(changed, [first])

    assert changed["valid"] is False
    assert "provenance differs" in changed["validation_errors"][0]


def test_partial_or_mixed_input_campaign_suppresses_comparisons() -> None:
    arms = ("inprocess", "valkey_ha")
    schedule = harness.build_arm_schedule(arms, 1)
    first = _valid_result("inprocess")
    analysis = harness.summarize_results([first], schedule)
    assert analysis["valid"] is False
    assert analysis["comparisons"] == []
    assert analysis["comparisons_suppressed"] is True

    second = _valid_result("valkey_ha", input_sha256="b" * 64)
    second["sample_index"] = 2
    analysis = harness.summarize_results([first, second], schedule)
    assert analysis["valid"] is False
    assert analysis["comparisons"] == []
    assert any("SHA-256 differs" in error for error in analysis["validation_errors"])
