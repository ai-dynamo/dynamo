# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
import asyncio
from pathlib import Path
import struct
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.router import valkey_module_saturation as saturation  # noqa: E402
from benchmarks.router.valkey_saturation import workload as saturation_workload  # noqa: E402


def test_event_hash_ranges_do_not_overlap_above_256_blocks() -> None:
    blocks = 1024
    first = saturation.event_hash_start(7, 1, blocks)
    second = saturation.event_hash_start(7, 2, blocks)

    assert first + blocks <= second
    assert second < saturation.event_hash_start(8, 0, blocks)
    event = saturation.store_event(1, 1, first, blocks)
    assert len(event) == 34 + 16 * blocks


def test_remove_event_and_churn_ring_have_stable_wire_identities() -> None:
    blocks = 3
    first = saturation.churn_hash_start(1, 4, 2, 1, 8, blocks)
    wrapped = saturation.churn_hash_start(1, 4, 2, 9, 8, blocks)
    other_connection = saturation.churn_hash_start(1, 4, 3, 1, 8, blocks)
    store_id, remove_id = saturation.churn_event_ids(1, 4, 2, 9)

    assert first == wrapped
    assert first + 8 * blocks <= other_connection
    assert remove_id == store_id + 1
    block_hashes = tuple(first + offset for offset in range(blocks))
    remove = saturation.remove_event(7, remove_id, block_hashes)
    assert remove == struct.pack(
        "!BBQIQIQQQ",
        saturation.WIRE_VERSION,
        saturation.EVENT_REMOVE,
        7,
        0,
        remove_id,
        blocks,
        *block_hashes,
    )


def test_churn_owned_builds_ordered_store_remove_pair_with_split_latency() -> None:
    topology = saturation.Topology(
        preset="raw",
        workers=(7,),
        writer_workers=(7,),
        frontends=1,
        owners=(7,),
        leased=True,
        owner_nonces={7: 99},
    )
    setup = saturation.WorkloadSetup(
        key=b"key",
        topology=topology,
        block_hashes=(10,),
        local_hashes=(10 ^ saturation.XOR_MASK,),
        match_payload=b"match",
        select_payload=b"select",
        admission_candidates=((7, 0, 8),),
        domain=b"test",
        lease_ms=1_000,
    )
    commands = saturation.churn_owned_commands(
        setup,
        phase_id=1,
        connections=1,
        connection_id=0,
        sequence=3,
        prefixes_per_connection=2,
        blocks=2,
    )
    first_hash = saturation.churn_hash_start(1, 1, 0, 3, 2, 2)
    store_id, remove_id = saturation.churn_event_ids(1, 1, 0, 3)
    store = saturation.store_event(7, store_id, first_hash, 2)
    remove = saturation.remove_event(7, remove_id, (first_hash, first_hash + 1))

    assert [command.kind for command in commands] == ["apply_owned", "apply_owned"]
    assert [command.latency_kind for command in commands] == [
        "apply_owned_store",
        "apply_owned_remove",
    ]
    assert [command.event_kind for command in commands] == ["store", "remove"]
    assert store in commands[0].encoded
    assert remove in commands[1].encoded


def test_v3_registration_requeries_after_explicit_stale_generation(
    monkeypatch,
) -> None:
    generations = iter((7, 8))
    registrations: list[bytes] = []

    async def fake_execute(port: int, *parts: bytes) -> saturation.Resp:
        del port
        if parts[0] == b"DYNKV.REGISTRATION_GENERATION":
            return saturation.Resp(b"$", struct.pack("!Q", next(generations)), 15)
        assert parts[0] == b"DYNKV.REGISTER_WORKER_RANKS"
        registrations.append(parts[3])
        if len(registrations) == 1:
            raise saturation.RespCommandError("DYNKV_STALE_REGISTRATION_GENERATION", 43)
        return saturation.Resp(b"+", b"OK", 5)

    monkeypatch.setattr(saturation_workload, "execute", fake_execute)
    result = asyncio.run(
        saturation.register_leased_worker(
            1,
            b"key",
            11,
            22,
            30_000,
            base_retry_delay_s=0,
        )
    )

    assert result == b"OK"
    decoded = [struct.unpack("!BQQQII", payload) for payload in registrations]
    assert decoded == [
        (3, 22, 30_000, 7, 1, 0),
        (3, 22, 30_000, 8, 1, 0),
    ]
    assert saturation.DYNKV_PROTOCOL["leased_registration_version"] == 3


def test_v3_registration_fails_closed_on_malformed_generation(monkeypatch) -> None:
    async def fake_execute(port: int, *parts: bytes) -> saturation.Resp:
        del port, parts
        return saturation.Resp(b"+", struct.pack("!Q", 7), 11)

    monkeypatch.setattr(saturation_workload, "execute", fake_execute)
    try:
        asyncio.run(saturation.register_leased_worker(1, b"key", 11, 22, 30_000))
    except ValueError as error:
        assert "8-byte bulk" in str(error)
    else:
        raise AssertionError("malformed registration generation must fail closed")


def test_query_and_admission_wire_payloads_have_expected_shape() -> None:
    hashes = (11, 12)
    match = saturation.match_request(hashes)
    select = saturation.select_request(hashes, ((1, 0, 5), (2, 1, 6)))
    reserve = saturation.reserve_request(
        b"prefill",
        3,
        4,
        1_000,
        tuple(value ^ saturation.XOR_MASK for value in hashes),
        ((1, 0, 8), (2, 0, 8)),
    )

    assert match == struct.pack(
        "!BIQQ",
        saturation.WIRE_VERSION,
        2,
        11 ^ saturation.XOR_MASK,
        12 ^ saturation.XOR_MASK,
    )
    assert select.startswith(match + struct.pack("!I", 2))
    assert len(select) == len(match) + 4 + 2 * 20
    assert reserve[0] == saturation.ADMISSION_VERSION
    assert reserve.endswith(struct.pack("!QIIQII", 1, 0, 8, 2, 0, 8))


def test_semantic_response_parsers_reject_malformed_or_empty_results() -> None:
    match_payload = bytearray(struct.pack("!BI", saturation.WIRE_VERSION, 1))
    match_payload.extend(struct.pack("!QIIQ", 9, 0, 2, 77))
    assert saturation.parse_match_response(bytes(match_payload)) == [(9, 0, 2, 77)]

    select_payload = bytes((saturation.WIRE_VERSION, 1)) + struct.pack(
        "!QIIQ", 9, 0, 2, 77
    )
    assert saturation.parse_select_response(select_payload) == (9, 0, 2, 77)

    reservation_payload = bytes(
        (saturation.ADMISSION_VERSION, saturation.ADMISSION_RESERVED)
    ) + struct.pack("!QQQIQII", 1, 2, 9, 0, 100, 2, 1)
    reservation = saturation.parse_reservation_response(reservation_payload)
    assert reservation.worker == 9
    assert (
        saturation.parse_release_response(bytes((saturation.ADMISSION_VERSION, 1))) == 1
    )

    try:
        saturation.parse_match_response(bytes(match_payload[:-1]))
    except ValueError as error:
        assert "length" in str(error)
    else:
        raise AssertionError("truncated MATCH response must fail")


def test_latency_series_reports_all_requested_percentiles_and_bounds_memory() -> None:
    series = saturation.LatencySeries(limit=8)
    for value in range(1, 65):
        series.record(value * 1_000_000)
    summary = series.summary()

    assert summary["observations"] == 64
    assert summary["samples"] <= 8
    assert summary["sampling_stride"] > 1
    assert summary["p50_ms"] is not None
    assert summary["p90_ms"] is not None
    assert summary["p95_ms"] is not None
    assert summary["p99_ms"] is not None
    assert summary["p99_9_ms"] is not None
    assert summary["max_ms"] <= 64

    single_sample = saturation.LatencySeries(limit=1)
    for value in range(10):
        single_sample.record(value)
    assert len(single_sample.samples_ns) <= 1


def test_sweep_schedule_rotates_points_and_summary_finds_95_percent_knee() -> None:
    schedule = saturation.build_sweep_schedule((1, 2), (1, 4), 2)
    assert schedule[:4] == [(1, 1, 1), (1, 1, 4), (1, 2, 1), (1, 2, 4)]
    assert schedule[4] == (2, 1, 4)

    samples = []
    for connections, pipeline, rate in ((1, 1, 50.0), (1, 4, 96.0), (2, 4, 100.0)):
        samples.append(
            {
                "connections": connections,
                "pipeline": pipeline,
                "iterations_per_s": rate,
                "commands_per_s": rate,
                "latency": {"match": {"p99_ms": float(connections * pipeline)}},
            }
        )
    summary = saturation.campaign_summary(samples)
    assert summary["peak_observed"]["iterations_per_s_median"] == 100.0
    assert summary["closed_loop_knee"]["connections"] == 1
    assert summary["closed_loop_knee"]["pipeline"] == 4


def test_dynamo_preset_is_four_worker_three_frontend_topology() -> None:
    args = Namespace(preset="dynamo", workers=None, frontends=None, owners=None)
    topology = saturation.build_topology(args, connections=64, mode="apply_owned")

    assert topology.workers == (1, 2, 3, 4)
    assert topology.writer_workers == (1, 2, 3, 4)
    assert topology.frontends == 3
    assert topology.owners == topology.workers
    assert topology.leased is True


def test_worker_scale_preset_exercises_one_thousand_shared_prefix_owners() -> None:
    args = Namespace(preset="worker-scale", workers=None, frontends=None, owners=None)

    topology = saturation.build_topology(args, connections=64, mode="match")

    assert len(topology.workers) == 1024
    assert topology.owners == topology.workers
    assert topology.frontends == 3
    assert topology.leased is True


def test_churn_semantics_reject_retained_nodes_or_owner_residue() -> None:
    topology = saturation.Topology(
        preset="raw",
        workers=(7,),
        writer_workers=(7,),
        frontends=1,
        owners=(7,),
        leased=True,
        owner_nonces={7: 99},
    )
    setup = saturation.WorkloadSetup(
        key=b"key",
        topology=topology,
        block_hashes=(10, 11),
        local_hashes=(10, 11),
        match_payload=b"match",
        select_payload=b"select",
        admission_candidates=((7, 0, 8),),
        domain=b"test",
        lease_ms=1_000,
    )
    counters = saturation.Counters(
        commands=2,
        iterations=1,
        events=2,
        blocks=4,
        commands_by_kind={"apply_owned": 2},
        events_by_kind={"store": 1, "remove": 1},
    )

    def telemetry(nodes: int, mutations: int, inactive_owners: int = 0):
        return saturation.Telemetry(
            commandstats={},
            server={},
            memory={},
            persistence={},
            stats={},
            module_stats=[nodes, 1, mutations],
            admission_reservations=0,
            lifecycle_stats=[1, 0, 0],
            gc_stats=[mutations, 0, 0, 0, 0, inactive_owners, 0, 0],
            config={},
            process_cpu_s=0,
            client_cpu_s=0,
            process_memory_kib={},
        )

    before = telemetry(2, 10)
    after = telemetry(2, 12)
    commandstats = {"commandstats": {"dynkv.apply_owned": {"calls": 2}}}
    saturation.validate_measured_state(
        "churn_owned", counters, before, after, commandstats, setup
    )

    for bad_after in (telemetry(3, 12), telemetry(2, 12, inactive_owners=1)):
        try:
            saturation.validate_measured_state(
                "churn_owned", counters, before, bad_after, commandstats, setup
            )
        except RuntimeError as error:
            assert "churn" in str(error)
        else:
            raise AssertionError("retained churn state must invalidate the sample")


def test_help_describes_owned_churn_as_store_remove_iterations() -> None:
    help_text = " ".join(saturation.build_parser().format_help().split())

    assert "churn_owned counts one STORE+REMOVE" in help_text
    assert "pair as one iteration" in help_text
    assert "--churn-prefixes-per-connection" in help_text


def test_invalid_campaign_writes_csv_and_suppresses_plots(tmp_path: Path) -> None:
    result = {
        "samples": [
            {
                "status": "invalid",
                "connections": 1,
                "pipeline": 1,
                "latency": {},
            }
        ],
        "summary": {"points": []},
    }
    args = Namespace(artifact_dir=tmp_path, output=None)
    stale_plot = tmp_path / "throughput-vs-concurrency.png"
    stale_plot.write_bytes(b"stale")

    rendered = saturation.emit_campaign_artifacts(result, args)

    assert rendered["artifacts"]["status"] == "plots_suppressed"
    assert rendered["artifacts"]["plots_generated"] is False
    assert (tmp_path / "samples.csv").is_file()
    assert not stale_plot.exists()


def test_file_provenance_detects_artifact_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "dynkv.so"
    artifact.write_bytes(b"first")
    first = saturation.file_provenance(artifact)
    artifact.write_bytes(b"second")
    second = saturation.file_provenance(artifact)

    assert first["path"] == second["path"]
    assert first["sha256"] != second["sha256"]
    assert first["size_bytes"] != second["size_bytes"]
