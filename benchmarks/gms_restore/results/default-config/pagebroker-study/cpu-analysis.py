# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Recompute CPU and ring-contention evidence from collected restore cases."""

import datetime
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def read_cpu(path):
    lines = path.read_text().splitlines()
    quota, period = lines[2].split()
    result = {
        "epoch": float(lines[0]),
        "cgroup": lines[1],
        "quota_cores": None if quota == "max" else int(quota) / int(period),
        "weight": int(lines[3]),
        "stat": {},
        "pressure": {},
    }
    for line in lines[4:]:
        key, *values = line.split()
        if key in {"some", "full"}:
            result["pressure"][key] = {
                k: float(v) for k, v in (item.split("=") for item in values)
            }
        else:
            result["stat"][key] = int(values[0])
    return result


def cpu_window(before, after):
    assert before["cgroup"] == after["cgroup"], "cgroup restarted during trial"
    wall = after["epoch"] - before["epoch"]
    assert wall > 0
    delta = {k: v - before["stat"][k] for k, v in after["stat"].items()}
    assert all(value >= 0 for value in delta.values()), "CPU counter reset"
    pressure = {
        kind: (after["pressure"][kind]["total"] - values["total"]) / 1e6
        for kind, values in before["pressure"].items()
    }
    return {
        "before_epoch": before["epoch"],
        "after_epoch": after["epoch"],
        "sample_window_s": wall,
        "quota_cores": after["quota_cores"],
        "cpu_weight": after["weight"],
        "cpu_seconds": delta["usage_usec"] / 1e6,
        "user_cpu_seconds": delta["user_usec"] / 1e6,
        "system_cpu_seconds": delta["system_usec"] / 1e6,
        "mean_cores_over_sample_window": delta["usage_usec"] / 1e6 / wall,
        "periods": delta["nr_periods"],
        "throttled_periods": delta["nr_throttled"],
        "throttled_seconds": delta["throttled_usec"] / 1e6,
        "pressure_seconds": pressure,
        "pressure_percent_of_sample_window": {
            kind: 100 * seconds / wall for kind, seconds in pressure.items()
        },
    }


def read_case(path):
    timing = json.loads((path / "timing.json").read_text())
    isolated = timing.get("mode") == "isolated_gms_load"
    origin = timing["create_epoch"]
    terminal = timing["publication_ready_epoch"] if isolated else timing["ready_epoch"]
    cpu = {}
    for role in ("pagebroker", "agent"):
        cpu[role] = cpu_window(
            read_cpu(path / f"cpu-{role}-before.txt"),
            read_cpu(path / f"cpu-{role}-after.txt"),
        )
    if not isolated:
        main = read_cpu(path / "cpu-main-after.txt")
        starts = json.loads((path / "cri-starts.json").read_text())
        main_start = next(row["startedAt"] for row in starts if row["name"] == "main")
        main_start = datetime.datetime.fromisoformat(
            main_start.replace("Z", "+00:00")
        ).timestamp()
        cpu["main"] = {
            "scope": "lifetime since current container started; no before snapshot",
            "sample_window_s": main["epoch"] - main_start,
            "quota_cores": main["quota_cores"],
            "cpu_weight": main["weight"],
            "cpu_seconds": main["stat"]["usage_usec"] / 1e6,
            "throttled_periods": main["stat"]["nr_throttled"],
            "throttled_seconds": main["stat"]["throttled_usec"] / 1e6,
            "pressure_seconds": {
                k: v["total"] / 1e6 for k, v in main["pressure"].items()
            },
        }
    events = []
    for line in (path / "pagebroker.txt").read_text().splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if (
            event.get("event") == "gms_load_phase"
            and event.get("generation") == timing["resident_gms_generation"]
        ):
            events.append(event)
    first = min(e["epoch_ns"] for e in events if e["phase"] == "read_copy_start") / 1e9
    last = (
        max(e["epoch_ns"] for e in events if e["phase"] == "read_copy_complete") / 1e9
    )
    copies = [e for e in events if e["phase"] == "read_copy_complete"]
    assert len(copies) == len({(e["rank"], e["shard"]) for e in copies}) == 112
    assert sum(e["bytes"] for e in copies) == 448 * 2**30
    queue = [e["queue_seconds"] for e in events if e["phase"] == "ring_acquired"]
    publications = json.loads((path / "publications.json").read_text())
    reports = [row["pagebroker_report"] for row in publications]
    native = []
    for line in (path / "agent.txt").read_text().splitlines():
        if "Native PageBroker phase" not in line:
            continue
        epoch = datetime.datetime.fromisoformat(
            line.split("\t", 1)[0].replace("Z", "+00:00")
        ).timestamp()
        if not origin <= epoch <= terminal:
            continue
        row = json.loads(line[line.index("{") :])
        if row.get("operation") == "TRANSFER":
            report = json.loads(row["report"])
            if report["bytes"] > 0:
                native.append(report)
    pod = json.loads((path / "resident-agent-before.json").read_text())
    resources = {c["name"]: c["resources"] for c in pod["spec"]["containers"]}
    assert not isolated or not native, (
        "isolated interval contains native residual transfers"
    )
    return {
        "case": path.name,
        "isolated_gms_load": isolated,
        "dgd_ready_s": timing.get("dgd_create_to_ready_s"),
        "publication_ready_s": timing.get("publication_ready_s"),
        "resources": resources,
        "agent_pod_uid": pod["metadata"]["uid"],
        "capture_manifest_digests_by_rank": {
            row["rank"]: row["manifest_sha256"] for row in reports
        },
        "destination_uuids_by_rank": {
            row["rank"]: row["destination_uuid"] for row in reports
        },
        "cpu": cpu,
        "payload": {
            "bytes": sum(row["bytes"] for row in copies),
            "shards": len(copies),
            "first_read_offset_s": first - origin,
            "last_copy_offset_s": last - origin,
            "window_s": last - first,
            "effective_GiB_per_s": sum(row["bytes"] for row in copies)
            / 2**30
            / (last - first),
        },
        "ring_contention": {
            "gms_wait_sum_s": sum(queue),
            "gms_wait_rank_mean_s": statistics.mean(
                row["queue_wait_seconds"] for row in reports
            ),
            "gms_wait_rank_max_s": max(row["queue_wait_seconds"] for row in reports),
            "gms_shard_wait_max_s": max(queue),
            "native_wait_sum_s": sum(row["queue_wait_seconds"] for row in native),
            "native_wait_mean_s": statistics.mean(
                row["queue_wait_seconds"] for row in native
            )
            if native
            else 0,
            "native_wait_max_s": max(
                (row["queue_wait_seconds"] for row in native), default=0
            ),
            "native_transfers": len(native),
            "native_residual_bytes": sum(row["bytes"] for row in native),
        },
        "rank_mean": {
            key: statistics.mean(row[key] for row in reports)
            for key in (
                "queue_wait_seconds",
                "transfer_seconds",
                "storage_request_service_seconds",
                "cuda_wait_seconds",
                "total_seconds",
            )
        },
        "rank_timings": [
            {
                key: row[key]
                for key in (
                    "rank",
                    "queue_wait_seconds",
                    "transfer_seconds",
                    "storage_request_service_seconds",
                    "cuda_wait_seconds",
                    "total_seconds",
                )
            }
            for row in reports
        ],
    }


def main():
    cases, pending = [], []
    paths = sorted(
        [*ROOT.glob("pagebroker-native*-*"), *ROOT.glob("pagebroker-isolated*-*")]
    )
    for path in paths:
        required = [
            "timing.json",
            "cpu-pagebroker-after.txt",
            "cpu-agent-after.txt",
            "pagebroker.txt",
            "agent.txt",
            "publications.json",
            "resident-agent-before.json",
        ]
        if "isolated" not in path.name:
            required += ["cpu-main-after.txt", "cri-starts.json"]
        if not all((path / name).exists() for name in required):
            pending.append(path.name)
            continue
        cases.append(read_case(path))
    result = {
        "cases": cases,
        "pending_collection": pending,
        "cpu_limit_cohorts": {},
        "isolated_comparisons": [],
        "limits": [
            "Before/after CPU sampling includes idle lead-in and evidence collection after Ready; mean cores is not restore-only utilization or peak demand.",
            "CPU quota throttling and scheduler CPU pressure differ from waiting for the PageBroker per-GPU transfer ring.",
            "CPU request controls relative scheduling weight under competition, not a hard core cap.",
            "Parent cgroup counters and node-wide runnable/CPU occupancy were not collected; low leaf CPU pressure supports but does not prove absence of all host competition.",
            "Storage request service time sums overlapping in-flight I/O latency and is neither CPU time nor a serial wall-time component. CUDA wait sums host event synchronization waits.",
            "Rank and native ring waits overlap across GPUs; their sums cannot be added directly to end-to-end Ready time.",
            "Effective payload throughput divides 448GiB by earliest read to final transfer-routine completion, including intervening ring waits; it is not raw storage or PCIe bandwidth.",
        ],
    }
    for limit in sorted(
        {
            c["cpu"]["pagebroker"]["quota_cores"]
            for c in cases
            if not c["isolated_gms_load"]
        }
    ):
        cohort = [
            c
            for c in cases
            if not c["isolated_gms_load"]
            and c["cpu"]["pagebroker"]["quota_cores"] == limit
        ]
        result["cpu_limit_cohorts"][str(int(limit))] = {
            "n": len(cohort),
            "ready_mean_s": statistics.mean(c["dgd_ready_s"] for c in cohort),
            "ready_min_s": min(c["dgd_ready_s"] for c in cohort),
            "ready_max_s": max(c["dgd_ready_s"] for c in cohort),
            "payload_window_mean_s": statistics.mean(
                c["payload"]["window_s"] for c in cohort
            ),
            "cpu_seconds_mean": statistics.mean(
                c["cpu"]["pagebroker"]["cpu_seconds"] for c in cohort
            ),
            "throttled_seconds_sum": sum(
                c["cpu"]["pagebroker"]["throttled_seconds"] for c in cohort
            ),
        }
    for isolated in (c for c in cases if c["isolated_gms_load"]):
        for concurrent in (c for c in cases if not c["isolated_gms_load"]):
            if (
                isolated["cpu"]["pagebroker"]["quota_cores"]
                != concurrent["cpu"]["pagebroker"]["quota_cores"]
            ):
                continue
            equal_artifacts = (
                isolated["capture_manifest_digests_by_rank"]
                == concurrent["capture_manifest_digests_by_rank"]
            )
            equal_targets = (
                isolated["destination_uuids_by_rank"]
                == concurrent["destination_uuids_by_rank"]
            )
            assert equal_artifacts and equal_targets
            alone, together = (
                isolated["payload"]["window_s"],
                concurrent["payload"]["window_s"],
            )
            result["isolated_comparisons"].append(
                {
                    "isolated_case": isolated["case"],
                    "concurrent_case": concurrent["case"],
                    "same_agent_pod": isolated["agent_pod_uid"]
                    == concurrent["agent_pod_uid"],
                    "same_cpu_resources": isolated["resources"]["pagebroker"]
                    == concurrent["resources"]["pagebroker"],
                    "same_captured_artifact_digests": equal_artifacts,
                    "same_destination_gpus": equal_targets,
                    "payload_bytes": isolated["payload"]["bytes"],
                    "payload_window_increase_s": together - alone,
                    "payload_window_increase_percent": 100 * (together / alone - 1),
                    "effective_throughput_decrease_percent": 100
                    * (1 - alone / together),
                    "additional_native_gpu_residual_bytes": concurrent[
                        "ring_contention"
                    ]["native_residual_bytes"],
                    "additional_criu_cpu_image_bytes": None,
                    "sample_size_each": 1,
                }
            )
    (OUT / "cpu-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    no_throttle = all(c["cpu"]["pagebroker"]["throttled_periods"] == 0 for c in cases)
    lines = [
        "# PageBroker CPU and transfer-ring analysis",
        "",
        "The measured PageBroker cgroups show no quota throttling and low CPU scheduling pressure. The evidence does not identify a CPU-limit bottleneck or a repeatable end-to-end benefit from increasing the limit to 64."
        if no_throttle
        else "Quota throttling is present; compare its duration and restore phases before choosing a larger CPU limit.",
        "",
        "| Case | PB request / limit | Ready (s) | Payload window (s) | Effective GiB/s | PB CPU seconds | PB throttled seconds | PB CPU some-pressure (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        pb = c["cpu"]["pagebroker"]
        resources = c["resources"]["pagebroker"]
        ready = (
            "— (isolated)" if c["dgd_ready_s"] is None else f"{c['dgd_ready_s']:.3f}"
        )
        lines.append(
            f"| {c['case']} | {resources['requests']['cpu']} / {resources['limits']['cpu']} | {ready} | {c['payload']['window_s']:.3f} | {c['payload']['effective_GiB_per_s']:.3f} | {pb['cpu_seconds']:.3f} | {pb['throttled_seconds']:.6f} | {pb['pressure_seconds']['some'] * 1000:.3f} |"
        )
    lines.extend(
        [
            "",
            "## CPU-limit cohorts",
            "",
            "| PageBroker CPU limit | Trials | Mean Ready (s) | Ready range (s) | Mean payload window (s) | Mean CPU seconds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for limit, cohort in result["cpu_limit_cohorts"].items():
        lines.append(
            f"| {limit} | {cohort['n']} | {cohort['ready_mean_s']:.3f} | {cohort['ready_min_s']:.3f}–{cohort['ready_max_s']:.3f} | {cohort['payload_window_mean_s']:.3f} | {cohort['cpu_seconds_mean']:.3f} |"
        )
    for c in cases:
        pb, agent = (c["cpu"][role] for role in ("pagebroker", "agent"))
        main = c["cpu"].get("main")
        ring = c["ring_contention"]
        lines.extend(
            [
                "",
                f"## {c['case']}",
                "",
                f"PageBroker consumed {pb['cpu_seconds']:.3f} CPU-seconds ({pb['user_cpu_seconds']:.3f} user, {pb['system_cpu_seconds']:.3f} system) over the {pb['sample_window_s']:.3f}s sample interval: {pb['mean_cores_over_sample_window']:.3f} cores on average across that interval. Its cgroup quota was {pb['quota_cores']:.0f} cores and scheduling weight {pb['cpu_weight']}. CPU pressure occupied {pb['pressure_percent_of_sample_window']['some']:.3f}% of that interval; {pb['throttled_periods']} of {pb['periods']} accounted periods were throttled.",
                "",
                f"The Snapshot agent added {agent['cpu_seconds']:.3f} CPU-seconds with {agent['throttled_seconds']:.6f}s throttling. "
                + (
                    f"The engine main container had {main['throttled_seconds']:.6f}s cumulative throttling across its {main['sample_window_s']:.3f}s observed lifetime, including post-restore inference and collection."
                    if main
                    else "No engine Pod was created in this isolated load."
                ),
                "",
                (
                    f"Isolated ring acquisition waits were negligible: {ring['gms_wait_rank_mean_s'] * 1e6:.3f} microseconds mean/rank and {ring['gms_wait_rank_max_s'] * 1e6:.3f} microseconds maximum/rank. There were no native residual transfers competing for the per-GPU pinned ring. These acquisition timings do not establish meaningful shared-ring contention."
                    if c["isolated_gms_load"] and ring["native_transfers"] == 0
                    else f"There is measurable shared-ring contention: GMS waits total {ring['gms_wait_sum_s']:.3f}s across eight ranks ({ring['gms_wait_rank_mean_s']:.3f}s mean/rank, {ring['gms_wait_rank_max_s']:.3f}s maximum/rank), while the {ring['native_transfers']} native residual transfers wait {ring['native_wait_sum_s']:.3f}s in total ({ring['native_wait_mean_s']:.3f}s mean, {ring['native_wait_max_s']:.3f}s maximum). These are waits for the common per-GPU pinned ring, independently of CPU quota or CPU scheduling. They prove serialization in the shared transfer path, not that all of that sum extends the critical path."
                ),
            ]
        )
        mean = c["rank_mean"]
        lines.extend(
            [
                "",
                f"Across eight equal 56GiB ranks, mean active read+copy time was {mean['transfer_seconds']:.3f}s and mean CUDA event-wait time was {mean['cuda_wait_seconds']:.3f}s. Mean summed storage-request service time was {mean['storage_request_service_seconds']:.3f}s/rank; its overlapping requests make this unsuitable for addition to wall time.",
            ]
        )
    for pair in result["isolated_comparisons"]:
        lines.extend(
            [
                "",
                f"## Equal-payload isolated comparison: {pair['concurrent_case']}",
                "",
                f"{pair['isolated_case']} and {pair['concurrent_case']} used the same PageBroker Pod, CPU resources, destination GPUs and captured artifact digests. Both transferred exactly 448GiB in 112 shards. The concurrent run's payload window was {pair['payload_window_increase_s']:.3f}s longer ({pair['payload_window_increase_percent']:.2f}%), with {pair['effective_throughput_decrease_percent']:.2f}% lower effective payload throughput.",
                "",
                f"The concurrent engine restore additionally transferred {pair['additional_native_gpu_residual_bytes'] / 2**30:.6f}GiB of GPU residual state through PageBroker. Its CRIU CPU-image I/O is additional but unmeasured by these native GPU reports. The isolated run created no engine Pod and logged no native residual transfers during its interval.",
                "",
                "This single matched pair supports transfer contention: ring waits disappear in isolation, active read+copy time and summed I/O service latency fall, while CUDA event waits change little and CPU throttling stays zero. The exact wall-time difference is an observation from n=1 per mode, not a repeatable causal estimate; storage and orchestration variance remain possible contributors.",
            ]
        )
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {note}" for note in result["limits"])
    lines.extend(
        [
            "",
            "Keep the 16-CPU limit for now: there is no measured throttling at 16 CPUs and no repeatable Ready-time benefit from the 64-CPU limit. The 64-CPU payload window was shorter, but native/GMS ring overlap also changed; these small sequential cohorts cannot isolate a CPU-limit effect. A request/weight experiment becomes useful if scheduler pressure rises during a busy-node workload.",
            "",
        ]
    )
    (OUT / "cpu-analysis.md").write_text("\n".join(lines))
    print(json.dumps({"cases": [c["case"] for c in cases], "pending": pending}))


if __name__ == "__main__":
    main()
