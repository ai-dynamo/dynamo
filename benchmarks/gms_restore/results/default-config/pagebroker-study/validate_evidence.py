# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Offline qualification of an archived native PageBroker/GMS restore.

Run from any directory with Python 3. No Kubernetes or payload reads occur.
The default case writes validation.json. --no-write validates without writing.
Saved case evidence is never modified.
"""

import argparse
import hashlib
import json
import re
import sys
from itertools import pairwise
from pathlib import Path

STUDY = Path(__file__).resolve().parent
BASE = STUDY.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="pagebroker-native16-1", choices=[
    "pagebroker-native16-1", "pagebroker-native16-2", "pagebroker-native64-1", "pagebroker-native64-2"])
parser.add_argument("--no-write", action="store_true")
args = parser.parse_args()
CASE = BASE / args.case
sys.path.insert(0, str(BASE.parents[1]))
from resolve_plan import resolve_destinations


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    return json.loads(path.read_text())


def events(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line[line.index("{"):])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def one(values, **fields):
    matching = [v for v in values if all(v.get(k) == x for k, x in fields.items())]
    check(len(matching) == 1, f"expected one event: {fields}")
    return matching[0]


def statuses(pod):
    result = {}
    for key in ("containerStatuses", "initContainerStatuses"):
        for value in pod["status"].get(key, []):
            check(value["restartCount"] == 0, f"restart: {value['name']}")
            result[value["name"]] = value["containerID"]
    return result


def cpu(path):
    lines = path.read_text().splitlines()
    return {"epoch": float(lines[0]), "cgroup": lines[1], "quota": lines[2],
            **{p[0]: int(p[1]) for line in lines[4:] if len(p := line.split()) == 2}}


check((CASE / "cpu-main-after.txt").is_file(), "case collection incomplete")
t = load(CASE / "timing.json")
t0 = t["create_epoch"]
capture = load(BASE / "capture.json")
selected = load(STUDY / "snapshot-gms-capture.json")
plan = load(CASE / "plan.json")
claim = load(CASE / "claim.json")
ranks = {r["rank"]: r for r in plan["ranks"]}
check(len(ranks) == len(plan["ranks"]) == 8, "rank count")
check(plan["layout"] == capture["layout"] == {"tp": 8, "pp": 1, "dp": 1}, "layout")
check(plan["capture_id"] == capture["capture_id"] == t["capture_id"], "capture identity")
check(plan["claim_uid"] == claim["metadata"]["uid"], "claim UID")
check(all(selected[k] == v for k, v in capture.items()), "selected capture changed")
destinations = resolve_destinations(capture, claim, load(CASE / "slices.json"))
for original in capture["ranks"]:
    rank = ranks[original["rank"]]
    check(all(rank[k] == v for k, v in original.items()), "exact capture ID/size/artifact drift")
    check(rank["request"] == f"tp-{rank['rank']}" and
          rank["destination_uuid"] == destinations[rank["request"]], "DRA rank/UUID join")
    check(rank["local_device"] == 0 and rank["socket_device"] == rank["rank"] and
          rank["socket_dir"] == "/gms", "socket/local ordinal")
expected_map = ",".join(f"{ranks[i]['source_uuid']}={ranks[i]['destination_uuid']}" for i in range(8))
check(expected_map == plan["cuda_device_map"], "CUDA map")

pod = load(CASE / "ready-pod.json")
created = load(CASE / "operator-created-pod.json")
check(pod["metadata"]["uid"] == created["metadata"]["uid"], "engine Pod replaced")
statuses(pod)
check([c["name"] for c in pod["spec"]["containers"]] == ["main"], "engine has extra containers")
main = pod["spec"]["containers"][0]
env = {e["name"]: e.get("value") for e in main["env"]}
check(env["SNAPSHOT_CUDA_DEVICE_MAP"] == expected_map and env["DYN_GMS_USE_V1"] == "true" and
      env["GMS_SOCKET_DIR"] == "/gms", "engine mapping/env")
check(main["resources"]["requests"]["cpu"] == "32" and
      main["resources"]["limits"]["cpu"] == "96", "engine CPU settings")
volumes = {v["name"]: v for v in pod["spec"]["volumes"]}
shm_mount = one(main["volumeMounts"], mountPath="/dev/shm")
check(volumes[shm_mount["name"]]["emptyDir"]["sizeLimit"] == "32Gi", "engine shared memory")
check(volumes["snapshot-cuda"]["hostPath"]["path"] == t["preinstalled_cuda_host_path"] and
      one(main["volumeMounts"], name="snapshot-cuda")["readOnly"], "preinstalled CUDA bundle")
check(t["runtime_discovery"] and t["runtime_network_discovery"] and
      t["agent_revision"] == "b209afc332631c5a4c8f747d59dccecd32b4d03e", "agent discovery revision")

dc, dr = load(CASE / "dgd-created.json"), load(CASE / "dgd-ready.json")
uid = dc["metadata"]["uid"]
owners = load(CASE / "ownership.json")
check([o["kind"] for o in owners] == ["Pod", "ReplicaSet", "Deployment",
      "DynamoComponentDeployment", "DynamoGraphDeployment"], "operator ownership chain")
check(owners[0]["uid"] == pod["metadata"]["uid"] and owners[-1]["uid"] == uid == dr["metadata"]["uid"], "owner UIDs")
for child, parent in pairwise(owners):
    check(any(o["uid"] == parent["uid"] and o["kind"] == parent["kind"] and o.get("controller")
              for o in child["ownerReferences"]), "broken owner chain")
check(any(c["type"] == "Ready" and c["status"] == "True" for c in dr["status"]["conditions"]), "DGD not Ready")
watch_counts = {}
for filename in ("virtual-watch.jsonl", "host-watch.jsonl"):
    watch = [w for w in events(CASE / filename) if "uid" in w]
    check(len({w["uid"] for w in watch}) == 1, "watched Pod replaced")
    check(all(s["restartCount"] == 0 for w in watch for key in ("containers", "init")
              for s in w.get(key) or []), "watched container restart")
    if filename.startswith("virtual"):
        check(watch[0]["uid"] == pod["metadata"]["uid"], "virtual watch UID")
    watch_counts[filename] = len(watch)

ready = load(CASE / "resident-gms-ready.json")
check(ready["ready"] and ready["state"] == "waiting" and ready["publications"] == 0 and
      ready["capture_id"] is None and ready["generation"] is None and ready["error"] is None,
      "generic pristine readiness")
service_generation = ready["service_generation"]
check(service_generation != uid == t["resident_gms_generation"], "service/load generations")
online = {x["rank"]: x for x in ready["online"]}
check(len(online) == len(ready["online"]) == 8 and set(online) == set(range(8)), "online ranks")
manifest = load(CASE / "resident-manifest.json")
cm = one(manifest["items"], kind="ConfigMap")
ds = one(manifest["items"], kind="DaemonSet")
check(set(cm["data"]) == {"pagebroker_server.py", "pagebroker_coordinator.py", "pagebroker_pb2.py",
      "resident_coordinator.py", "resolve_plan.py", "verify_publication.py", "claim.json", "slices.json"}, "capture-bound DS ConfigMap")
ds_spec = ds["spec"]["template"]["spec"]
gms_before, gms_after = load(CASE / "resident-gms-before.json"), load(CASE / "resident-gms-after.json")
for prefix in ("resident-gms", "resident-agent"):
    before, after = load(CASE / f"{prefix}-before.json"), load(CASE / f"{prefix}-after.json")
    check(before["metadata"]["uid"] == after["metadata"]["uid"], f"{prefix} replaced")
    check(statuses(before) == statuses(after), f"{prefix} container IDs changed")
    check(before["spec"]["nodeName"] == pod["spec"]["nodeName"], f"{prefix} node mismatch")
check(any(o["kind"] == "DaemonSet" for o in gms_before["metadata"]["ownerReferences"]), "GMS not DS owned")
check(gms_before["spec"]["resourceClaims"] == pod["spec"]["resourceClaims"], "shared claim mismatch")
check(t["resident_services_ready_verified_epoch"] < t0 and t["resident_gms_ready_verified_epoch"] <= t0,
      "resident service readiness after timer")
check(t["workers"] == ready["transfer_slots_per_gpu"] == 32 and t["chunk_mib"] == ready["chunk_mib"] == 128 and t["numa"], "PB tuning")
for container in ds_spec["containers"]:
    command = container.get("command", []) + container.get("args", [])
    check(not {"--capture-id", "--generation", "--artifact"}.intersection(command), "capture-specific service arguments")
    check(command[command.index("--service-generation") + 1] == service_generation, "service generation argument")
    check(not any(capture["capture_id"] in v.get("value", "") for v in container.get("env", [])), "capture-specific service env")

selection = load(CASE / "snapshot-selection.json")
trigger = load(CASE / "resident-gms-trigger.json")
response = trigger["response"]
check(selection == t["snapshot_selection"] and trigger == t["resident_gms_load_trigger"], "selection/trigger evidence")
check(selection["dgd_uid"] == response["dgd_uid"] == response["generation"] == uid, "DGD load binding")
check(selection["snapshot_id"] == response["snapshot_id"] == selected["snapshot_content_uid"], "Snapshot content binding")
check(selection["capture_manifest_path"] == response["capture_manifest_path"] ==
      f"/checkpoints/artifacts/{selection['snapshot_id']}/gms/capture.json", "selected manifest path")
check(hashlib.sha256((STUDY / "snapshot-gms-capture.json").read_bytes()).hexdigest() == response["capture_manifest_sha256"], "selected manifest SHA")
check(t0 < t["dgd_create_return_epoch"] <= selection["discovery_started_epoch"] <=
      selection["discovery_completed_epoch"] <= trigger["request_epoch"] <= response["load_request_epoch"] <=
      response["manifest_discovery_start_epoch"] <= response["manifest_discovery_complete_epoch"] <=
      trigger["trigger_written_epoch"] <= trigger["return_epoch"], "post-DGD selection/discovery sequence")

pb_events = events(CASE / "pagebroker.txt")
pb_ready = one(pb_events, event="ready", retained_contexts=8)
check(pb_ready["retained_contexts"] == 8 and pb_ready["buffer_count"] == 32 and
      pb_ready["pinned_bytes_per_device"] == 4 * 1024**3 and pb_ready["chunk_bytes"] == 128 * 1024**2,
      "PB context/ring configuration")
# A resident PageBroker log can contain older completed loads. Fence this
# trial's evidence by its fresh DGD UID, just as the native protocol does.
phases = [e for e in pb_events if e.get("event") == "gms_load_phase" and e.get("generation") == uid]
pubs_list = load(CASE / "publications.json")
pubs = {p["rank"]: p for p in pubs_list}
check(len(pubs) == len(pubs_list) == 8, "publication count")
all_paths, all_ids, rank_summaries = set(), set(), []
for i in range(8):
    rank, state, pub = ranks[i], online[i], pubs[i]
    check(state["uuid"] == rank["destination_uuid"] and state["service_generation"] == service_generation and
          "capture_id" not in state and "generation" not in state and state["mode"] == "pagebroker", "generic rank identity")
    check(state["local_device"] == 0 and state["visible_device_count"] == 1 and state["socket_device"] == i and state["socket_dir"] == "/gms", "local single GPU socket identity")
    check(all(state[k] == 0 for k in ("payload_bytes_read", "weight_allocations", "pinned_bytes", "loader_lanes_ready")), "GMS preloaded payload/context ring")
    check(not state["cuda_context_current"] and not state["cuda_primary_context_active"], "GMS online CUDA context")
    check(set(state["server_domains_responsive"]) == {"weights", "kv_cache"}, "GMS endpoints")
    check(state["daemon_started_epoch"] < state["online_epoch"] < t0, "GMS readiness timing")
    gms = one(ds_spec["containers"], name=f"gms-{i}")
    check(gms["resources"]["claims"] == [{"name": "gpus", "request": f"tp-{i}"}] and
          gms["resources"]["requests"]["cpu"] == "1" and gms["resources"]["limits"]["cpu"] == "8", "rank claim/CPU")
    check({m["name"] for m in gms["volumeMounts"]} == {"app", "gms"} and "--numa" in gms["command"], "GMS artifact mount/NUMA")
    logs = events(CASE / f"gms-{i}.txt")
    committed_server = one(logs, event="weights_committed")
    for event in [one(logs, event="allocation_context_state"), committed_server]:
        check(event["context_state"]["current_context"] == 0 and not event["context_state"]["primary_context_active"], "GMS context at allocation/commit")
    check(committed_server["weights_server_nonce"] == state["weights_server_nonce"], "server nonce changed")
    previous_numa = one(events(BASE / "daemonset-network-2" / f"gms-{i}.txt"), event="numa_affinity")
    numa = one(logs, event="numa_affinity")
    check((numa["node"], numa["cpus"]) == (previous_numa["node"], previous_numa["cpus"]), "rank NUMA affinity drift")
    allocations = rank["allocations"]
    ids = {a["allocation_id"] for a in allocations}
    check(len(allocations) == len(ids) == 28 and not all_ids.intersection(ids), "exact unique allocation IDs")
    all_ids.update(ids)
    expected_shards = {a["shard"] for a in allocations}
    check(len(expected_shards) == 14 and sum(a["aligned_size"] for a in allocations) == 56 * 1024**3, "allocation/shard sizes")
    for shard in expected_shards:
        extents = sorted((a["offset"], a["aligned_size"]) for a in allocations if a["shard"] == shard)
        check(extents == [(0, 2 * 1024**3), (2 * 1024**3, 2 * 1024**3)], "contiguous captured extents")
        all_paths.add(str(Path(rank["artifact"]) / shard))
    rank_phases = [e for e in phases if e["rank"] == i]
    check(all(e["capture_id"] == capture["capture_id"] and e["generation"] == uid and
              e["destination_uuid"] == rank["destination_uuid"] for e in rank_phases), "native phase identity")
    starts = [e for e in rank_phases if e["phase"] == "read_copy_start"]
    ends = [e for e in rank_phases if e["phase"] == "read_copy_complete"]
    check(len(starts) == len(ends) == 14 and {e["shard"] for e in starts} == expected_shards == {e["shard"] for e in ends}, "112 exact source shards")
    check(all(e["o_direct"] and e["bytes"] == 4 * 1024**3 for e in starts), "O_DIRECT source flags/bytes")
    check(sum(e["bytes"] for e in ends) == 56 * 1024**3, "completed native bytes")
    for start in starts:
        check(pub["started_epoch"] * 1e9 <= start["epoch_ns"] <= one(ends, shard=start["shard"])["epoch_ns"], "read timing")
    ordered = [one(rank_phases, phase=phase)["epoch_ns"] / 1e9 for phase in (
        "validated", "lease_start", "lease_acquired", "allocation_import_start", "allocation_import_complete",
        "copies_drained", "imports_released", "commit_start", "committed")]
    check(ordered == sorted(ordered) and trigger["trigger_written_epoch"] <= pub["started_epoch"] <= ordered[0], "native lifecycle ordering")
    check(max(e["epoch_ns"] for e in ends) / 1e9 <= ordered[-4], "copies not drained")
    report = pub["pagebroker_report"]
    check(report["server_nonce"] == state["weights_server_nonce"] and report["manifest_sha256"] == rank["manifest_sha256"], "nonce/manifest report")
    check(pub["uuid"] == report["destination_uuid"] == rank["destination_uuid"] and
          pub["dgd_uid"] == pub["generation"] == report["generation"] == uid and
          pub["capture_id"] == report["capture_id"] == capture["capture_id"], "publication identity")
    check(report["bytes"] == committed_server["allocated_bytes"] == 56 * 1024**3 and
          report["allocations"] == committed_server["allocation_count"] == 28, "server published inventory counts")
    check(ordered[-2] <= report["committed_epoch_ns"] / 1e9 <= ordered[-1] <= pub["published_epoch"], "commit then publish")
    rank_summaries.append({"rank": i, "allocation_count": 28, "bytes": report["bytes"], "shards": 14,
                           "first_read_offset_s": min(e["epoch_ns"] / 1e9 for e in starts) - t0,
                           "committed_offset_s": report["committed_epoch_ns"] / 1e9 - t0})
check(len(all_ids) == 224 and len(all_paths) == 112, "total unique IDs/paths")

gate = float((CASE / "gate.txt").read_text())
verified = one(events(CASE / "coordinator.txt"), event="all_ranks_verified")
wake, = [e for e in events(CASE / "main.txt") if "entered_epoch" in e and "passed_epoch" in e]
check(max(p["published_epoch"] for p in pubs.values()) <= gate == verified["published_epoch"] <=
      wake["entered_epoch"] <= wake["passed_epoch"] <= t["ready_epoch"], "publication gate -> engine resume")
source_objects = load(BASE / "source-manifest-nccl.json")["items"]
app_cm = one(source_objects, kind="ConfigMap")
app = app_cm["data"]["app.py"]
check(app.index("GMS_WAKE_GATE") < app.index('engine.resume_memory_occupation(tags=["weights", "kv_cache"])'), "captured engine gate placement")
check("berlin" in t["restored_text"].lower() and "rayleigh" in load(CASE / "inference.json")["text"].lower(), "restored and HTTP inference")
source_main = one(one(source_objects, kind="Pod")["spec"]["containers"], name="main")
source_env = {e["name"]: e.get("value") for e in source_main["env"]}
inherited_nccl = {k: v for k, v in source_env.items() if k.startswith(("NCCL_", "TORCH_NCCL_"))}
check(all(env[k] == v for k, v in inherited_nccl.items()), "inherited NCCL configuration changed")
source_log = (BASE / "source-main.txt").read_text()
check("FlashInfer AllReduce Fusion enabled and workspace initialized: backend=mnnvl" in source_log and
      "enforce_disable_flashinfer_allreduce_fusion=False" in source_log and
      "disable_custom_all_reduce=False" in source_log and "disable_cuda_graph=False" in source_log,
      "captured engine communication configuration")

mount, = load(CASE / "pvc-mount.json")["filesystems"]
check(mount["fstype"] == "nfs" and "/pvc-" in mount["source"], "PVC NFS source")
check({"ro", "nconnect=32", "nosharecache", "nosharetransport", "spread_reads"} <= set(mount["options"].split(",")), "PVC mount tuning")
agent = load(CASE / "resident-agent-before.json")
pb = one(agent["spec"]["containers"], name="pagebroker")
pb_mount = one(pb["volumeMounts"], mountPath="/checkpoints/gms-pvc")
pb_volume = one(agent["spec"]["volumes"], name=pb_mount["name"])
check(pb_mount["readOnly"] and pb_volume["hostPath"] == one(ds_spec["volumes"], name="artifacts")["hostPath"], "PB/coordinator same PVC source")
pb_before, pb_after = cpu(CASE / "cpu-pagebroker-before.txt"), cpu(CASE / "cpu-pagebroker-after.txt")
expected_cpu_limit = 16 if "native16" in CASE.name else 64
check(int(t["pagebroker_cpu_limit"]) == expected_cpu_limit, "case CPU cohort")
check(pb_before["cgroup"] == pb_after["cgroup"] and
      pb_before["quota"] == pb_after["quota"] == f"{expected_cpu_limit * 100000} 100000", "PB CPU quota")
check(pb["resources"]["requests"]["cpu"] == "2" and
      pb["resources"]["limits"]["cpu"] == t["pagebroker_cpu_limit"], "PB resource setting")
checksums = (STUDY / "cuda-bundle-sha256.txt").read_text().splitlines()
source_checksums = [f"{digest}  {Path(filename).name}" for digest, filename in
                    (line.split() for line in (BASE / "daemonset-study/preinstall-source-SHA256SUMS").read_text().splitlines())]
check(checksums == load(STUDY / "cuda-bundle-verified.json")["sha256_lines"] ==
      (BASE / "daemonset-study/preinstall-SHA256SUMS").read_text().splitlines() ==
      source_checksums, "qualified shim bytes changed")
check(load(STUDY / "snapshot-gms-metadata.json")["payload_bytes_copied"] == 0, "payload hot staging")

patterns = {
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "jwt": r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "aws_access_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "bearer": r"Bearer\s+[A-Za-z0-9._-]{24,}",
    "kubeconfig_key": r"client-key-data:\s*\S{30,}",
}
scanned, matches = 0, []
for path in sorted([*CASE.iterdir(), *STUDY.iterdir()]):
    if not path.is_file() or path.name in {"validation.json", "validate_evidence.py"} or path.suffix in {".png"}:
        continue
    try:
        text = path.read_text()
    except UnicodeError:
        continue
    scanned += 1
    for name, pattern in patterns.items():
        if re.search(pattern, text):
            matches.append({"file": str(path.relative_to(BASE)), "pattern": name})
check(not matches, "secret-like content: filenames/pattern classes only " + json.dumps(matches))

report = {
    "schema_version": 1,
    "case": CASE.name,
    "status": "passed",
    "method": "Offline assertions over archived case objects, logs, capture plan and pinned bundle hashes; no cluster mutation or payload reads.",
    "measurement": {"origin": "DGD create request", "origin_epoch": t0, "ready_s": t["ready_epoch"] - t0,
        "dgd_post_return_s": t["dgd_create_return_epoch"] - t0,
        "snapshot_selection_complete_s": selection["discovery_completed_epoch"] - t0,
        "manifest_discovery_start_s": response["manifest_discovery_start_epoch"] - t0,
        "manifest_discovery_complete_s": response["manifest_discovery_complete_epoch"] - t0,
        "first_payload_read_s": min(x["first_read_offset_s"] for x in rank_summaries),
        "last_commit_s": max(x["committed_offset_s"] for x in rank_summaries),
        "all_ready_s": gate - t0, "engine_wake_gate_s": wake["passed_epoch"] - t0,
        "last_server_online_s": max(x["online_epoch"] for x in online.values()) - t0},
    "checks": {key: True for key in ["generic_service_has_no_capture_selection", "snapshot_selected_after_created_dgd_uid",
        "metadata_discovery_after_dgd_creation", "captured_rank_ids_sizes_hashes_unchanged", "named_dra_request_uuid_join",
        "engine_cuda_map_matches_gms_plan", "gms_single_visible_gpu_and_rank_socket", "gms_zero_preload_and_no_loader_ring",
        "gms_no_current_or_primary_cuda_context_at_online_allocation_commit", "server_nonces_match_native_commit_reports",
        "native_direct_reads_follow_load_request", "native_copies_drained_and_imports_released_before_commit",
        "all_commits_then_verified_publication_gate_then_engine_resume", "operator_owner_chain", "no_pod_replacement_or_container_restart",
        "default_berlin_and_http_rayleigh_inference", "retained_capture_nccl_and_fused_communication_configuration",
        "same_qualified_cuda_shim_bundle", "same_rank_numa_affinity_as_daemonset_network_2"]},
    "payload": {"owner": "PageBroker native GPU engine", "bytes": sum(x["bytes"] for x in rank_summaries),
        "gib": 448, "unique_source_paths": len(all_paths), "o_direct_read_copy_records": 112,
        "unique_allocation_ids": len(all_ids), "ranks": 8, "payload_bytes_copied_when_publishing_snapshot_metadata": 0,
        "source_path_evidence": "Native shard names joined to the exact per-rank artifact path in plan.json; PB mount aliases the same PVC as coordinator.",
        "open_time_evidence": "Native source opens occur in the load RPC after trigger and before validated; logs timestamp read/copy, not each open syscall."},
    "tuning": {"pagebroker_cpu_request": 2, "pagebroker_cpu_limit": expected_cpu_limit, "pagebroker_slots_per_gpu": 32,
        "chunk_mib": 128, "pagebroker_retained_contexts": 8, "pagebroker_pinned_gib_per_gpu": 4,
        "gms_cpu_request_per_rank": 1, "gms_cpu_limit_per_rank": 8, "main_cpu_request": 32, "main_cpu_limit": 96,
        "pagebroker_cpu_seconds_between_samples": (pb_after["usage_usec"] - pb_before["usage_usec"]) / 1e6,
        "pagebroker_throttled_usec_delta": pb_after["throttled_usec"] - pb_before["throttled_usec"]},
    "inherited_nccl_environment": inherited_nccl,
    "rank_measurements": rank_summaries,
    "watch_records": watch_counts,
    "capture_manifest_sha256": response["capture_manifest_sha256"],
    "plan_sha256": hashlib.sha256((CASE / "plan.json").read_bytes()).hexdigest(),
    "cuda_bundle_sha256_lines": checksums,
    "secret_pattern_scan": {"text_files": scanned, "matches": matches, "scope": "this case and current top-level study artifacts"},
    "limits": [
        "One functional qualification, not a statistically established speedup.",
        "Operator-created DGD custom podTemplate restores a captured SGLang Engine API; native Dynamo checkpointRef/frontend/GMS integration remains experimental.",
        "Source and destination UUIDs are identical on the held DRA claim; no cross-node or permuted-GPU qualification.",
        "448 GiB counts padded captured allocation bytes, not unique model tensor bytes.",
        "O_DIRECT avoids client page-cache payload reads; NFS server/storage cache state was not flushed or characterized.",
        "Context checks observe the GMS main-thread current context and device primary context at sampled stages; PageBroker intentionally owns CUDA contexts.",
        "Exact inventory equality is enforced in native pre-commit and coordinator RO verification; archived reports contain counts/hashes plus success, not a second full inventory dump.",
        "The generic GMS service is independent of captures, but this coordinator qualification accepts one load and has no reset/reuse API.",
        "Only capture metadata is colocated with Snapshot content; exact payload artifacts remain on their existing PVC paths.",
        "API creation timestamps have one-second precision; watch observation times are used for sub-second ordering."
    ],
}
if not args.no_write:
    filename = "validation.json" if CASE.name == "pagebroker-native16-1" else f"validation-{CASE.name}.json"
    (STUDY / filename).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"case": CASE.name, "status": report["status"], "ready_s": report["measurement"]["ready_s"],
                  "payload_gib": 448, "paths": 112, "allocation_ids": 224, "secret_matches": len(matches)}))
