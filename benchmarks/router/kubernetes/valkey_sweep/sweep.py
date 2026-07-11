#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deploy and run the HA Valkey router Kubernetes scaling matrix."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Sequence

from .aiperf_runner import RuntimeTopology, run_point, validate_metrics
from .artifacts import canonical_digest, file_digest, write_json_atomic
from .cluster import (
    Cluster,
    active_image_inventory,
    apply_stack,
    load_generator_capacity,
    pods,
    render_stack,
    rollout,
    sentinel_policy_snapshot,
    verify_active_images,
    wait_for_ha,
    wait_for_no_pods,
    wait_for_registered_mockers,
    wait_for_valkey_value,
)
from .ha import (
    dynamo_application_canary,
    elected_identity,
    ha_snapshot,
    prove_failover_and_restart,
)
from .methodology import methodology_binding, verify_methodology
from .network import prove_network_isolation
from .model import (
    CONCURRENCIES,
    FRONTEND_COUNTS,
    FRONTEND_DEPLOYMENT,
    ISL,
    MOCKER_COUNTS,
    MOCKER_DEPLOYMENT,
    MODEL,
    NAMESPACE,
    OSL,
    VALKEY_GIT_REVISION,
    MatrixPoint,
    matrix_points,
    router_write_durability,
    runtime_namespace,
)


CAMPAIGN_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,39}[a-z0-9])?")
__all__ = ["matrix_points", "render_stack"]


def campaign_contract(
    *,
    campaign: str,
    image: str,
    points: Sequence[MatrixPoint],
    methodology: dict[str, Any],
    prove_failover: bool,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "campaign": campaign,
        "namespace": NAMESPACE,
        "image": image,
        "methodology": methodology,
        "prove_failover": prove_failover,
        "model": MODEL,
        "configured_isl": ISL,
        "configured_osl": OSL,
        "offered_request_rate": "inf",
        "router_write_durability": router_write_durability(),
        "matrix": [point._asdict() for point in points],
    }
    return {**unsigned, "contract_digest": canonical_digest(unsigned)}


def campaign_manifest(
    *,
    contract: dict[str, Any],
    binding: dict[str, Any],
    network_isolation: dict[str, Any],
    sentinel_policy: dict[str, Any],
    ha_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 3,
        "contract": contract,
        "provenance": binding,
        "network_isolation": network_isolation,
        "sentinel_policy": sentinel_policy,
        "ha_proof": ha_proof or {"status": "not_requested"},
    }
    return {**unsigned, "manifest_digest": canonical_digest(unsigned)}


def validate_campaign_contract(
    existing: dict[str, Any], expected: dict[str, Any]
) -> None:
    digest = existing.get("contract_digest")
    unsigned = {
        key: value for key, value in existing.items() if key != "contract_digest"
    }
    if digest != canonical_digest(unsigned) or existing != expected:
        raise ValueError(
            "resume campaign contract does not exactly match its fingerprint or "
            "the requested image, methodology, failover intent, workload, and matrix"
        )


def validate_manifest_fingerprint(manifest: dict[str, Any]) -> None:
    digest = manifest.get("manifest_digest")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if digest != canonical_digest(unsigned):
        raise ValueError("campaign manifest fingerprint is invalid")


def validate_resume_manifest(
    existing: dict[str, Any], expected: dict[str, Any]
) -> None:
    validate_manifest_fingerprint(existing)
    if existing != expected:
        raise ValueError(
            "resume manifest does not exactly match its fingerprint or the requested "
            "image, provenance, HA proof, workload, and matrix"
        )


def completed_result_matches(
    result: dict[str, Any],
    attempt: dict[str, Any],
    point: MatrixPoint,
    binding: dict[str, Any],
    campaign: str,
    manifest_digest: str,
    summary_digest: str,
    validated_metrics: dict[str, float],
) -> bool:
    return (
        result.get("status") == "ok"
        and result.get("campaign") == campaign
        and result.get("manifest_digest") == manifest_digest
        and result.get("point") == point._asdict()
        and result.get("provenance") == binding
        and result.get("aiperf_summary_digest") == summary_digest
        and result.get("metrics") == validated_metrics
        and attempt.get("status") == "complete"
        and attempt.get("campaign") == campaign
        and attempt.get("manifest_digest") == manifest_digest
        and attempt.get("point") == point._asdict()
        and attempt.get("provenance") == binding
        and attempt.get("result_digest") == canonical_digest(result)
        and attempt.get("aiperf_summary_digest") == summary_digest
        and attempt.get("generation") == result.get("attempt_generation")
        and attempt.get("result") == "result.json"
    )


def begin_attempt(
    point_dir: Path,
    point: MatrixPoint,
    binding: dict[str, Any],
    campaign: str,
    manifest_digest: str,
) -> dict[str, Any]:
    attempt = {
        "status": "starting",
        "generation": secrets.token_hex(8),
        "campaign": campaign,
        "manifest_digest": manifest_digest,
        "point": point._asdict(),
        "provenance": binding,
    }
    write_json_atomic(point_dir / "attempt.json", attempt)
    return attempt


def _flush_replicated_group(cluster: Cluster, master_name: str) -> dict[str, list[str]]:
    _, replica, _, master = elected_identity(cluster, master_name)
    flush = cluster.valkey(master[0], "FLUSHDB")
    if flush != ["OK"]:
        raise RuntimeError(f"{master_name} reset failed: FLUSHDB={flush}")
    primary_dbsize = wait_for_valkey_value(cluster, master[0], ("DBSIZE",), ["0"])
    replica_dbsize = wait_for_valkey_value(cluster, replica, ("DBSIZE",), ["0"])
    return {
        "flush": flush,
        "primary_dbsize": primary_dbsize,
        "replica_dbsize": replica_dbsize,
    }


def configure_topology(
    cluster: Cluster,
    campaign: str,
    point: MatrixPoint,
    binding: dict[str, Any],
    generation: str,
) -> RuntimeTopology:
    point_namespace = runtime_namespace(campaign, point, generation)
    cluster.kubectl(
        (
            "scale",
            f"deployment/{FRONTEND_DEPLOYMENT}",
            f"deployment/{MOCKER_DEPLOYMENT}",
            "--replicas=0",
        )
    )
    wait_for_no_pods(cluster, FRONTEND_DEPLOYMENT)
    wait_for_no_pods(cluster, MOCKER_DEPLOYMENT)
    router_reset = _flush_replicated_group(cluster, "dynamo-router")
    tokenizer_reset = _flush_replicated_group(cluster, "dynamo-tokenizer")

    cluster.kubectl(
        (
            "set",
            "env",
            f"deployment/{MOCKER_DEPLOYMENT}",
            f"DYN_NAMESPACE={point_namespace}",
        )
    )
    cluster.kubectl(
        ("scale", f"deployment/{MOCKER_DEPLOYMENT}", f"--replicas={point.mockers}")
    )
    rollout(cluster, MOCKER_DEPLOYMENT)
    stats = wait_for_registered_mockers(cluster, point_namespace, point.mockers)

    cluster.kubectl(
        (
            "set",
            "env",
            f"deployment/{FRONTEND_DEPLOYMENT}",
            f"DYN_NAMESPACE={point_namespace}",
            f"DYN_ROUTER_MIN_INITIAL_WORKERS={point.mockers}",
        )
    )
    cluster.kubectl(
        ("scale", f"deployment/{FRONTEND_DEPLOYMENT}", f"--replicas={point.frontends}")
    )
    rollout(cluster, FRONTEND_DEPLOYMENT)
    cluster.kubectl(
        (
            "wait",
            "--for=condition=Ready",
            "pod",
            f"-l=app.kubernetes.io/name={FRONTEND_DEPLOYMENT}",
            "--timeout=10m",
        ),
        timeout=660,
    )

    frontend_pods = pods(cluster, f"app.kubernetes.io/name={FRONTEND_DEPLOYMENT}")
    ready_pods = sorted(
        (
            (pod["metadata"]["name"], pod["status"].get("podIP"))
            for pod in frontend_pods
            if pod["status"].get("phase") == "Running"
        ),
        key=lambda item: item[0],
    )
    if len(ready_pods) != point.frontends or any(ip is None for _, ip in ready_pods):
        raise RuntimeError(
            f"frontend pod set does not match M={point.frontends}: {ready_pods}"
        )
    urls = [f"http://{ip}:8000" for _, ip in ready_pods]
    for url in urls:
        cluster.client_exec(
            (
                "python",
                "-c",
                "import sys,urllib.request; "
                "r=urllib.request.urlopen(sys.argv[1], timeout=30); "
                "assert r.status == 200",
                f"{url}/v1/models",
            ),
            timeout=60,
        )
    wait_for_ha(cluster)
    loadgen_capacity = load_generator_capacity(cluster, point.concurrency)
    return {
        "runtime_namespace": point_namespace,
        "attempt_generation": generation,
        "registered_index_stats": stats,
        "frontend_pods": [name for name, _ in ready_pods],
        "frontend_urls": urls,
        "router_reset": router_reset,
        "router_write_durability": router_write_durability(),
        "tokenizer_reset": tokenizer_reset,
        "pre_runtime_snapshot": verify_active_images(cluster, binding, point),
        "pre_ha_snapshot": ha_snapshot(cluster),
        "load_generator_capacity": loadgen_capacity,
    }


def image_binding(cluster: Cluster, image: str) -> dict[str, Any]:
    core = cluster.client_exec(
        (
            "/app/.venv/bin/python",
            "-c",
            "import json,os,dynamo._core as c; print(json.dumps({"
            "'revision':c.__build_git_revision__,'dirty':c.__build_git_dirty__,"
            "'valkey_revision':os.environ.get('VALKEY_IMAGE_GIT_REVISION')}))",
        ),
        timeout=60,
    )
    core_identity = json.loads(core.stdout)
    revision = core_identity.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise RuntimeError(f"image core has invalid source revision: {core_identity}")
    if core_identity.get("dirty") is not False:
        raise RuntimeError(f"image core is not source-clean: {core_identity}")
    valkey_revision = core_identity.get("valkey_revision")
    if valkey_revision != VALKEY_GIT_REVISION:
        raise RuntimeError(
            f"image Valkey revision differs from expected {VALKEY_GIT_REVISION}: "
            f"{valkey_revision!r}"
        )
    inventory = active_image_inventory(cluster)
    image_ids = sorted({value for values in inventory.values() for value in values})
    if len(image_ids) != 1 or "@sha256:" not in image_ids[0]:
        raise RuntimeError(f"expected one immutable image digest, observed {image_ids}")
    image_references = {
        status["image"]
        for pod in pods(cluster, "app.kubernetes.io/part-of=valkey-router-sweep")
        for status in (
            pod["status"].get("initContainerStatuses", [])
            + pod["status"].get("containerStatuses", [])
        )
    }
    if image_references != {image}:
        raise RuntimeError(
            f"active pods do not use requested image reference {image!r}: "
            f"{sorted(image_references)}"
        )
    return {
        "image": image,
        "core_revision": revision,
        "core_dirty": False,
        "valkey_revision": valkey_revision,
        "image_ids": image_ids,
    }


def selected_points(args: argparse.Namespace) -> list[MatrixPoint]:
    frontends = tuple(args.frontends or FRONTEND_COUNTS)
    concurrencies = tuple(args.concurrencies or CONCURRENCIES)
    mockers = tuple(args.mockers or MOCKER_COUNTS)
    if not set(frontends) <= set(FRONTEND_COUNTS):
        raise ValueError(f"frontends must be a subset of {FRONTEND_COUNTS}")
    if not set(concurrencies) <= set(CONCURRENCIES):
        raise ValueError(f"concurrencies must be a subset of {CONCURRENCIES}")
    if not set(mockers) <= set(MOCKER_COUNTS):
        raise ValueError(f"mockers must be a subset of {MOCKER_COUNTS}")
    return [
        point
        for point in matrix_points()
        if point.frontends in frontends
        and point.concurrency in concurrencies
        and point.mockers in mockers
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--apply-stack", action="store_true")
    parser.add_argument("--prove-failover", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frontends", type=int, action="append")
    parser.add_argument("--concurrencies", type=int, action="append")
    parser.add_argument("--mockers", type=int, action="append")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not CAMPAIGN_RE.fullmatch(args.campaign):
        raise ValueError(
            "campaign must be a lowercase DNS label of at most 41 characters"
        )
    points = selected_points(args)
    cluster = Cluster(NAMESPACE)
    args.results.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results / "manifest.json"
    existing_manifest = _load_json(manifest_path) if manifest_path.is_file() else None
    if existing_manifest is not None and not args.resume:
        raise ValueError("results directory already has a manifest; use --resume")
    if existing_manifest is None and args.resume:
        raise ValueError("--resume requires an existing immutable manifest")
    methodology = methodology_binding(args.image)
    if existing_manifest is not None:
        validate_manifest_fingerprint(existing_manifest)
        existing_contract = existing_manifest.get("contract")
        if not isinstance(existing_contract, dict):
            raise ValueError("campaign manifest has no static contract")
        prove_failover = existing_contract.get("prove_failover") is True
        if args.prove_failover and not prove_failover:
            raise ValueError(
                "--prove-failover cannot change the immutable resume contract"
            )
        contract = campaign_contract(
            campaign=args.campaign,
            image=args.image,
            points=points,
            methodology=methodology,
            prove_failover=prove_failover,
        )
        validate_campaign_contract(existing_contract, contract)
        verify_methodology(
            methodology, existing_manifest["provenance"]["core_revision"]
        )
    else:
        prove_failover = args.prove_failover
        contract = campaign_contract(
            campaign=args.campaign,
            image=args.image,
            points=points,
            methodology=methodology,
            prove_failover=prove_failover,
        )

    if args.apply_stack:
        apply_stack(cluster, args.image)
    else:
        wait_for_ha(cluster)
    binding = image_binding(cluster, args.image)
    verify_methodology(methodology, binding["core_revision"])
    sentinel_policy = sentinel_policy_snapshot(cluster)
    network_isolation = prove_network_isolation(cluster, args.image)
    if existing_manifest is not None:
        ha_proof = existing_manifest["ha_proof"]
        if prove_failover and ha_proof.get("status") != "passed":
            raise ValueError(
                "resume contract requires a passed proof in the immutable manifest"
            )
    else:
        if prove_failover:
            proof_point = MatrixPoint(1, CONCURRENCIES[0], MOCKER_COUNTS[0])
            proof_topology = configure_topology(
                cluster,
                args.campaign,
                proof_point,
                binding,
                secrets.token_hex(8),
            )
            ha_proof = prove_failover_and_restart(
                cluster,
                lambda: dynamo_application_canary(
                    cluster,
                    proof_topology["frontend_urls"][0],
                    proof_topology["runtime_namespace"],
                    proof_point.mockers,
                ),
            )
        else:
            ha_proof = {"status": "not_requested"}
    manifest = campaign_manifest(
        contract=contract,
        binding=binding,
        network_isolation=network_isolation,
        sentinel_policy=sentinel_policy,
        ha_proof=ha_proof,
    )
    if existing_manifest is not None:
        validate_resume_manifest(existing_manifest, manifest)
    else:
        write_json_atomic(manifest_path, manifest)
    manifest_digest = manifest["manifest_digest"]

    results: list[dict[str, Any]] = []
    for point in points:
        point_dir = args.results / point.slug
        result_path = point_dir / "result.json"
        attempt_path = point_dir / "attempt.json"
        if args.resume and result_path.is_file():
            if not attempt_path.is_file():
                raise ValueError(
                    f"completed artifact for {point.slug} has no attempt record"
                )
            result = _load_json(result_path)
            attempt = _load_json(attempt_path)
            if attempt.get("status") == "starting":
                print(
                    f"[resume] rerunning uncommitted attempt for {point.slug}",
                    flush=True,
                )
            elif attempt.get("status") != "complete":
                raise ValueError(
                    f"attempt record for {point.slug} has invalid status: "
                    f"{attempt.get('status')!r}"
                )
            else:
                summary_path = point_dir / "profile_export_aiperf.json"
                if not summary_path.is_file():
                    raise ValueError(
                        f"completed artifact for {point.slug} has no raw AIPerf summary"
                    )
                summary_metrics = _load_json(summary_path)
                frontend_urls = result.get("frontend_urls")
                if not isinstance(frontend_urls, list) or not all(
                    isinstance(url, str) for url in frontend_urls
                ):
                    raise ValueError(
                        f"completed artifact for {point.slug} has invalid frontend URLs"
                    )
                validated_metrics = validate_metrics(
                    summary_metrics, point, frontend_urls
                )
                summary_digest = file_digest(summary_path)
                if completed_result_matches(
                    result,
                    attempt,
                    point,
                    binding,
                    args.campaign,
                    manifest_digest,
                    summary_digest,
                    validated_metrics,
                ):
                    print(f"[resume] skipping completed {point.slug}", flush=True)
                    results.append(result)
                    continue
                raise ValueError(
                    f"existing completed artifact for {point.slug} is not bound to "
                    "this campaign, manifest, attempt, point, and image provenance"
                )
        point_dir.mkdir(parents=True, exist_ok=True)
        attempt = begin_attempt(
            point_dir, point, binding, args.campaign, manifest_digest
        )
        topology = configure_topology(
            cluster, args.campaign, point, binding, attempt["generation"]
        )
        result = run_point(
            cluster,
            args.campaign,
            manifest_digest,
            point,
            topology,
            point_dir,
            binding,
        )
        results.append(result)
        write_json_atomic(
            attempt_path,
            {
                **attempt,
                "status": "complete",
                "result": "result.json",
                "result_digest": canonical_digest(result),
                "aiperf_summary_digest": result["aiperf_summary_digest"],
            },
        )
        write_json_atomic(args.results / "results.json", results)
        print(
            f"[complete] {point.slug}: "
            f"{result['metrics']['request_throughput_rps']:.3f} RPS",
            flush=True,
        )
    write_json_atomic(args.results / "results.json", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
