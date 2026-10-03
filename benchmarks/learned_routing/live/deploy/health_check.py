# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness, smoke and cold-reset checks for the live deployment (stdlib only).

Subcommands (each writes a JSON report with ``--out`` and exits non-zero when a required check
fails):

``wait``      every worker's system server is ready, the frontend lists ``--expect`` generate
              instances in this namespace, and ``/v1/models`` serves the model.
``smoke``     instance->worker mapping by pinned requests, exact ISL/forced OSL on token-ID prompts,
              KV events reaching the frontend's indexer, prefix reuse through the router, and
              policy evidence from the frontend's config dump.
``reset``     cold-state reset: idle check, positive control, ``/engine/flush_cache`` on every
              worker, proof that a flushed prefix no longer hits, final flush.
``snapshot``  raw ``/metrics`` of the frontend and every worker, for before/after each payload.

Requests use ``/v1/completions`` with integer token prompts, ``ignore_eos``, ``min_tokens`` =
``max_tokens`` and ``nvext.extra_fields=["worker_id"]``; probe prompts are seeded random token IDs
that no workload produces. These are diagnostics, not benchmark numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_TOKEN_RANGE = (1000, 150000)  # well below Qwen3's special tokens (>= 151643)
PIN_HEADER = "x-dynamo-worker-instance-id"
SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


class CheckFailed(RuntimeError):
    pass


def http(
    method: str,
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def get_json(url: str, timeout: float = 10.0):
    status, body = http("GET", url, timeout=timeout)
    if status != 200:
        raise CheckFailed(f"GET {url} -> HTTP {status}: {body[:200]!r}")
    return json.loads(body)


def metrics(url: str) -> list[tuple[str, dict, float]]:
    status, body = http("GET", url, timeout=10.0)
    if status != 200:
        raise CheckFailed(f"GET {url} -> HTTP {status}")
    samples = []
    for line in body.decode(errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        try:
            samples.append((name, dict(LABEL.findall(labels or "")), float(value)))
        except ValueError:
            continue
    return samples


def metric_sum(samples, name: str, **labels) -> float | None:
    """Sum of samples named ``name`` or ``name_total`` whose labels match; None if absent."""
    names = {name, f"{name}_total"}
    found = [
        v
        for n, lab, v in samples
        if n in names and all(lab.get(k) == w for k, w in labels.items())
    ]
    return sum(found) if found else None


def metric_sum_suffix(samples, suffix: str, **labels) -> float | None:
    found = [
        v
        for n, lab, v in samples
        if (n.endswith(suffix) or n.endswith(suffix + "_total"))
        and all(lab.get(k) == w for k, w in labels.items())
    ]
    return sum(found) if found else None


def probe_tokens(tag: str, n: int) -> list[int]:
    rng = random.Random(hashlib.sha256(tag.encode()).digest())
    return [rng.randrange(*USER_TOKEN_RANGE) for _ in range(n)]


class Deployment:
    def __init__(self, args):
        self.frontend = args.frontend.rstrip("/")
        self.workers = [w if "://" in w else f"http://{w}" for w in args.worker]
        self.model = args.model
        self.namespace = args.namespace
        # Probe prompts differ per invocation, so a later check never hits an earlier probe's cache.
        self.nonce = str(time.time_ns())
        self.report: dict = {
            "checks": {},
            "frontend": self.frontend,
            "workers": self.workers,
            "nonce": self.nonce,
        }

    def tokens(self, tag: str, n: int) -> list[int]:
        return probe_tokens(f"{self.nonce}-{tag}", n)

    def record(self, name: str, ok: bool, **detail) -> None:
        self.report["checks"][name] = {"ok": ok, **detail}
        if not ok:
            raise CheckFailed(f"{name}: {json.dumps(detail, sort_keys=True)[:600]}")

    def complete(
        self,
        tokens: list[int],
        osl: int,
        pin: int | None = None,
        timeout: float = 600.0,
    ) -> dict:
        body = {
            "model": self.model,
            "prompt": tokens,
            "max_tokens": osl,
            "min_tokens": osl,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "nvext": {"extra_fields": ["worker_id"]},
        }
        headers = {PIN_HEADER: str(pin)} if pin is not None else {}
        request = urllib.request.Request(
            f"{self.frontend}/v1/completions",
            data=json.dumps(body).encode(),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        for key, value in headers.items():
            request.add_header(key, value)
        started = time.monotonic()
        first = None
        usage = None
        worker = None
        chunks = 0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw in response:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    if chunk.get("choices"):
                        chunks += 1
                        if first is None:
                            first = time.monotonic()
                    usage = chunk.get("usage") or usage
                    info = (chunk.get("nvext") or {}).get("worker_id")
                    if info:
                        worker = info.get("decode_worker_id") or info.get(
                            "prefill_worker_id"
                        )
        except urllib.error.HTTPError as exc:
            raise CheckFailed(
                f"completion HTTP {exc.code}: {exc.read()[:300]!r}"
            ) from exc
        ended = time.monotonic()
        return {
            "isl": len(tokens),
            "osl": osl,
            "usage": usage,
            "worker_id": worker,
            "chunks": chunks,
            "ttft_s": None if first is None else first - started,
            "e2e_s": ended - started,
        }

    def instances(self) -> list[int]:
        health = get_json(f"{self.frontend}/health")
        ids = sorted(
            int(i["instance_id"])
            for i in health.get("instances", [])
            if i.get("endpoint") == "generate"
            and (self.namespace is None or i.get("namespace") == self.namespace)
        )
        return ids

    def worker_metrics(self) -> list[list]:
        return [metrics(f"{w}/metrics") for w in self.workers]


def wait(dep: Deployment, expect: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    state: dict = {}
    while True:
        try:
            ready = []
            for worker in dep.workers:
                status, body = http("GET", f"{worker}/health", timeout=5.0)
                health = json.loads(body) if body else {}
                endpoints = health.get("endpoints") or {}
                ready.append(
                    (status == 200 and health.get("status") == "ready")
                    or endpoints.get("generate") == "ready"
                )
            ids = dep.instances()
            models = get_json(f"{dep.frontend}/v1/models").get("data", [])
            served = [m.get("id") for m in models]
            state = {"workers_ready": ready, "instances": ids, "models": served}
            if all(ready) and len(ids) >= expect and dep.model in served:
                break
        except (OSError, CheckFailed, json.JSONDecodeError) as exc:
            state = {"last_error": str(exc)[:300], **state}
        if time.monotonic() > deadline:
            dep.record("ready", False, timeout_s=timeout, expect=expect, **state)
        time.sleep(5.0)
    dep.record("ready", len(state["instances"]) == expect, expect=expect, **state)


def map_instances(dep: Deployment) -> dict[int, int]:
    """Map frontend instance IDs to ``--worker`` indices.

    Each worker's runtime metrics carry its instance ID as the hex ``worker_id`` label of the
    ``generate`` endpoint. Without that label, pin one tiny request per instance and take the
    worker whose vLLM success counter moves.
    """
    instances = dep.instances()
    labelled: dict[int, int] = {}
    for index, samples in enumerate(dep.worker_metrics()):
        for name, labels, _ in samples:
            if labels.get("dynamo_endpoint") == "generate" and "worker_id" in labels:
                labelled[int(labels["worker_id"], 16)] = index
    if sorted(labelled) == instances and len(set(labelled.values())) == len(
        dep.workers
    ):
        # Confirm the pin reaches the instance it names before relying on pins elsewhere.
        pins = [dep.complete(dep.tokens(f"map-{i}", 32), 2, pin=i) for i in instances]
        ok = all(r["worker_id"] in (None, i) for r, i in zip(pins, instances))
        dep.record(
            "instance_map",
            ok,
            method="worker_id label",
            mapping={str(k): v for k, v in labelled.items()},
            pinned_worker_ids=[r["worker_id"] for r in pins],
        )
        return labelled
    mapping: dict[int, int] = {}
    for instance in instances:
        before = [
            metric_sum(s, "vllm:request_success") or 0.0 for s in dep.worker_metrics()
        ]
        result = dep.complete(dep.tokens(f"map-{instance}", 32), 2, pin=instance)
        owner = None
        for _ in range(20):
            after = [
                metric_sum(s, "vllm:request_success") or 0.0
                for s in dep.worker_metrics()
            ]
            moved = [i for i, (a, b) in enumerate(zip(after, before)) if a > b]
            if len(moved) == 1:
                owner = moved[0]
                break
            time.sleep(0.5)
        if owner is None or result["worker_id"] not in (None, instance):
            dep.record("instance_map", False, instance=instance, result=result)
        mapping[instance] = owner
    ok = sorted(mapping.values()) == list(range(len(dep.workers)))
    dep.record(
        "instance_map",
        ok,
        method="pinned request + vllm:request_success",
        mapping={str(k): v for k, v in mapping.items()},
    )
    return mapping


def check_forced_osl(dep: Deployment, mapping: dict[int, int]) -> None:
    results = []
    for n, instance in enumerate(sorted(mapping)):
        isl = 1000 + 37 * n
        result = dep.complete(dep.tokens(f"osl-{instance}", isl), 64, pin=instance)
        usage = result["usage"] or {}
        result["ok"] = (
            usage.get("prompt_tokens") == isl
            and usage.get("completion_tokens") == 64
            and result["worker_id"] in (None, instance)
        )
        results.append(result)
    dep.record("exact_isl_forced_osl", all(r["ok"] for r in results), results=results)


def check_kv_events_and_reuse(
    dep: Deployment, expect_affinity: bool, block: int
) -> None:
    def stored(samples):
        return (
            metric_sum_suffix(samples, "kv_cache_events_applied", event_type="stored")
            or 0.0
        )

    base = dep.tokens("reuse-base", 2048)
    before = stored(metrics(f"{dep.frontend}/metrics"))
    first = dep.complete(base, 4)
    applied = None
    for _ in range(40):
        applied = stored(metrics(f"{dep.frontend}/metrics"))
        if applied > before:
            break
        time.sleep(0.25)
    dep.record(
        "kv_events_reach_router",
        applied is not None and applied > before,
        stored_before=before,
        stored_after=applied,
        first=first,
    )
    mismatch = metric_sum_suffix(
        metrics(f"{dep.frontend}/metrics"), "kv_event_source_mismatch_workers"
    )
    dep.record("kv_event_source_consistent", not mismatch, mismatch_workers=mismatch)

    hits_before = [
        metric_sum(s, "vllm:prefix_cache_hits") for s in dep.worker_metrics()
    ]
    second = dep.complete(base + dep.tokens("reuse-suffix", 128), 4)
    time.sleep(1.0)
    hits_after = [metric_sum(s, "vllm:prefix_cache_hits") for s in dep.worker_metrics()]
    hit_delta = [
        None if a is None or b is None else a - b
        for a, b in zip(hits_after, hits_before)
    ]
    cached = ((second["usage"] or {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    same = first["worker_id"] is not None and first["worker_id"] == second["worker_id"]
    # The engine reports the reused prefix per request; every full block of the shared prefix
    # except the request's last must hit when the router sends the extension to the same worker.
    reused = cached is not None and cached >= len(base) - block
    ok = same and reused if expect_affinity else True
    dep.record(
        "prefix_reuse",
        ok,
        first_worker=first["worker_id"],
        second_worker=second["worker_id"],
        same_worker=same,
        second_cached_tokens=cached,
        vllm_prefix_hit_tokens_by_worker=hit_delta,
        expect_affinity=expect_affinity,
    )


def check_policy_evidence(
    dep: Deployment, plan_path: Path | None, dump_path: Path | None
) -> None:
    if plan_path is None or dump_path is None:
        return
    plan = json.loads(plan_path.read_text())
    config = json.loads(dump_path.read_text()).get("config", {})
    detail: dict = {"policy": plan["name"], "router_mode": config.get("router_mode")}
    expected_mode = "round-robin" if plan["router_mode"] == "round_robin" else "kv"
    ok = config.get("router_mode") == expected_mode
    if plan.get("policy_yaml"):
        path = Path(config.get("router_policy_config") or "")
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        detail.update(router_policy_config=str(path), sha256=digest)
        ok = ok and digest == plan["policy_yaml_sha256"]
        # Every KvRouterConfig field the frontend config carries must equal the planned kwargs.
        diffs = {
            k: (config.get(k), v)
            for k, v in plan.get("kv_router_kwargs", {}).items()
            if k != "router_policy_config" and k in config and config.get(k) != v
        }
        detail["kwarg_diffs"] = diffs
        ok = ok and not diffs
    dep.record("policy_evidence", ok, **detail)


def reset(dep: Deployment, mapping: dict[int, int], block: int, verify: bool) -> None:
    def idle(samples):
        running = metric_sum(samples, "vllm:num_requests_running") or 0.0
        waiting = metric_sum(samples, "vllm:num_requests_waiting") or 0.0
        return running == 0 and waiting == 0

    for _ in range(60):
        if all(idle(s) for s in dep.worker_metrics()):
            break
        time.sleep(1.0)
    else:
        dep.record(
            "reset_idle", False, note="workers still running requests; flush would fail"
        )

    def flush_all() -> list:
        out = []
        for worker in dep.workers:
            status, body = http(
                "POST", f"{worker}/engine/flush_cache", body={}, timeout=60.0
            )
            text = body.decode(errors="replace")
            out.append({"worker": worker, "status": status, "body": text[:200]})
            if status != 200 or '"ok"' not in text:
                dep.record("flush", False, responses=out)
        return out

    def cached(result) -> int | None:
        return ((result["usage"] or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )

    proof = []
    if verify:
        # Per worker: a sentinel prompt, its repeat (positive control: the engine reports the
        # prefix as cached), a flush, and the repeat again (must report zero cached tokens).
        # The engine-reported cached_tokens is primary; vLLM's prefix-hit counter is recorded too.
        for instance, index in sorted(mapping.items()):
            sentinel = dep.tokens(f"cold-{instance}", 32 * block)
            worker = dep.workers[index]

            def hits():
                return metric_sum(
                    metrics(f"{worker}/metrics"), "vllm:prefix_cache_hits"
                )

            dep.complete(sentinel, 1, pin=instance)
            h0 = hits()
            control = dep.complete(sentinel, 1, pin=instance)
            h1 = hits()
            status, _ = http(
                "POST", f"{worker}/engine/flush_cache", body={}, timeout=60.0
            )
            after = dep.complete(sentinel, 1, pin=instance)
            h2 = hits()
            entry = {
                "instance": instance,
                "worker": index,
                "flush_status": status,
                "control_cached_tokens": cached(control),
                "after_flush_cached_tokens": cached(after),
                "control_prefix_hits": None if h0 is None or h1 is None else h1 - h0,
                "after_flush_prefix_hits": None
                if h1 is None or h2 is None
                else h2 - h1,
            }
            entry["ok"] = (
                status == 200
                and (entry["control_cached_tokens"] or 0) >= len(sentinel) - block
                and entry["after_flush_cached_tokens"] == 0
                and entry["after_flush_prefix_hits"] in (None, 0)
            )
            proof.append(entry)
    responses = flush_all()
    dep.record("cold_reset", all(p["ok"] for p in proof), proof=proof, flush=responses)


def snapshot(dep: Deployment, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in [("frontend", dep.frontend)] + [
        (f"worker{i}", w) for i, w in enumerate(dep.workers)
    ]:
        status, body = http("GET", f"{url}/metrics", timeout=10.0)
        (out_dir / f"{tag}.{name}.prom").write_bytes(body if status == 200 else b"")
    dep.record("snapshot", True, dir=str(out_dir), tag=tag)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["wait", "smoke", "reset", "snapshot"])
    parser.add_argument("--frontend", required=True, help="http://host:port")
    parser.add_argument(
        "--worker", action="append", default=[], help="host:system_port (repeat)"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--namespace", default=None)
    parser.add_argument(
        "--expect", type=int, default=None, help="generate instances expected"
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--expect-affinity",
        action="store_true",
        help="require the prefix-extension probe to land on the same worker",
    )
    parser.add_argument("--policy-plan", type=Path, default=None)
    parser.add_argument("--config-dump", type=Path, default=None)
    parser.add_argument(
        "--no-verify", action="store_true", help="reset: flush without the proof"
    )
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--tag", default="snap")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    dep = Deployment(args)
    dep.report.update(
        command=args.command, started_utc=time.strftime("%FT%TZ", time.gmtime())
    )
    status = 0
    try:
        if args.command == "wait":
            wait(dep, args.expect or len(dep.workers), args.timeout)
        elif args.command == "smoke":
            mapping = map_instances(dep)
            check_forced_osl(dep, mapping)
            plan = json.loads(args.policy_plan.read_text()) if args.policy_plan else {}
            spec = plan.get("spec") or {}
            # Only default@defaults is known to send a prefix extension to the prefix's worker.
            expect_affinity = args.expect_affinity or (
                spec.get("type") == "dynamo-default-cost-fn"
                and not spec.get("parameters")
                and not spec.get("router_config")
            )
            if plan.get("router_mode") == "round_robin":
                # Round-robin has no KV indexer, so the frontend applies no KV events.
                dep.record("kv_events_reach_router", True, not_applicable="round_robin")
            else:
                check_kv_events_and_reuse(dep, expect_affinity, args.block_size)
            check_policy_evidence(dep, args.policy_plan, args.config_dump)
        elif args.command == "reset":
            mapping = map_instances(dep)
            reset(dep, mapping, args.block_size, verify=not args.no_verify)
        elif args.command == "snapshot":
            snapshot(dep, args.snapshot_dir or args.out.parent, args.tag)
    except (CheckFailed, OSError, json.JSONDecodeError) as exc:
        dep.report["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        status = 1
    dep.report["ok"] = status == 0
    dep.report["ended_utc"] = time.strftime("%FT%TZ", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dep.report, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "command": args.command,
                "ok": dep.report["ok"],
                "error": dep.report.get("error"),
            }
        )
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
