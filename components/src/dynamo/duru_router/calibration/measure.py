# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure solo accepted work through the same worker configuration used to serve."""

import hashlib
import json
import math
import random
import statistics
import tempfile
import time
from pathlib import Path

from .plan import METRIC, Plan, atomic_json
from .predict import check_tables


def decode_window(events, prompt_length, warmup_tokens, expected_tokens):
    """Average service intervals for actual token advances within the decode window.

    Events contain (cumulative token count, scheduled_at, end). elapsed_s sums
    execution intervals after the first usable observation, excluding idle gaps.
    Exclude first-token/prefill startup and the final completion event. No CUDA
    synchronization or per-token RPC is introduced into the serving path.
    """
    if not events or events[-1][0] != expected_tokens:
        raise ValueError("Request did not produce its required output length")
    if any(
        not math.isfinite(begin) or not math.isfinite(end) or end <= begin
        for _, begin, end in events
    ):
        raise ValueError("Decode service intervals must be finite and positive")
    if any(n2 <= n1 or t2 < t1 for (n1, _, t1), (n2, _, t2) in zip(events, events[1:])):
        raise ValueError("Nonmonotonic token counts or timestamps")
    usable = [event for event in events if warmup_tokens < event[0] < expected_tokens]
    if len(usable) < 2:
        raise ValueError("Insufficient streaming observations; increase decode-steps")
    (first, _, begin), (last, _, end) = usable[0], usable[-1]
    steps = last - first
    if steps < max(1, (expected_tokens - warmup_tokens - 2) // 2) or end <= begin:
        raise ValueError("Too much of the decode window was coalesced or unobserved")
    elapsed = math.fsum(end - begin for _, begin, end in usable[1:])
    return dict(
        wall_ms=1000 * elapsed / steps,
        K=prompt_length + (first + last - 1) / 2,
        first_count=first,
        last_count=last,
        steps=steps,
        elapsed_s=elapsed,
        events=events,
    )


def fit(points, plan: Plan, contract):
    prefill = {
        (row["P"], row["D"]): row["wall_ms"]
        for row in points
        if row["phase"] == "prefill"
    }
    if set(prefill) != {(p, d) for p in plan.P for d in plan.D}:
        raise ValueError("Incomplete prefill grid")
    decode_rows = [row for row in points if row["phase"] == "decode"]
    if {row["K"] for row in decode_rows} != set(plan.K):
        raise ValueError("Incomplete decode grid")
    result = dict(
        schema_version=1,
        timing_metric=METRIC,
        contract=contract,
        max_prefill=plan.max_prefill,
        prefill=dict(
            P=plan.P,
            D=plan.D,
            P_discontinuities=plan.P_discontinuities,
            wall_ms=[[prefill[p, d] for d in plan.D] for p in plan.P],
        ),
        decode=dict(
            grid=sorted(
                [
                    dict(K=row["measured_K"], wall_ms=row["wall_ms"])
                    for row in decode_rows
                ],
                key=lambda row: row["K"],
            )
        ),
    )
    check_tables(result)
    return result


def _valid_point(row):
    if not isinstance(row, dict) or row.get("phase") not in ("prefill", "decode"):
        return False
    fields = ("wall_ms", "measured_K") if row["phase"] == "decode" else ("wall_ms",)
    return all(
        type(row.get(field)) in (int, float)
        and math.isfinite(row[field])
        and row[field] > 0
        for field in fields
    )


def measure(
    directory,
    contract,
    plan: Plan | dict,
    submit,
    drain,
    ledger,
    reset,
    *,
    resume=False,
    vocab_size=32000,
):
    """Run serial probes, checkpointing each point; no held-out validation.

    submit(prompt_ids, output_count, request_id, cache_salt) returns
    output_tokens and cached_tokens. drain() flushes the accepted-output JSONL
    ledger. reset() clears prefix caches while idle. The caller owns the worker.
    Resume keeps valid checkpointed points and remeasures invalid points.
    """
    if not isinstance(plan, Plan):
        plan = Plan.from_dict(plan)
    if plan.timing_metric != METRIC or contract.get("timing_metric") != METRIC:
        raise ValueError("Calibration requires accepted engine-service timing")
    if type(vocab_size) is not int or vocab_size < 512:
        raise ValueError("Provide the serving tokenizer vocabulary size (at least 512)")
    if plan.repeats < 1 or plan.warmups < 1:
        raise ValueError("Need at least one warmup and measured repetition")
    directory, ledger = Path(directory), Path(ledger)
    identity = dict(
        contract=contract,
        plan=plan.to_dict(),
        vocab_size=vocab_size,
        decode_timing="accepted_step_intervals_v1",
    )
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    points = []
    if directory.exists():
        if not resume:
            raise FileExistsError(directory)
        saved = json.loads((directory / "checkpoint.json").read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(
                "Cannot resume a changed measurement method, plan, vocabulary, or runtime contract"
            )
        points = saved["points"]
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Publish the run only once its identity and initial checkpoint exist.
        with tempfile.TemporaryDirectory(
            dir=directory.parent, prefix=f".{directory.name}-"
        ) as staging_name:
            staging = Path(staging_name)
            atomic_json(staging / "plan.json", plan.to_dict())
            atomic_json(staging / "contract.json", contract)
            atomic_json(
                staging / "checkpoint.json", dict(fingerprint=fingerprint, points=[])
            )
            staging.rename(directory)

    def save(candidate):
        atomic_json(
            directory / "checkpoint.json",
            dict(fingerprint=fingerprint, points=candidate),
        )

    def key(row):
        return row["phase"], row.get("P"), row.get("D"), row.get("K")

    planned = {key(case) for case in plan.cases}
    points = [row for row in points if _valid_point(row) and key(row) in planned]
    done = {key(row) for row in points}
    offset = ledger.stat().st_size
    attempt, serial = time.time_ns(), 0

    def request(prompt, count, salt):
        nonlocal serial, offset
        if len(prompt) + count > plan.cap:
            raise ValueError("Calibration request exceeds context limit")
        serial += 1
        rid = f"duru-calibration-{attempt}-{serial}"
        response = submit(prompt, count, rid, salt)
        drain()
        steps = []
        with ledger.open() as stream:
            stream.seek(offset)
            for line in stream:
                record = json.loads(line)
                if record["event"] == "accepted_output":
                    steps.extend(
                        s
                        for s in record["steps"]
                        if s["client_request_id"] == rid and s["accepted"]
                    )
            offset = stream.tell()
        if (
            response["output_tokens"] != count
            or sum(s["generated_tokens"] for s in steps) != count
        ):
            raise ValueError("Accepted-token ledger does not match request output")
        return response, steps

    current_d, prefix = None, []
    for case in plan.cases:
        if key(case) in done:
            continue
        phase = case["phase"]
        if phase == "prefill":
            d, p, count = case["D"], case["P"], 1
            salt = f"duru-calibration-{attempt}-D{d}"
            if d != current_d:
                reset()
                current_d = d
                rng = random.Random(f"prefix-{d}")
                prefix = [rng.randrange(100, vocab_size - 2) for _ in range(d)]
                if d:
                    request(prefix + [vocab_size - 1], 1, salt)
        elif phase == "decode":
            reset()
            current_d, prefix = None, []
            d, p, count = 0, case["K"], plan.decode_tokens
            salt = f"duru-calibration-{attempt}-K{p}"
        else:
            raise ValueError("Unknown calibration phase")
        samples = []
        for rep in range(plan.warmups + plan.repeats):
            rng = random.Random(f"{case}-{rep}")
            suffix = [rng.randrange(100, vocab_size - 2) for _ in range(p)]
            response, steps = request(prefix + suffix, count, salt)
            if phase == "prefill":
                if (
                    response["cached_tokens"] != d
                    or len(steps) != 1
                    or steps[0]["p"] != p
                    or steps[0]["d"] != d
                ):
                    raise ValueError(
                        "Observed prefill/cache shape differs from planned point"
                    )
                sample = dict(
                    wall_ms=1000 * (steps[0]["end"] - steps[0]["scheduled_at"])
                )
            else:
                tokens, events = 0, []
                for step in sorted(steps, key=lambda s: s["end"]):
                    tokens += step["generated_tokens"]
                    if step["generated_tokens"]:
                        events.append((tokens, step["scheduled_at"], step["end"]))
                sample = decode_window(events, p, plan.decode_warmup_tokens, count)
            if not math.isfinite(sample["wall_ms"]) or sample["wall_ms"] <= 0:
                raise ValueError("Calibration timing must be finite and positive")
            with (directory / "samples.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        dict(
                            case=case,
                            repeat=rep,
                            attempt=attempt,
                            sample=sample,
                            steps=steps,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
            if rep >= plan.warmups:
                samples.append(sample)
        row = dict(case, wall_ms=statistics.median(s["wall_ms"] for s in samples))
        if phase == "decode":
            row["measured_K"] = statistics.mean(s["K"] for s in samples)
        if not _valid_point(row):
            raise ValueError(
                "Calibration point measurements must be finite and positive"
            )
        candidate = [*points, row]
        save(candidate)
        points = candidate
        done.add(key(row))
    result = fit(points, plan, contract)
    atomic_json(directory / "calibration.json", result)
    return result


def load(path, expected_contract):
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema_version") != 1 or artifact.get("timing_metric") != METRIC:
        raise ValueError("Unsupported calibration artifact or timing definition")
    if artifact["contract"] != expected_contract:
        raise ValueError("Calibration and serving runtime contracts differ")
    if type(artifact.get("max_prefill")) is not int or artifact["max_prefill"] < 1:
        raise ValueError("Calibration must specify a positive prefill token budget")
    check_tables(artifact)
    return artifact
