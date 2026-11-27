#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned Weka replay and score held-out native predictions."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

TRACE_SHA256 = "954157040cbae47fef3bec70532bdb3406c75ed894e207e502461d70fa06639c"


def micros(seconds):
    value = round(seconds * 1_000_000)
    if value < 0:
        raise ValueError("negative Weka timestamp")
    return value


def prepare(trace, output):
    digest = hashlib.sha256()
    by_fold = defaultdict(list)
    requests = []
    counts = Counter()
    with trace.open("rb") as source:
        for episode, line in enumerate(source, 1):
            digest.update(line)
            play = json.loads(line)
            fold = (
                "held_out"
                if hashlib.sha256(play["id"].encode()).digest()[0] % 4 == 0
                else "train"
            )
            counts[fold + "_episodes"] += 1
            strands = {None: 0}
            events = []
            for node in play["requests"]:
                if node["type"] == "subagent":
                    strand = strands.setdefault(node["agent_id"], len(strands))
                    turns = node["requests"]
                elif node["type"] in ("s", "n"):
                    strand, turns = 0, [node]
                else:
                    raise ValueError(f"unsupported Weka request type {node['type']}")
                if strand >= 1000:
                    raise ValueError("too many strands for one episode")
                session = episode * 1000 + strand
                for turn in turns:
                    arrival = micros(turn["t"])
                    duration = micros(turn["api_time"])
                    ttft = (
                        micros(turn["ttft"]) if turn.get("ttft") is not None else None
                    )
                    isl, osl, model = turn["in"], turn["out"], turn["model"]
                    if duration <= 0 or isl <= 0 or osl < 0:
                        counts["invalid_lengths_or_duration"] += 1
                        continue
                    first = (
                        arrival + ttft if ttft is not None and ttft < duration else None
                    )
                    end = arrival + duration
                    request = len(requests) + 1
                    requests.append(
                        dict(
                            id=request,
                            episode=episode,
                            session=session,
                            fold=fold,
                            strand="subagent" if strand else "main",
                            model=model,
                            input_tokens=isl,
                            output_tokens=osl,
                            arrival_us=arrival,
                            first_us=first,
                            end_us=end,
                        )
                    )
                    # Completions at an arrival's timestamp are available to that arrival.
                    events.append(
                        (
                            arrival,
                            1,
                            request,
                            f"P\t{request}\t{model}\t{session}\t{isl}\n",
                        )
                    )
                    events.append((end, 0, request, f"C\t{request}\t{osl}\n"))
            by_fold[fold].extend(event[-1] for event in sorted(events))
    if digest.hexdigest() != TRACE_SHA256:
        raise ValueError("expected the pinned public Weka trace")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "requests.tsv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(requests[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(requests)
    with (output / "events.tsv").open("x") as stream:
        stream.writelines(by_fold["train"])
        stream.write("FIT\n")
        stream.writelines(by_fold["held_out"])
    counts.update(row["fold"] for row in requests)
    (output / "manifest.json").write_text(
        json.dumps(dict(trace_sha256=TRACE_SHA256, counts=counts), indent=2) + "\n"
    )


def predictions(path):
    with path.open() as stream:
        result = {}
        for row in csv.DictReader(stream, delimiter="\t"):
            rid = int(row["id"])
            tokens = float(row["predicted_tokens"])
            if rid in result or not math.isfinite(tokens):
                raise ValueError(f"duplicate or nonfinite prediction for request {rid}")
            result[rid] = (tokens, int(row["backend_priority"]))
    return result


def competing_pairs(rows):
    groups = defaultdict(list)
    for r in rows:
        # Each recorded episode has its own relative clock. A request belongs to
        # the waiting proxy only while its observed first output is in the future.
        if r["first_us"] is not None and r["first_us"] > r["arrival_us"]:
            groups[r["episode"], r["model"]].append(r)
    for requests in groups.values():
        waiting = []
        for request in sorted(requests, key=lambda r: (r["arrival_us"], r["id"])):
            waiting = [
                older for older in waiting if older["first_us"] > request["arrival_us"]
            ]
            for older in waiting:
                if older["output_tokens"] == request["output_tokens"]:
                    continue
                short, long = sorted((older, request), key=lambda r: r["output_tokens"])
                yield short["id"], long["id"]
            waiting.append(request)


def priority_order(predictions, short, long):
    short_priority = predictions[short][1]
    long_priority = predictions[long][1]
    if short_priority > long_priority:
        return "correct"
    if short_priority < long_priority:
        return "inverted"
    return "tied"


def score(requests, predictions_by_arm):
    with requests.open() as stream:
        rows = []
        for r in csv.DictReader(stream, delimiter="\t"):
            if r["fold"] == "held_out":
                for key in ("id", "episode", "output_tokens", "arrival_us"):
                    r[key] = int(r[key])
                r["first_us"] = int(r["first_us"]) if r["first_us"] else None
                rows.append(r)
    ids = {r["id"] for r in rows}
    if (
        not rows
        or len(ids) != len(rows)
        or any(p.keys() != ids for p in predictions_by_arm.values())
    ):
        raise ValueError("predictions must cover exactly the unique held-out requests")
    orders = ("correct", "inverted", "tied")
    transitions = {before: dict.fromkeys(orders, 0) for before in orders}
    counts = {arm: Counter() for arm in predictions_by_arm}
    pairs = 0
    for short, long in competing_pairs(rows):
        before = priority_order(predictions_by_arm["no_history"], short, long)
        after = priority_order(predictions_by_arm["history"], short, long)
        counts["no_history"][before] += 1
        counts["history"][after] += 1
        transitions[before][after] += 1
        pairs += 1
    result = {}
    for arm, p in predictions_by_arm.items():
        inversion = counts[arm]["inverted"]
        ties = counts[arm]["tied"]
        result[arm] = dict(
            length_mae_tokens=mean(
                abs(p[r["id"]][0] - r["output_tokens"]) for r in rows
            ),
            priority_inversions=inversion,
            priority_ties=ties,
            correctly_ordered=counts[arm]["correct"],
            inversion_fraction=inversion / pairs if pairs else None,
            inversion_fraction_excluding_ties=inversion / (pairs - ties)
            if pairs > ties
            else None,
        )
    return dict(
        held_out_requests=len(rows),
        requests_with_first_token=sum(r["first_us"] is not None for r in rows),
        competing_pairs=pairs,
        arms=result,
        no_history_to_history=transitions,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("trace", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    score_cmd = commands.add_parser("score")
    for name in ("requests", "history", "no-history", "output"):
        score_cmd.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.trace, args.output)
    else:
        result = score(
            args.requests,
            {
                "history": predictions(args.history),
                "no_history": predictions(args.no_history),
            },
        )
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
