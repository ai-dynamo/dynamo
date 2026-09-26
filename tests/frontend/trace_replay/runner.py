# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay fixture cases against a live frontend and grade every reply.

Each trajectory replays its turns in order, like an agent would; trajectories
run concurrently up to `concurrency` in-flight requests. Every case is sent to
each configured endpoint and mode, graded against the trace (`compare.grade`),
and cross-checked against the chat-stream reply of the same case.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from tests.frontend.trace_replay import compare
from tests.frontend.trace_replay.endpoints import (
    ENDPOINTS,
    JsonDict,
    Turn,
    build_request,
    normalize,
    supports,
)

logger = logging.getLogger(__name__)

REFERENCE = ("chat", "stream")
_METRIC_LINE = re.compile(r"^(dynamo_frontend_[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")
_METRIC_NAMES = re.compile(r"requests_total|disconnect|cancel")
_DISCONNECT_AFTER_EVENTS = 3
_EXAMPLES_PER_CATEGORY = 5
_PROGRESS_EVERY = 500


def load_jsonl(path: Path) -> list[JsonDict]:
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


@dataclass
class Fixtures:
    trajectories: dict[str, JsonDict]
    cases: list[JsonDict]
    manifest: JsonDict

    @classmethod
    def load(cls, directory: Path) -> Fixtures:
        trajectories = {
            t["trajectory"]: t for t in load_jsonl(directory / "trajectories.jsonl")
        }
        manifest = json.loads((directory / "build_stats.json").read_text())
        return cls(trajectories, load_jsonl(directory / "cases.jsonl"), manifest)


@dataclass
class RunConfig:
    base_url: str
    model: str
    endpoints: list[str] = field(default_factory=lambda: list(ENDPOINTS))
    modes: list[str] = field(default_factory=lambda: ["stream", "unary"])
    concurrency: int = 8
    raw: bool = False  # the frontend serves this model without parsers
    variants: set[str] | None = None
    max_trajectories: int | None = None
    disconnect_every: int = 0  # drop every Nth full chat stream after a few events
    timeout_s: float = 600.0
    raw_dir: Path | None = None  # keep raw replies of mismatches for inspection


@dataclass
class Result:
    key: str
    variant: str
    trajectory: str
    endpoint: str
    mode: str
    status: int
    tier: str = "mismatch"
    categories: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    events: int = 0
    completion_tokens: int | None = None
    detail: JsonDict = field(default_factory=dict)


class Replay:
    def __init__(self, fixtures: Fixtures, config: RunConfig):
        self.fixtures = fixtures
        self.config = config
        self.markup: list[str] = fixtures.manifest["markup"]
        self.results: list[Result] = []
        self.turns: dict[tuple[str, str, str], Turn] = {}
        self.disconnects = 0
        self._raw_examples: collections.Counter = collections.Counter()
        self._semaphore = asyncio.Semaphore(config.concurrency)

    def selected_cases(self) -> dict[str, list[JsonDict]]:
        by_trajectory: dict[str, list[JsonDict]] = collections.defaultdict(list)
        for case in self.fixtures.cases:
            if self.config.variants and case["variant"] not in self.config.variants:
                continue
            by_trajectory[case["trajectory"]].append(case)
        chosen = list(by_trajectory)[: self.config.max_trajectories]
        return {trajectory: by_trajectory[trajectory] for trajectory in chosen}

    async def run(self) -> JsonDict:
        limits = httpx.Limits(max_connections=self.config.concurrency * 2)
        timeout = httpx.Timeout(self.config.timeout_s, connect=30.0)
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            metrics_before = await self._metrics(client)
            started = time.monotonic()
            await asyncio.gather(
                *(
                    self._replay_trajectory(client, cases)
                    for cases in self.selected_cases().values()
                )
            )
            wall_s = time.monotonic() - started
            metrics_after = await self._metrics(client)
        return self.report(wall_s, _metric_deltas(metrics_before, metrics_after))

    async def _replay_trajectory(
        self, client: httpx.AsyncClient, cases: list[JsonDict]
    ) -> None:
        full_seen = 0
        for case in cases:
            for endpoint in self.config.endpoints:
                if not supports(endpoint, case):
                    continue
                for mode in self.config.modes:
                    async with self._semaphore:
                        result = await self._run_case(client, case, endpoint, mode)
                    self.results.append(result)
                    if len(self.results) % _PROGRESS_EVERY == 0:
                        mismatches = sum(r.tier == "mismatch" for r in self.results)
                        logger.info(
                            "%d requests done, %d mismatches",
                            len(self.results),
                            mismatches,
                        )
            if case["variant"] == "full":
                full_seen += 1
                every = self.config.disconnect_every
                if every and full_seen % every == 0:
                    async with self._semaphore:
                        await self._disconnect(client, case)

    async def _run_case(
        self, client: httpx.AsyncClient, case: JsonDict, endpoint: str, mode: str
    ) -> Result:
        trajectory = self.fixtures.trajectories[case["trajectory"]]
        stream = mode == "stream"
        body = build_request(endpoint, self.config.model, trajectory, case, stream)
        url = self.config.base_url + ENDPOINTS[endpoint]
        result = Result(
            case["key"], case["variant"], case["trajectory"], endpoint, mode, 0
        )
        started = time.monotonic()
        try:
            payload: Any
            if stream:
                lines: list[str] = []
                async with client.stream("POST", url, json=body) as response:
                    result.status = response.status_code
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if result.ttft_ms is None and line.startswith("data:"):
                                result.ttft_ms = (time.monotonic() - started) * 1000
                            lines.append(line)
                    else:
                        result.detail["error"] = (await response.aread()).decode()[:500]
                payload = lines
            else:
                response = await client.post(url, json=body)
                result.status = response.status_code
                payload = response.json() if response.status_code == 200 else None
                if payload is None:
                    result.detail["error"] = response.text[:500]
        except (httpx.HTTPError, json.JSONDecodeError) as err:
            result.detail["error"] = f"{type(err).__name__}: {err}"[:500]
            payload = None
        result.latency_ms = (time.monotonic() - started) * 1000
        if result.status != 200 or payload is None:
            result.categories = ["http_error"]
            return result
        try:
            turn = normalize(endpoint, stream, payload)
        except (KeyError, TypeError, ValueError) as err:
            result.categories = ["unparseable_reply"]
            result.detail["error"] = f"{type(err).__name__}: {err}"[:500]
            return result
        graded = compare.grade(turn, case, self.markup, raw=self.config.raw)
        result.tier = graded.tier
        result.categories = graded.categories
        result.detail.update(graded.detail)
        result.violations = turn.violations
        result.events = turn.events
        result.completion_tokens = turn.completion_tokens
        self.turns[(case["key"], endpoint, mode)] = turn
        # Keep every mismatch, but only a few examples of each stream violation.
        signatures = [
            f"{endpoint}/{mode}:{v}"
            for v in set(result.violations)
            if self._raw_examples[f"{endpoint}/{mode}:{v}"] < _EXAMPLES_PER_CATEGORY
        ]
        if self.config.raw_dir is not None and (
            result.tier == "mismatch" or signatures
        ):
            self._raw_examples.update(signatures)
            self._save_raw(result, body, payload)
        return result

    def _save_raw(self, result: Result, body: JsonDict, payload: Any) -> None:
        assert self.config.raw_dir is not None
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        name = f"{result.key.replace(':', '_')}.{result.endpoint}.{result.mode}"
        reply = "\n".join(payload) if isinstance(payload, list) else json.dumps(payload)
        (self.config.raw_dir / f"{name}.reply.txt").write_text(reply)
        request = {
            k: v for k, v in body.items() if k not in ("messages", "input", "tools")
        }
        (self.config.raw_dir / f"{name}.request.json").write_text(json.dumps(request))

    async def _disconnect(self, client: httpx.AsyncClient, case: JsonDict) -> None:
        """Open a chat stream and hang up after a few events, as a cancelling client does."""
        trajectory = self.fixtures.trajectories[case["trajectory"]]
        body = build_request("chat", self.config.model, trajectory, case, True)
        events = 0
        try:
            async with client.stream(
                "POST", self.config.base_url + ENDPOINTS["chat"], json=body
            ) as response:
                async for line in response.aiter_lines():
                    events += line.startswith("data:")
                    if events >= _DISCONNECT_AFTER_EVENTS:
                        break
        except httpx.HTTPError as err:
            logger.warning("disconnect probe for %s failed: %s", case["key"], err)
            return
        self.disconnects += 1

    async def _metrics(self, client: httpx.AsyncClient) -> dict[str, float]:
        try:
            response = await client.get(self.config.base_url + "/metrics")
        except httpx.HTTPError:
            return {}
        values: dict[str, float] = {}
        for line in response.text.splitlines():
            match = _METRIC_LINE.match(line)
            if match and _METRIC_NAMES.search(match.group(1)):
                values[match.group(1) + (match.group(2) or "")] = float(match.group(3))
        return values

    def cross_checks(self) -> JsonDict:
        """Replies of one case must agree across modes and endpoints."""
        counts: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        examples: dict[str, list[JsonDict]] = collections.defaultdict(list)
        keys = {key for key, _, _ in self.turns}
        for key in sorted(keys):
            reference = self.turns.get((key, *REFERENCE))
            if reference is None:
                continue
            for endpoint in self.config.endpoints:
                for mode in self.config.modes:
                    if (endpoint, mode) == REFERENCE:
                        continue
                    other = self.turns.get((key, endpoint, mode))
                    if other is None:
                        continue
                    cell = f"{endpoint}/{mode} vs chat/stream"
                    differences = compare.same_turn(reference, other)
                    counts[cell]["compared"] += 1
                    for difference in differences:
                        counts[cell][difference] += 1
                    if differences and len(examples[cell]) < _EXAMPLES_PER_CATEGORY:
                        examples[cell].append({"key": key, "differs_in": differences})
        return {"counts": {k: dict(v) for k, v in counts.items()}, "examples": examples}

    def report(self, wall_s: float, metrics_delta: dict[str, float]) -> JsonDict:
        cells: dict[str, JsonDict] = {}
        examples: dict[str, list[JsonDict]] = collections.defaultdict(list)
        by_variant: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for result in self.results:
            cell = cells.setdefault(
                f"{result.endpoint}/{result.mode}",
                {
                    "requests": 0,
                    "tiers": collections.Counter(),
                    "categories": collections.Counter(),
                    "violations": collections.Counter(),
                    "ttft_ms": [],
                    "latency_ms": [],
                },
            )
            cell["requests"] += 1
            cell["tiers"][result.tier] += 1
            cell["categories"].update(result.categories)
            cell["violations"].update(set(result.violations))
            if result.ttft_ms is not None:
                cell["ttft_ms"].append(result.ttft_ms)
            cell["latency_ms"].append(result.latency_ms)
            by_variant[result.variant][result.tier] += 1
            for category in result.categories:
                bucket = f"{result.endpoint}/{result.mode}:{category}"
                if len(examples[bucket]) < _EXAMPLES_PER_CATEGORY:
                    examples[bucket].append(
                        {"key": result.key, "detail": result.detail}
                    )
        for cell in cells.values():
            cell["tiers"] = dict(cell["tiers"])
            cell["categories"] = dict(cell["categories"].most_common())
            cell["violations"] = dict(cell["violations"].most_common())
            for name in ("ttft_ms", "latency_ms"):
                samples = sorted(cell.pop(name))
                cell[name] = _percentiles(samples)
        trajectories = {result.trajectory for result in self.results}
        return {
            "model": self.config.model,
            "raw": self.config.raw,
            "requests": len(self.results),
            "trajectories": len(trajectories),
            "cases": len({result.key for result in self.results}),
            "wall_s": round(wall_s, 1),
            "concurrency": self.config.concurrency,
            "disconnect_probes": self.disconnects,
            "fixtures": self.fixtures.manifest.get("stats", {}),
            "cells": cells,
            "by_variant": {k: dict(v) for k, v in by_variant.items()},
            "cross": self.cross_checks(),
            "metrics_delta": metrics_delta,
            "examples": examples,
        }


def _percentiles(samples: list[float]) -> JsonDict:
    if not samples:
        return {}
    return {
        "p50": round(statistics.median(samples), 1),
        "p99": round(samples[min(len(samples) - 1, int(len(samples) * 0.99))], 1),
        "max": round(samples[-1], 1),
    }


def _metric_deltas(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    return {
        name: after[name] - before.get(name, 0.0)
        for name in sorted(after)
        if after[name] != before.get(name, 0.0)
    }


def render_markdown(report: JsonDict) -> str:
    lines = [
        f"# Trace replay: {report['model']}{' (parsers off)' if report['raw'] else ''}",
        "",
        f"{report['requests']} requests, {report['cases']} cases, "
        f"{report['trajectories']} trajectories, concurrency {report['concurrency']}, "
        f"{report['wall_s']} s.",
        "",
        "| endpoint/mode | requests | "
        + " | ".join(compare.TIERS)
        + " | TTFT p50/p99 ms | latency p50/p99 ms |",
        "|---|---:|" + "---:|" * len(compare.TIERS) + "---|---|",
    ]
    for name, cell in sorted(report["cells"].items()):
        tiers = " | ".join(str(cell["tiers"].get(tier, 0)) for tier in compare.TIERS)
        ttft = cell["ttft_ms"]
        latency = cell["latency_ms"]
        lines.append(
            f"| {name} | {cell['requests']} | {tiers} | "
            f"{ttft.get('p50', '-')}/{ttft.get('p99', '-')} | "
            f"{latency.get('p50', '-')}/{latency.get('p99', '-')} |"
        )
    lines += ["", "## Mismatch categories", ""]
    for name, cell in sorted(report["cells"].items()):
        if cell["categories"]:
            lines.append(
                f"- {name}: "
                + ", ".join(f"{k}={v}" for k, v in cell["categories"].items())
            )
    lines += ["", "## Stream protocol violations", ""]
    for name, cell in sorted(report["cells"].items()):
        if cell["violations"]:
            lines.append(
                f"- {name}: "
                + ", ".join(f"{k}={v}" for k, v in cell["violations"].items())
            )
    lines += ["", "## Tiers by variant", ""]
    for variant, tiers in sorted(report["by_variant"].items()):
        lines.append(
            f"- {variant}: " + ", ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
        )
    lines += ["", "## Cross-checks (disagreements with chat/stream)", ""]
    for cell, counts in sorted(report["cross"]["counts"].items()):
        lines.append(f"- {cell}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    lines += ["", "## Frontend metric deltas", ""]
    lines += [
        f"- `{name}`: {value:g}" for name, value in report["metrics_delta"].items()
    ]
    lines.append(f"- disconnect probes sent: {report['disconnect_probes']}")
    return "\n".join(lines) + "\n"


def write_outputs(replay: Replay, report: JsonDict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    (out_dir / "report.md").write_text(render_markdown(report))
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for result in replay.results:
            record = asdict(result)
            if result.tier not in ("mismatch", "ws_args"):
                record.pop("detail")
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
