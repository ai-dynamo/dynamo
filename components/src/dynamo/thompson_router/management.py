# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HTTP management server for learner state persistence, config hot-reload, and reset.

All tunable params are applied directly to the live KvThompsonRouter instance.

Diagnostic endpoints:
  GET /decisions      — last N routing decisions with overlap/score details
  GET /decisions/summary — agreement rate, worker distribution, overlap stats
"""

import collections
import logging
import time

from aiohttp import web

from dynamo.thompson_router.router import (
    TUNABLE_ROUTER_PARAMS,
    KvThompsonRouter,
    RoutingDecision,
)

logger = logging.getLogger(__name__)

MAX_DECISION_HISTORY = 50_000


class RouterManagementServer:
    """HTTP server for learner state persistence, config hot-reload, and reset."""

    def __init__(self, router: KvThompsonRouter, port: int = 8084):
        self._router = router
        self._port = port
        self._runner: web.AppRunner | None = None
        self._decisions: collections.deque[dict] = collections.deque(
            maxlen=MAX_DECISION_HISTORY
        )

    def record_decision(
        self, decision: RoutingDecision, hints: dict, elapsed_ms: float, tokens_out: int
    ) -> None:
        """Record a routing decision for diagnostic inspection."""
        self._decisions.append(
            {
                "ts": time.time(),
                "prefix_id": decision.prefix_id,
                "chosen": decision.chosen,
                "native_pick": decision.native_pick,
                "agreed": decision.chosen == decision.native_pick,
                "osl": hints.get("osl", 0),
                "iat": hints.get("iat", 0),
                "reuse_budget": hints.get("reuse_budget", 0),
                "tokens_in": hints.get("tokens_in", 0),
                "latency_sensitivity": hints.get("latency_sensitivity", 2.0),
                "tokens_out": tokens_out,
                "elapsed_ms": round(elapsed_ms, 1),
                "workers": decision.worker_details,
            }
        )

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/state", self._get_state)
        app.router.add_post("/state", self._load_state)
        app.router.add_post("/state/reset", self._reset_state)
        app.router.add_get("/config", self._get_config)
        app.router.add_post("/config", self._set_config)
        app.router.add_get("/decisions", self._get_decisions)
        app.router.add_get("/decisions/summary", self._get_decisions_summary)
        app.router.add_get("/metrics", self._get_metrics)
        app.router.add_post("/metrics/reset", self._reset_metrics)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        logger.info("RouterManagementServer listening on :%d", self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _health(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response(
            {
                "status": "ok",
                "router_type": "kv_thompson",
                "workers": r.beta_learner.worker_ids,
            }
        )

    async def _get_state(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response(
            {
                "beta_learner": r.beta_learner.to_dict(),
            }
        )

    async def _load_state(self, request: web.Request) -> web.Response:
        r = self._router
        data = await request.json()
        if "beta_learner" in data:
            r.beta_learner.load_state(data["beta_learner"])
        logger.info("Learner state loaded via HTTP")
        return web.json_response({"status": "loaded"})

    async def _reset_state(self, _request: web.Request) -> web.Response:
        r = self._router
        r.beta_learner.reset_all()
        r.latency_tracker.reset()
        self._decisions.clear()
        logger.info("Learner state reset to pristine via HTTP (decisions cleared)")
        return web.json_response({"status": "reset"})

    async def _get_config(self, _request: web.Request) -> web.Response:
        r = self._router
        return web.json_response(
            {
                "lambda_ranking": r.lambda_ranking,
                "lambda_stickiness": r.lambda_stickiness,
                "w_cache": r.w_cache,
                "w_queue": r.w_queue,
                "w_osl_load": r.w_osl_load,
                "w_sensitivity": r.w_sensitivity,
                "alpha_reuse": r.alpha_reuse,
                "sticky_bonus": r.sticky_bonus,
                "stickiness_overlap_cap": r.stickiness_overlap_cap,
                "epsilon": r.epsilon,
                "beta_decay": r.beta_learner.decay,
            }
        )

    async def _set_config(self, request: web.Request) -> web.Response:
        """Hot-reload tunable params directly onto the live router instance."""
        data = await request.json()
        r = self._router
        applied = {}

        # All 9 tunable params + beta_decay
        param_map = {
            "lambda_ranking": "lambda_ranking",
            "lambda_stickiness": "lambda_stickiness",
            "w_cache": "w_cache",
            "w_queue": "w_queue",
            "w_osl_load": "w_osl_load",
            "w_sensitivity": "w_sensitivity",
            "alpha_reuse": "alpha_reuse",
            "sticky_bonus": "sticky_bonus",
            "stickiness_overlap_cap": "stickiness_overlap_cap",
            "epsilon": "epsilon",
        }
        for key, attr in param_map.items():
            if key in data:
                val = float(data[key])
                setattr(r, attr, val)
                applied[key] = val

        if "beta_decay" in data:
            r.beta_learner.decay = float(data["beta_decay"])
            applied["beta_decay"] = r.beta_learner.decay

        logger.info("Router config hot-reloaded via HTTP: %s", applied)
        return web.json_response({"status": "applied", "params": applied})

    async def _get_decisions(self, request: web.Request) -> web.Response:
        """Return last N routing decisions with full details."""
        n = int(request.query.get("n", "20"))
        recent = list(self._decisions)[-n:]
        return web.json_response({"count": len(recent), "decisions": recent})

    async def _get_metrics(self, _request: web.Request) -> web.Response:
        """Return rolling instrumentation stats."""
        return web.json_response(self._router.stats.snapshot())

    async def _reset_metrics(self, _request: web.Request) -> web.Response:
        """Reset rolling stats counters."""
        self._router.stats.reset()
        logger.info("Router metrics reset via HTTP")
        return web.json_response({"status": "reset"})

    async def _get_decisions_summary(self, _request: web.Request) -> web.Response:
        """Aggregate stats over recent decisions."""
        decisions = list(self._decisions)
        if not decisions:
            return web.json_response({"error": "no decisions recorded yet"})

        total = len(decisions)
        agreed = sum(1 for d in decisions if d["agreed"])

        worker_counts: dict[int, int] = {}
        prefix_workers: dict[str, set] = {}
        overlap_when_chosen: list[float] = []
        overlap_when_not: list[float] = []

        for d in decisions:
            wid = d["chosen"]
            worker_counts[wid] = worker_counts.get(wid, 0) + 1

            pid = d["prefix_id"]
            if pid:
                prefix_workers.setdefault(pid, set()).add(wid)

            for w in d.get("workers", []):
                if w["id"] == d["chosen"]:
                    overlap_when_chosen.append(w["kv_overlap"])
                else:
                    overlap_when_not.append(w["kv_overlap"])

        prefixes_with_scatter = sum(
            1 for ws in prefix_workers.values() if len(ws) > 1
        )

        avg_overlap_chosen = (
            sum(overlap_when_chosen) / len(overlap_when_chosen)
            if overlap_when_chosen
            else 0
        )
        avg_overlap_others = (
            sum(overlap_when_not) / len(overlap_when_not)
            if overlap_when_not
            else 0
        )

        return web.json_response(
            {
                "total_decisions": total,
                "agreed_with_native": agreed,
                "agreement_rate": round(agreed / total, 4) if total else 0,
                "worker_distribution": {
                    str(k): v for k, v in sorted(worker_counts.items())
                },
                "unique_prefixes": len(prefix_workers),
                "prefixes_scattered_across_workers": prefixes_with_scatter,
                "prefix_stickiness_rate": round(
                    1.0 - prefixes_with_scatter / max(1, len(prefix_workers)), 4
                ),
                "avg_kv_overlap_chosen_worker": round(avg_overlap_chosen, 4),
                "avg_kv_overlap_other_workers": round(avg_overlap_others, 4),
                "overlap_advantage": round(
                    avg_overlap_chosen - avg_overlap_others, 4
                ),
            }
        )
