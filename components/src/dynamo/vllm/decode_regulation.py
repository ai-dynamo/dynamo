# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-step decode regulation inside the vLLM scheduler.

Under agent load, interactive requests miss their deadline because every decode
step is slow (a large batch with long contexts), not because their first token
is late. The lever is which *running* requests decode in each step.

Policy, once per ``schedule()`` call, over running requests in the decode phase:

  interactive   never skipped. With DYN_DECODE_REGULATION_PRIORITY_MIN set:
                Dynamo request priority (nvext.agent_hints.priority) >= that
                value; every lower-priority request is regulated. Unset: a
                request without agent context, or whose session id starts with
                DYN_DECODE_REGULATION_INTERACTIVE_PREFIX
  forced        regulated, skipped K - 1 steps in a row; decodes this step
                (progress bound)
  opportunistic the other regulated requests; first-fit while the predicted
                step time stays inside T0 + rho * (tau - T0), most-skipped
                first, then cheapest context

Nothing is skipped while no interactive request is decoding, or when the whole
batch already fits the tightest interactive target, so the policy is free
outside overload. The prefill chunk
expected in the same step is added to every prediction.

Mechanism (vLLM v0.31.0): a skipped request keeps its KV and its place in
``self.running``; the skip sets ``request.next_decode_eligible_step`` to
``current_step + 2`` before the parent ``schedule()`` increments ``current_step``,
so the parent's running loop passes over it for exactly this step. Requests are
never removed from ``self.running`` (that breaks preemption victim choice and
the running-request cap).

This module has no vLLM import, so the policy and the skip mixin are unit-tested
against a fake scheduler; ``decode_regulation_scheduler.DecodeRegulationScheduler``
binds the mixin to Dynamo's ``InstrumentedScheduler``.

Environment (prefix DYN_DECODE_REGULATION_):
  MODE              off | observe | class_blind (default off). class_blind
                    regulates; observe makes the same decisions but never
                    skips: the summary, metrics and trace show what regulation
                    would do, and the trace calibrates COEFFS
                    (decode_regulation_fit.py)
  COEFFS            a,b,c of T(n, L) = a + b*n + c*L ms
  PREFILL_COEFFS    q,p: a step with P > 0 prefill tokens adds q + p*P ms
                    (default 0,0: prefill ignored)
  TAU_INTERACTIVE_MS  fixed per-token target when DEADLINE is unset
  RHO               share of the slack opportunistic decodes may use
  K                 a request decodes at least once every K steps
  MAX_OPP_CTX       coarse guard: opportunistic context per step (0 = off)
  PRIORITY_MIN      requests with Dynamo priority >= this are interactive
  INTERACTIVE_PREFIX  session-id prefix marking interactive sessions when
                    PRIORITY_MIN is unset
  FAILSAFE_STEPS    fail open (stock vLLM scheduling) after this many
                    consecutive decode-only steps with the step-time bias
                    pinned at its 0.5 / 2.0 clamp (0 = never)
  DEADLINE          ttft_ms,per_token_ms: interactive requests get a
                    deadline-aware target (BROS-style residual budget):
                    margin x (arrival + ttft_ms + per_token_ms x max_tokens -
                    now) / (max_tokens - generated). The step target is the
                    tightest; unset: the fixed TAU_INTERACTIVE_MS
  SLACK_MARGIN      the margin above (default 0.9)
  TRACE             JSONL path for per-step records
  LOG_S             seconds between summary log lines
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

MODES = ("off", "observe", "class_blind")
EXTRA_ARGS_KEY = "dynamo_decode_regulation"
# Prometheus family (dynamo.prometheus_names.name_prefix.COMPONENT + suffix),
# written from the EngineCore process through PROMETHEUS_MULTIPROC_DIR and
# forwarded by the worker's /metrics like the embedding-cache connector's.
DECODE_REGULATION_METRIC_PREFIX = "dynamo_component_decode_regulation"


@dataclass
class DecodeRegulationConfig:
    mode: str = "off"
    coeffs: tuple = (6.4013, 0.227066, 0.000016055)
    prefill_coeffs: tuple = (0.0, 0.0)
    tau_interactive_ms: float = 18.48
    rho: float = 0.8
    k: int = 4
    max_opp_ctx: int = 0
    interactive_prefix: str = "interactive-"
    priority_min: Optional[int] = None
    failsafe_steps: int = 200
    deadline: Optional[tuple] = None  # (ttft_ms, per_token_ms)
    slack_margin: float = 0.9
    trace_path: str = ""
    log_s: float = 10.0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> "DecodeRegulationConfig":
        env = os.environ if env is None else env
        c = cls()
        mode = env.get("DYN_DECODE_REGULATION_MODE", "off").strip().lower() or "off"
        if mode in ("1", "true", "on"):
            mode = "class_blind"
        if mode not in MODES:
            raise ValueError(
                f"DYN_DECODE_REGULATION_MODE={mode!r}: expected one of {MODES}"
            )
        c.mode = mode
        if env.get("DYN_DECODE_REGULATION_COEFFS"):
            parts = [float(x) for x in env["DYN_DECODE_REGULATION_COEFFS"].split(",")]
            if len(parts) != 3:
                raise ValueError(
                    "DYN_DECODE_REGULATION_COEFFS needs three numbers a,b,c"
                )
            c.coeffs = tuple(parts)
        if env.get("DYN_DECODE_REGULATION_PREFILL_COEFFS"):
            parts = [
                float(x) for x in env["DYN_DECODE_REGULATION_PREFILL_COEFFS"].split(",")
            ]
            if len(parts) != 2:
                raise ValueError(
                    "DYN_DECODE_REGULATION_PREFILL_COEFFS needs two numbers q,p"
                )
            c.prefill_coeffs = tuple(parts)
        c.tau_interactive_ms = float(
            env.get("DYN_DECODE_REGULATION_TAU_INTERACTIVE_MS", c.tau_interactive_ms)
        )
        c.rho = float(env.get("DYN_DECODE_REGULATION_RHO", c.rho))
        c.k = int(env.get("DYN_DECODE_REGULATION_K", c.k))
        if c.k < 1:
            raise ValueError("DYN_DECODE_REGULATION_K must be >= 1")
        c.max_opp_ctx = int(env.get("DYN_DECODE_REGULATION_MAX_OPP_CTX", c.max_opp_ctx))
        c.interactive_prefix = env.get(
            "DYN_DECODE_REGULATION_INTERACTIVE_PREFIX", c.interactive_prefix
        )
        if env.get("DYN_DECODE_REGULATION_PRIORITY_MIN", "").strip():
            c.priority_min = int(env["DYN_DECODE_REGULATION_PRIORITY_MIN"])
        c.failsafe_steps = int(
            env.get("DYN_DECODE_REGULATION_FAILSAFE_STEPS", c.failsafe_steps)
        )
        if env.get("DYN_DECODE_REGULATION_DEADLINE", "").strip():
            parts = [float(x) for x in env["DYN_DECODE_REGULATION_DEADLINE"].split(",")]
            if len(parts) != 2:
                raise ValueError(
                    "DYN_DECODE_REGULATION_DEADLINE needs two numbers ttft_ms,per_token_ms"
                )
            c.deadline = tuple(parts)
        c.slack_margin = float(
            env.get("DYN_DECODE_REGULATION_SLACK_MARGIN", c.slack_margin)
        )
        c.trace_path = env.get("DYN_DECODE_REGULATION_TRACE", "")
        c.log_s = float(env.get("DYN_DECODE_REGULATION_LOG_S", c.log_s))
        return c


# ---------------------------------------------------------------------------
# Identity: handler side (agent context -> SamplingParams.extra_args)
# ---------------------------------------------------------------------------


def request_priority(request: Mapping[str, Any]) -> int:
    """Dynamo semantic priority (higher is more important; unset is 0)."""
    routing = request.get("routing")
    p = routing.get("priority") if isinstance(routing, Mapping) else None
    try:
        return int(p) if p is not None else 0
    except (TypeError, ValueError):
        return 0


def identity_from_request(
    request: Mapping[str, Any],
    interactive_prefix: str,
    priority_min: Optional[int] = None,
) -> dict:
    """{"interactive": bool} for the engine.

    With ``priority_min``, a request is interactive iff its Dynamo priority is
    at least ``priority_min``. Without it, a request is interactive unless it
    carries an agent session id that does not start with ``interactive_prefix``."""
    if priority_min is not None:
        return {"interactive": request_priority(request) >= priority_min}
    ctx = request.get("agent_context")
    sid = ctx.get("session_id") if isinstance(ctx, Mapping) else None
    if not isinstance(sid, str) or not sid.strip():
        return {"interactive": True}
    prefix = interactive_prefix
    return {"interactive": bool(prefix) and sid.strip().startswith(prefix)}


def attach_identity(
    sampling_params: Any, request: Mapping[str, Any], cfg: DecodeRegulationConfig
) -> None:
    """Copy the request's regulation class into ``sampling_params.extra_args``.

    ``extra_args`` reaches the engine ``Request`` unchanged in vLLM v0.31.0 (the
    KV-connector transfer params ride the same field)."""
    if not cfg.enabled:
        return
    extra = sampling_params.extra_args
    extra = dict(extra) if isinstance(extra, dict) else {}
    extra[EXTRA_ARGS_KEY] = identity_from_request(
        request, cfg.interactive_prefix, cfg.priority_min
    )
    sampling_params.extra_args = extra


# ---------------------------------------------------------------------------
# Policy (pure)
# ---------------------------------------------------------------------------


@dataclass
class Cand:
    """A running request in the decode phase, as the policy sees it."""

    req_id: str
    ctx: int
    interactive: bool = True
    decoded: int = 0
    skip_run: int = 0
    # Per-token time this request can still afford (ms): its remaining deadline
    # over its remaining tokens (BROS-style residual budget). None: the fixed
    # class target applies.
    tau_ms: Optional[float] = None


@dataclass
class Decision:
    skip: set = field(default_factory=set)
    engaged: bool = False
    tau_ms: float = 0.0
    pred_all_ms: float = 0.0
    pred_kept_ms: float = 0.0
    n_forced: int = 0
    n_opp: int = 0
    n_managed: int = 0  # regulated (non-interactive) decode-phase requests
    n_interactive: int = 0  # decode-phase requests never skipped


class DecodeRegulationPolicy:
    """Chooses, each step, which decode-phase requests skip; keeps cross-step state."""

    def __init__(self, cfg: DecodeRegulationConfig) -> None:
        self.cfg = cfg
        self.bias = 1.0  # EWMA of actual / predicted on decode-only steps
        self._skip_run: dict = {}

    def predict(self, n: int, ctx: int) -> float:
        a, b, c = self.cfg.coeffs
        return self.bias * max(0.0, a + b * n + c * ctx)

    def predict_prefill(self, tokens: int) -> float:
        """Extra step time for a prefill chunk of ``tokens`` scheduled with the decodes."""
        if tokens <= 0:
            return 0.0
        q, p = self.cfg.prefill_coeffs
        return self.bias * max(0.0, q + p * tokens)

    def observe(
        self, predicted_ms: float, actual_ms: float, alpha: float = 0.05
    ) -> None:
        if predicted_ms <= 0 or actual_ms <= 0:
            return
        r = actual_ms / (predicted_ms / self.bias)
        self.bias = min(2.0, max(0.5, (1 - alpha) * self.bias + alpha * r))

    def decide(self, cands: Iterable[Cand], prefill_tokens: int = 0) -> Decision:
        cands = list(cands)
        live = {c.req_id for c in cands}
        self._skip_run = {r: v for r, v in self._skip_run.items() if r in live}
        for c in cands:
            c.skip_run = self._skip_run.get(c.req_id, 0)
        d = Decision()
        if not self.cfg.enabled or not cands:
            return self._commit(d, cands)

        managed = [c for c in cands if not c.interactive]
        d.n_managed = len(managed)
        d.n_interactive = len(cands) - len(managed)
        if not managed or not d.n_interactive:
            # nothing to regulate, or no interactive request to protect: every
            # skip would only slow the regulated requests down
            return self._commit(d, cands)
        d.tau_ms = self.cfg.tau_interactive_ms
        residual = [c.tau_ms for c in cands if c.interactive and c.tau_ms is not None]
        if residual:
            d.tau_ms = min(residual)
        extra = self.predict_prefill(prefill_tokens)
        d.pred_all_ms = self.predict(len(cands), sum(c.ctx for c in cands)) + extra
        if d.pred_all_ms <= d.tau_ms:
            d.pred_kept_ms = d.pred_all_ms
            return self._commit(d, cands)

        d.engaged = True
        forced = {c.req_id for c in managed if c.skip_run >= self.cfg.k - 1}
        base_ids = {c.req_id for c in cands if c.interactive} | forced
        n = len(base_ids)
        ctx = sum(c.ctx for c in cands if c.req_id in base_ids)
        d.n_forced = len(forced)

        t0 = self.predict(n, ctx) + extra
        limit = t0 + self.cfg.rho * max(0.0, d.tau_ms - t0)
        opp_ctx = 0
        # most-skipped first, then cheapest context
        order = sorted(
            (c for c in managed if c.req_id not in base_ids),
            key=lambda c: (-c.skip_run, c.ctx, c.req_id),
        )
        # predict(n + 1, ctx + c.ctx) + extra <= limit, unrolled: this loop
        # runs once per regulated request per step
        a, b, cc = self.cfg.coeffs
        bias, max_opp = self.bias, self.cfg.max_opp_ctx
        for c in order:
            guard_ok = not max_opp or opp_ctx + c.ctx <= max_opp
            t = bias * max(0.0, a + b * (n + 1) + cc * (ctx + c.ctx)) + extra
            if guard_ok and t <= limit:
                n, ctx, opp_ctx = n + 1, ctx + c.ctx, opp_ctx + c.ctx
                d.n_opp += 1
            else:
                d.skip.add(c.req_id)
        d.pred_kept_ms = self.predict(n, ctx) + extra
        return self._commit(d, cands)

    def _commit(self, d: Decision, cands: list) -> Decision:
        for c in cands:
            self._skip_run[c.req_id] = c.skip_run + 1 if c.req_id in d.skip else 0
        return d

    def skip_run(self, req_id: str) -> int:
        return self._skip_run.get(req_id, 0)


# ---------------------------------------------------------------------------
# Scheduler mixin
# ---------------------------------------------------------------------------


@dataclass
class _Static:
    """What a request's ``Cand`` needs that does not change between steps."""

    interactive: bool = True
    max_tokens: int = 0  # 0: no deadline-aware target
    arrival_ms: float = 0.0


def _static_from_request(req: Any, cfg: Optional[DecodeRegulationConfig]) -> _Static:
    sp = getattr(req, "sampling_params", None)
    extra = getattr(sp, "extra_args", None) if sp is not None else None
    ident = extra.get(EXTRA_ARGS_KEY) if isinstance(extra, dict) else None
    if isinstance(ident, dict) and not ident.get("interactive", True):
        return _Static(interactive=False)
    st = _Static()
    if cfg is not None and cfg.deadline is not None:
        max_tokens = getattr(sp, "max_tokens", None) if sp is not None else None
        arrival = getattr(req, "arrival_time", None)
        if max_tokens and arrival is not None:
            st.max_tokens, st.arrival_ms = int(max_tokens), arrival * 1e3
    return st


def _cand(
    req: Any, st: _Static, cfg: Optional[DecodeRegulationConfig], now_ms: float
) -> Optional[Cand]:
    if (
        getattr(req, "is_prefill_chunk", False)
        or req.num_computed_tokens < req.num_prompt_tokens
    ):
        return None
    c = Cand(
        req.request_id,
        req.num_computed_tokens,
        st.interactive,
        req.num_output_tokens,
    )
    if st.max_tokens and cfg is not None and cfg.deadline is not None:
        ttft_ms, per_token_ms = cfg.deadline
        n = max(st.max_tokens, req.num_output_tokens + 1)
        due_ms = st.arrival_ms + ttft_ms + per_token_ms * n
        c.tau_ms = cfg.slack_margin * (due_ms - now_ms) / (n - req.num_output_tokens)
    return c


def cand_from_request(
    req: Any, cfg: Optional[DecodeRegulationConfig] = None, now: Optional[float] = None
) -> Optional[Cand]:
    """A ``Cand`` for a running vLLM request in the decode phase, else None.

    With ``cfg.deadline`` set, an interactive request also gets ``tau_ms``: the
    per-token time it can still afford against its end-to-end deadline, taken
    at ``max_tokens`` (``now``: wall-clock seconds, as ``req.arrival_time``)."""
    now_ms = (time.time() if now is None else now) * 1e3
    return _cand(req, _static_from_request(req, cfg), cfg, now_ms)


class DecodeRegulationMixin:
    """Applies ``DecodeRegulationPolicy`` to a vLLM (Async)Scheduler; mix in before the scheduler.

    Expects the host to provide ``running``, ``current_step``, ``schedule()`` and
    ``update_from_output()`` with vLLM v0.31.0 semantics."""

    def _reg_init(
        self,
        cfg: Optional[DecodeRegulationConfig] = None,
        *,
        unsupported: str = "",
        model_name: str = "",
        dp_rank: int = 0,
        metrics_registry: Any = None,
    ) -> None:
        self._reg_cfg = cfg or DecodeRegulationConfig.from_env()
        if unsupported and self._reg_cfg.enabled:
            logger.warning("decode_regulation disabled: %s", unsupported)
            self._reg_cfg.mode = "off"
        self._reg_metrics = (
            RegulationMetrics.create(model_name, dp_rank, metrics_registry)
            if self._reg_cfg.enabled
            else None
        )
        self._reg_policy = DecodeRegulationPolicy(self._reg_cfg)
        self._reg_pending: deque = (
            deque()
        )  # (output id, t_sched, pred_ms, decode_only, decision)
        self._reg_last_update = 0.0
        self._reg_trace = (
            open(self._reg_cfg.trace_path, "a")
            if (self._reg_cfg.enabled and self._reg_cfg.trace_path)
            else None
        )
        self._reg_stats = _Stats()
        self._reg_static: dict = {}  # request id -> _Static
        self._reg_pinned = 0  # consecutive decode-only steps with bias at a clamp
        self._reg_next_log = time.monotonic() + self._reg_cfg.log_s
        if self._reg_cfg.enabled:
            logger.info(
                "decode_regulation mode=%s coeffs=%s tau=%.2f ms rho=%.2f K=%d guard=%d "
                "deadline=%s margin=%.2f",
                self._reg_cfg.mode,
                self._reg_cfg.coeffs,
                self._reg_cfg.tau_interactive_ms,
                self._reg_cfg.rho,
                self._reg_cfg.k,
                self._reg_cfg.max_opp_ctx,
                self._reg_cfg.deadline,
                self._reg_cfg.slack_margin,
            )

    def _reg_apply(self) -> Optional[Decision]:
        if not self._reg_cfg.enabled or getattr(self, "_bench_active", False):
            return None
        t_start = time.perf_counter()
        by_id = {}
        cands = []
        cfg, static = self._reg_cfg, self._reg_static
        now_ms = time.time() * 1e3
        for req in self.running:
            st = static.get(req.request_id)
            if st is None:
                st = static[req.request_id] = _static_from_request(req, cfg)
            c = _cand(req, st, cfg, now_ms)
            if c is not None:
                cands.append(c)
                by_id[c.req_id] = req
        if len(static) > 2 * len(self.running) + 64:
            live = {req.request_id for req in self.running}
            for rid in [r for r in static if r not in live]:
                del static[rid]
        d = self._reg_policy.decide(cands, self._reg_prefill_estimate())
        if d.skip and self._reg_cfg.mode != "observe":
            # schedule() increments current_step first, so this skips exactly one step.
            nxt = self.current_step + 2
            for rid in d.skip:
                req = by_id[rid]
                req.next_decode_eligible_step = max(req.next_decode_eligible_step, nxt)
        elapsed = time.perf_counter() - t_start
        self._reg_stats.on_overhead(elapsed)
        if self._reg_metrics is not None:
            self._reg_metrics.on_decision(d, elapsed)
        return d

    def _reg_prefill_estimate(self) -> int:
        """Prefill tokens the coming step will likely carry: running prefill chunks,
        then waiting prompts, capped at the step's token budget."""
        if not any(self._reg_cfg.prefill_coeffs):
            return 0
        budget = getattr(self, "max_num_scheduled_tokens", 0) or 8192
        tokens = 0
        for req in self.running:
            if req.num_computed_tokens < req.num_prompt_tokens:
                tokens += req.num_prompt_tokens - req.num_computed_tokens
                if tokens >= budget:
                    return budget
        for req in self.waiting:
            tokens += req.num_prompt_tokens - req.num_computed_tokens
            if tokens >= budget:
                return budget
        return tokens

    def _reg_after_schedule(self, output: Any, d: Optional[Decision]) -> None:
        if d is None or output.total_num_scheduled_tokens <= 0:
            return
        running = {r.request_id: r for r in self.running}
        n = ctx = 0
        decode_only = True
        for rid, k in output.num_scheduled_tokens.items():
            if k != 1:
                decode_only = False
                continue
            r = running.get(rid)
            if r is not None:
                n, ctx = n + 1, ctx + r.num_computed_tokens
        pred = self._reg_policy.predict(n, ctx)
        self._reg_pending.append(
            (id(output), time.monotonic(), pred, decode_only, d, n, ctx)
        )
        self._reg_stats.on_decision(d, self._reg_policy)

    def _reg_on_update(self, output: Any) -> None:
        if not self._reg_pending:
            return
        now = time.monotonic()
        if all(p[0] != id(output) for p in self._reg_pending):
            return  # an empty output (never recorded)
        while self._reg_pending[0][0] != id(output):
            self._reg_pending.popleft()
        _, t_sched, pred, decode_only, d, n, ctx = self._reg_pending.popleft()
        actual = (now - max(t_sched, self._reg_last_update)) * 1e3
        self._reg_last_update = now
        if decode_only and n:
            self._reg_policy.observe(pred, actual)
            if self._reg_metrics is not None:
                self._reg_metrics.bias.set(self._reg_policy.bias)
            self._reg_stats.on_step(pred, actual)
            self._reg_failsafe()
        if self._reg_trace is not None:
            self._reg_trace.write(
                json.dumps(
                    {
                        "t": round(now, 4),
                        "n": n,
                        "ctx": ctx,
                        "decode_only": decode_only,
                        "engaged": d.engaged,
                        "skip": len(d.skip),
                        "forced": d.n_forced,
                        "opp": d.n_opp,
                        "tau": d.tau_ms,
                        "pred_all": round(d.pred_all_ms, 3),
                        "pred": round(pred, 3),
                        "actual": round(actual, 3),
                        "bias": round(self._reg_policy.bias, 4),
                    }
                )
                + "\n"
            )
        if now >= self._reg_next_log:
            self._reg_next_log = now + self._reg_cfg.log_s
            if self._reg_trace is not None:
                self._reg_trace.flush()
            logger.info(
                "decode_regulation %s",
                self._reg_stats.summary_and_reset(self._reg_policy.bias),
            )

    def _reg_failsafe(self) -> None:
        """Fail open when the step-time model is persistently wrong.

        The bias corrects a model that is off by up to 2x either way; pinned at
        the clamp, predictions (and so skips) can no longer be trusted, e.g.
        coefficients fitted for another GPU or model. Skips stop and the
        scheduler behaves as stock vLLM until restart."""
        limit = self._reg_cfg.failsafe_steps
        if limit <= 0:
            return
        b = self._reg_policy.bias
        self._reg_pinned = self._reg_pinned + 1 if b <= 0.5 or b >= 2.0 else 0
        if self._reg_pinned >= limit:
            logger.warning(
                "decode_regulation disabled: step-time bias pinned at %.2f for %d steps; "
                "refit DYN_DECODE_REGULATION_COEFFS for this GPU and model (fit_step_model.py)",
                b,
                self._reg_pinned,
            )
            self._reg_cfg.mode = "off"
            self._reg_stats.disabled = True
            if self._reg_metrics is not None:
                self._reg_metrics.enabled.set(0)

    def _reg_close(self) -> None:
        if self._reg_trace is not None:
            self._reg_trace.close()
            self._reg_trace = None

    # --- overrides -------------------------------------------------------

    def schedule(self, *args, **kwargs):
        d = self._reg_apply()
        output = super().schedule(*args, **kwargs)
        self._reg_after_schedule(output, d)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        if self._reg_cfg.enabled:
            self._reg_on_update(scheduler_output)
        return result

    def shutdown(self) -> None:
        self._reg_close()
        super().shutdown()


class RegulationMetrics:
    """Prometheus series for decode regulation, per (model, dp_rank).

    Counted in request-steps (one request in one scheduler step):
    ``skipped_total / regulated_total`` is the share of low-priority decode
    work deferred, ``engaged_steps_total / steps_total`` how often the worker
    ran overloaded. ``enabled`` drops to 0 when the fail-open guard trips."""

    def __init__(self, model_name: str, dp_rank: int, registry: Any) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge

        registry = registry if registry is not None else CollectorRegistry()
        names = ["model", "dp_rank"]  # dynamo_* hierarchy labels are injected
        values = {"model": model_name, "dp_rank": str(dp_rank)}
        p = DECODE_REGULATION_METRIC_PREFIX

        def counter(name: str, doc: str):
            return Counter(f"{p}_{name}", doc, names, registry=registry).labels(
                **values
            )

        def gauge(name: str, doc: str):
            return Gauge(
                f"{p}_{name}",
                doc,
                names,
                registry=registry,
                multiprocess_mode="mostrecent",
            ).labels(**values)

        self.steps = counter("steps_total", "Scheduler steps evaluated.")
        self.engaged = counter(
            "engaged_steps_total", "Steps where the batch broke the target."
        )
        self.interactive = counter(
            "interactive_total", "Decode request-steps never regulated."
        )
        self.regulated = counter(
            "regulated_total", "Decode request-steps eligible for regulation."
        )
        self.skipped = counter("skipped_total", "Decode request-steps skipped.")
        self.forced = counter(
            "forced_total", "Request-steps run by the progress bound (K)."
        )
        self.decision = counter(
            "decision_seconds_total", "Time spent choosing the step's decodes."
        )
        self.enabled = gauge("enabled", "1 while regulating, 0 after fail-open.")
        self.bias = gauge("step_time_bias", "Actual / predicted step time (EWMA).")
        self.enabled.set(1)
        self.bias.set(1.0)

    @classmethod
    def create(
        cls, model_name: str, dp_rank: int, registry: Any = None
    ) -> Optional["RegulationMetrics"]:
        try:
            return cls(model_name, dp_rank, registry)
        except ImportError:
            logger.debug("prometheus_client unavailable: no decode regulation metrics")
            return None

    def on_decision(self, d: Decision, seconds: float) -> None:
        self.steps.inc()
        self.decision.inc(seconds)
        self.interactive.inc(d.n_interactive)
        self.regulated.inc(d.n_managed)
        if d.engaged:
            self.engaged.inc()
            self.skipped.inc(len(d.skip))
            self.forced.inc(d.n_forced)


class _Stats:
    """Counters between summary log lines (skipped requests are invisible to FPM)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = self.engaged = self.skipped = self.forced = 0
        self.regulated = 0  # regulated request-steps seen while engaged
        self.seen_interactive = self.seen_regulated = 0  # request-steps, all steps
        self.max_skip_run = 0
        self.err = []
        self.overhead_us = []
        self.disabled = getattr(self, "disabled", False)

    def on_decision(self, d: Decision, policy: DecodeRegulationPolicy) -> None:
        self.steps += 1
        self.engaged += d.engaged
        self.seen_interactive += d.n_interactive
        self.seen_regulated += d.n_managed
        if d.engaged:
            self.regulated += d.n_managed
        self.skipped += len(d.skip)
        self.forced += d.n_forced
        for rid in d.skip:
            self.max_skip_run = max(self.max_skip_run, policy.skip_run(rid))

    def on_overhead(self, seconds: float) -> None:
        self.overhead_us.append(seconds * 1e6)

    def on_step(self, pred: float, actual: float) -> None:
        self.err.append(abs(pred - actual) / actual)

    def summary_and_reset(self, bias: float) -> str:
        e = sorted(self.err)
        mape = f"{e[len(e) // 2]:.1%}" if e else "-"
        share = f"{self.skipped / self.regulated:.1%}" if self.regulated else "-"
        o = sorted(self.overhead_us)
        over = f"{o[len(o) // 2]:.0f}/{o[99 * len(o) // 100]:.0f}us" if o else "-"
        s = (
            f"steps={self.steps} interactive={self.seen_interactive} "
            f"regulated={self.seen_regulated} engaged={self.engaged} skipped={self.skipped} "
            f"skip_share={share} forced={self.forced} "
            f"max_skip_run={self.max_skip_run} pred_err_p50={mape} bias={bias:.3f} "
            f"decide_p50/p99={over}" + (" DISABLED" if self.disabled else "")
        )
        self.reset()
        return s
