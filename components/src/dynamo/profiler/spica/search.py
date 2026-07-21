# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The smart sweep: SearchSpace -> ranked candidates (best-first).

One Vizier study per ``deployment_mode`` branch searches the parallel-config + knob
space (backend is one of the knobs); each suggestion is unrolled, translated to a
deployment, evaluated by replay, scored, and fed back to the optimizer; feasible
candidates are ranked across branches.

Each round is a **barrier**: the study suggests trials until ``per_round`` unique full
samples complete successfully (ask), they are evaluated **in parallel across worker
processes** (``SweepConfig.parallel_evals``; ``<= 1`` runs sequentially), then their
scores are fed back (tell). Exact duplicates use a run-local result cache and trigger
replacement asks. Vizier ask/tell stay on the main process — workers run only the pure
unroll->deploy->replay->score and never touch the study (the Vizier trial handle never
crosses the process boundary). The load-predictor winner is resolved once and injected
into every unroll.

``evaluator`` and ``sampler_factory`` are injectable so the loop is unit-testable
without real replay / Vizier (use ``parallel_evals=1`` to avoid spawning processes).
"""

from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from tqdm import tqdm

from .config import Candidate, OptimizationGoal, SmartSearchConfig
from .deploy import build_deployment
from .evaluator import ReplayEvaluator
from .kv_estimate import _load_memory_estimator, resolve_backend_version
from .kv_load import InfeasibleKVCapacity, resolve_kv_load
from .load_predictor_sweep import LoadPredictorResult, sweep_load_predictor
from .planner import filter_scaling_policies, scaling_fields
from .sample import unroll_sample
from .sampler import BranchSampler, Suggestion, make_branch_sampler
from .score import is_feasible, make_candidate, pareto_front, rank
from .search_space import BranchSpace, enumerate_branches


class _Evaluator(Protocol):
    def evaluate(
        self, plan: Any, *, concurrency_override: int | None = None
    ) -> dict[str, float]:
        ...


# Result of evaluating one suggestion (no Vizier here): (candidate|None, observe_metrics|None,
# outcome, reason). observe_metrics is the dict fed to sampler.observe — {"objective": score}
# for a single-objective sweep, or {obj_name: raw_value, ...} under a pareto goal. outcome in
# {"feasible","infeasible","failed"}. Both "failed" (replay error) and "infeasible" (over
# gpu_budget) carry a reason and no metrics -> the loop tells the sampler observe_infeasible
# for them (a gated trial is never fed back as a high score). "unsupported" is decided on the
# main process before evaluation and never reaches the worker.
_EvalResult = tuple[Candidate | None, dict[str, float] | None, str, str]


def _freeze(value: Any) -> Any:
    """Convert a nested suggestion/context value into a stable hashable key."""
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze(asdict(value))
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _suggestion_cache_key(suggestion: Suggestion, context: Any) -> Any:
    """A run-local full-sample key; parallel equality alone is not a cache hit."""
    return (context, _freeze(suggestion.selection), _freeze(suggestion.parallel_config))


def _evaluate_one(
    selection: dict[str, Any],
    parallel_config: Any,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    load_predictor: LoadPredictorResult,
    evaluator: _Evaluator,
) -> _EvalResult:
    """Pure unroll -> deploy -> replay -> score for one (already backend-supported)
    suggestion. No Vizier, no shared mutable state -> safe to run in a worker process.
    """
    try:
        sample = unroll_sample(
            search_space=config.search_space,
            selection=selection,
            parallel_config=parallel_config,
            load_predictor=load_predictor,
        )
        backend_version = resolve_backend_version(
            config.search_space.hardware_sku, selection["backend"]
        )
        # The resolved perf-model version is part of the evaluated contract. Keep it
        # on the candidate so downstream artifact generation cannot independently
        # select a different backend version.
        sample["backend_version"] = backend_version
        concurrency = config.workload.concurrency
        if "kv_load_ratio" in selection:
            ratio = float(selection["kv_load_ratio"])
            resolution = resolve_kv_load(
                sample,
                workload=config.workload,
                parallel_config=parallel_config,
                ratio=ratio,
                backend_version=backend_version,
            )
            concurrency = resolution.concurrency
            sample["kv_load_ratio"] = resolution.ratio
            sample["kv_load_concurrency_capacity"] = resolution.concurrency_capacity
            load_role = "decode" if sample["deployment_mode"] == "disagg" else "agg"
            sample["kv_load_capacity_tokens"] = resolution.role_capacity_tokens[
                load_role
            ]
            for role, tokens in resolution.role_capacity_tokens.items():
                sample[f"{role}_kv_capacity_tokens"] = tokens
        if concurrency is not None:
            # Preserve the concrete load on every candidate, including a fixed absolute
            # concurrency and one derived from kv_load_ratio.
            sample["concurrency"] = concurrency
        plan = build_deployment(
            sample,
            backend_version=backend_version,
            optimization_target=goal.target.planner_optimization_target,
            planner_sla=goal.sla,
        )
    except InfeasibleKVCapacity as exc:
        return None, None, "infeasible", f"candidate KV capacity infeasible: {exc}"
    except Exception as exc:
        return (
            None,
            None,
            "failed",
            f"candidate build failed: {type(exc).__name__}: {exc}",
        )
    try:
        report = evaluator.evaluate(plan, concurrency_override=concurrency)
    except Exception as exc:  # one candidate failing must not abort the sweep
        return None, None, "failed", f"replay failed: {type(exc).__name__}: {exc}"
    if not is_feasible(int(sample["used_gpus"]), config.search_space.gpu_budget):
        # Over gpu_budget: report as infeasible to the optimizer (observe_infeasible, not
        # observe(metrics)) so a high score doesn't steer the sampler into the infeasible
        # region. The trial is gated, not ranked.
        return (
            None,
            None,
            "infeasible",
            f"over gpu_budget: used_gpus={int(sample['used_gpus'])} > gpu_budget={config.search_space.gpu_budget}",
        )
    if goal.is_pareto:
        candidate = make_candidate(
            sample,
            report,
            goal.target,
            pareto_objectives=goal.resolved_pareto_objectives,
        )
        # Pareto objectives are reported raw (each metric carries its own MAXIMIZE/MINIMIZE goal).
        observe_metrics: dict[str, float] = dict(candidate.objectives or {})
    else:
        candidate = make_candidate(sample, report, goal.target)
        observe_metrics = {
            "objective": candidate.score
        }  # single metric, pre-signed higher-is-better
    return candidate, observe_metrics, "feasible", ""


# Worker-process plumbing: the shared read-only state (config/goal/load_predictor/
# evaluator) is sent once via the pool initializer and stashed as a module global, so
# each task only ships the per-suggestion (selection, parallel_config).
_WORKER_CTX: dict[str, Any] = {}


def _init_worker(
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    load_predictor: LoadPredictorResult,
    evaluator: _Evaluator,
) -> None:
    _WORKER_CTX.update(
        config=config, goal=goal, load_predictor=load_predictor, evaluator=evaluator
    )


def _worker_eval(selection: dict[str, Any], parallel_config: Any) -> _EvalResult:
    return _evaluate_one(selection, parallel_config, **_WORKER_CTX)


def run_smart_search(
    config: SmartSearchConfig,
    *,
    evaluator: _Evaluator | None = None,
    sampler_factory: Callable[..., BranchSampler] = make_branch_sampler,
    show_progress: bool = True,
    on_round: Callable[[int, list[Candidate]], None] | None = None,
) -> list[Candidate]:
    """Run the sweep and return feasible candidates sorted best-first.

    ``evaluator`` defaults to a :class:`ReplayEvaluator` over the workload+goal;
    inject a fake to test the loop without replay. ``sampler_factory`` defaults to
    the Vizier-backed sampler. Within a round, suggestions are evaluated across
    ``SweepConfig.parallel_evals`` **spawned** worker processes (``<= 1`` -> sequential,
    no pool). With ``parallel_evals > 1`` the caller must guard its entrypoint with
    ``if __name__ == "__main__":`` (spawn re-imports the module) — the ``python -m dynamo.profiler.spica``
    CLI already does; ad-hoc scripts must too, or set ``parallel_evals=1``.
    ``show_progress`` draws a tqdm bar over the
    candidate evaluations (live feasible/failed tally + best score) and prints a
    one-line summary at the end; set False for quiet/non-interactive runs.
    """
    # AIConfigurator 0.9 does not provide the memory estimator needed to turn a
    # candidate-relative KV ratio into concrete concurrency. Detect that unsupported
    # mode before branch enumeration/Vizier work instead of returning an empty result
    # after every candidate fails to build.
    if config.workload.kv_load_ratio is not None:
        _load_memory_estimator()

    goal = config.goal
    # Predictive throughput scaling only works under the planner's "sla" target
    # (a goodput sweep). For throughput/latency sweeps, drop the throughput-scaling
    # policies up front so neither the sampler nor the load-predictor sub-sweep sees
    # them. (Disabled / load_* still run; static-path goodput is fine once the mocker
    # is SLA-aware.)
    kept, dropped = filter_scaling_policies(
        config.search_space.planner_scaling_policy,
        allow_throughput=(goal.target.planner_optimization_target == "sla"),
    )
    if dropped:
        if not kept:
            raise ValueError(
                f"every planner_scaling_policy enables throughput scaling, which a "
                f"'{goal.target.value}' sweep can't use (it has no SLA — use a goodput target, "
                f"or include 'disabled' / a load_* policy)"
            )
        if show_progress:
            tqdm.write(
                f"smart-sweep: dropped {len(dropped)} throughput-scaling policy option(s) "
                f"for target={goal.target.value} (needs SLA): {dropped}"
            )
        config = config.model_copy(
            update={
                "search_space": config.search_space.model_copy(
                    update={"planner_scaling_policy": kept}
                )
            }
        )

    # Goodput can be defined by an end-to-end SLA, but the planner's SLA scaling
    # target can only be seeded from ttft+itl. With e2e-only SLA, keep static
    # candidates and drop every scaling policy before Vizier can sample it.
    sla = goal.sla
    if (
        goal.target.planner_optimization_target == "sla"
        and sla is not None
        and (sla.ttft_ms is None or sla.itl_ms is None)
    ):
        kept = []
        dropped = []
        for policy in config.search_space.planner_scaling_policy:
            fields = scaling_fields(policy)
            target = (
                dropped
                if (
                    fields["enable_throughput_scaling"] or fields["enable_load_scaling"]
                )
                else kept
            )
            target.append(policy)
        if dropped:
            if not kept:
                raise ValueError(
                    "every planner_scaling_policy enables planner scaling, but an e2e-only SLA "
                    "cannot seed the planner's ttft/itl scaling target; use ttft_ms+itl_ms, "
                    "or include 'disabled'"
                )
            if show_progress:
                tqdm.write(
                    f"smart-sweep: dropped {len(dropped)} planner-scaling policy option(s) "
                    f"for e2e-only SLA (planner needs ttft_ms+itl_ms): {dropped}"
                )
            config = config.model_copy(
                update={
                    "search_space": config.search_space.model_copy(
                        update={"planner_scaling_policy": kept}
                    )
                }
            )

    # Thread the configured context_length into KV feasibility so parallel configs that
    # can't fit the requested sequence length are dropped up front (None -> model max).
    branches: list[BranchSpace] = enumerate_branches(
        config, max_seq_len=config.search_space.context_length
    )
    load_predictor = sweep_load_predictor(config, show_progress=show_progress)
    if evaluator is None:
        evaluator = ReplayEvaluator(config.workload, goal)

    sweep = config.sweep
    per_round = sweep.candidates_per_round or sweep.parallel_evals
    # Target number of successful unique replay configurations across all rounds.
    total = len(branches) * sweep.max_rounds * per_round
    candidates: list[Candidate] = []
    tally = {
        "feasible": 0,
        "infeasible": 0,
        "failed": 0,
        "unsupported": 0,
        "cache_hit": 0,
    }
    failure_reasons: dict[str, int] = {}
    # Unique per run: Vizier's datastore persists studies by id, so a fixed id would
    # make a later run inherit a stale study (and its old param space) -> decode crash.
    run_nonce = uuid.uuid4().hex[:8]
    # Multi-objective (pareto) -> one Vizier metric per objective (each with its own
    # direction); single-objective -> the sampler's default single maximized "objective".
    sampler_objectives = (
        [(t.value, t.maximize) for t in goal.resolved_pareto_objectives]
        if goal.is_pareto
        else None
    )
    cache_context = _freeze(
        {
            "search_space": config.search_space.model_dump(mode="python"),
            "workload": config.workload.model_dump(mode="python"),
            "goal": goal.model_dump(mode="python"),
            "load_predictor": load_predictor,
        }
    )
    replay_cache: dict[Any, tuple[Candidate, dict[str, float]]] = {}

    def _best() -> float | None:
        return max((c.score for c in candidates), default=None)

    # Parallel across worker processes when parallel_evals > 1: spawn (not fork —
    # dynamo's tokio runtime isn't fork-safe); shared read-only state goes once via the
    # initializer; one pool for the whole run amortizes the per-worker dynamo import.
    use_pool = sweep.parallel_evals > 1 and per_round > 1
    max_eval_seconds = sweep.max_eval_seconds
    worker_count = min(sweep.parallel_evals, per_round)

    def _new_pool() -> ProcessPoolExecutor:
        # spawn (not fork — dynamo's tokio runtime isn't fork-safe); shared read-only state
        # goes once via the initializer; one pool amortizes the per-worker dynamo import.
        return ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(config, goal, load_predictor, evaluator),
        )

    # One-element box so a runtime timeout can kill the hung pool and swap in a fresh one
    # (the closures below read/replace pool_box[0]).
    pool_box: list[Any] = [_new_pool() if use_pool else None]

    def _terminate_pool(pool: ProcessPoolExecutor | None) -> None:
        if pool is None:
            return
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        pool.shutdown(wait=False, cancel_futures=True)

    def _replace_pool() -> None:
        _terminate_pool(pool_box[0])
        pool_box[0] = _new_pool()

    @contextmanager
    def _pool_lifecycle():
        try:
            yield
        finally:
            _terminate_pool(pool_box[0])
            pool_box[0] = None

    def _pool_error(detail: str) -> RuntimeError:
        return RuntimeError(
            f"smart-sweep worker pool failed while {detail}. parallel_evals>1 uses "
            "spawned processes; guard a script entrypoint with `if __name__ == "
            '"__main__":`, or set sweep.parallel_evals=1 to evaluate sequentially.'
        )

    def _eval_batch(todo: list[Suggestion]):
        """Yield ``(suggestion, _EvalResult)`` for each supported suggestion — across worker
        processes when a pool is set, else sequentially in-process. On the pool path it
        evaluates waves no larger than the worker count so queued work never consumes a
        candidate's ``max_eval_seconds`` budget. A replay that overruns is reported
        infeasible ("exceed runtime") and the wave's workers are force-killed (a shared
        pool can't cancel a running task), then a fresh pool handles later waves.
        """
        if pool_box[0] is None:
            for s in todo:
                yield (
                    s,
                    _evaluate_one(
                        s.selection,
                        s.parallel_config,
                        config=config,
                        goal=goal,
                        load_predictor=load_predictor,
                        evaluator=evaluator,
                    ),
                )
            return

        for start in range(0, len(todo), worker_count):
            wave = todo[start : start + worker_count]
            pool = pool_box[0]
            assert pool is not None
            try:
                # submit() can raise when an initializer or an earlier task killed
                # the pool, so keep it inside the friendly-error wrapper.
                futures = {
                    pool.submit(
                        _worker_eval, suggestion.selection, suggestion.parallel_config
                    ): suggestion
                    for suggestion in wave
                }
            except BrokenProcessPool as exc:
                raise _pool_error("submitting a candidate wave") from exc

            pending = set(futures)
            deadline = time.monotonic() + max_eval_seconds if max_eval_seconds else None
            while pending:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                done, pending = wait(
                    pending, timeout=remaining, return_when=FIRST_COMPLETED
                )
                if not done:
                    break
                for future in done:
                    try:
                        result = future.result()
                    except BrokenProcessPool as exc:
                        raise _pool_error("collecting a candidate result") from exc
                    except Exception as exc:
                        raise _pool_error(
                            f"collecting a candidate result ({type(exc).__name__}: {exc})"
                        ) from exc
                    yield futures[future], result

            if pending:
                seconds = max_eval_seconds or 0.0
                for future in pending:
                    yield (
                        futures[future],
                        (
                            None,
                            None,
                            "infeasible",
                            f"exceed runtime: replay > {seconds:.0f}s",
                        ),
                    )
                _replace_pool()

    with _pool_lifecycle(), tqdm(
        total=total, desc="smart-sweep", unit="eval", disable=not show_progress
    ) as bar:

        def _record(outcome: str, candidate: Candidate | None) -> None:
            tally[outcome] += 1
            if candidate is not None:
                candidates.append(candidate)
                bar.update(1)
            best = _best()
            bar.set_postfix(
                feasible=tally["feasible"],
                failed=tally["failed"],
                best=("-" if best is None else f"{best:.4g}"),
            )

        round_no = 0
        for branch in branches:
            branch_stalled = False
            sampler = sampler_factory(
                branch,
                study_id=f"spica_{branch.deployment_mode}_{run_nonce}",
                objectives=sampler_objectives,
            )
            bar.set_description(f"smart-sweep {branch.deployment_mode}")
            for _ in range(sweep.max_rounds):
                unique_this_round = 0
                trial_attempts = 0
                max_trial_attempts = (
                    per_round * 11
                )  # requested batch + at most 10x replacement trials
                while (
                    unique_this_round < per_round
                    and trial_attempts < max_trial_attempts
                ):
                    ask_count = min(
                        per_round - unique_this_round,
                        max_trial_attempts - trial_attempts,
                    )
                    suggestions = sampler.suggest(
                        ask_count
                    )  # ask stays on the main process
                    if not suggestions:
                        break
                    trial_attempts += len(suggestions)

                    # Deduplicate against completed cache entries and within this ask batch.
                    # A duplicate trial still receives the cached measurement so f(z) remains
                    # deterministic, but only the first full sample reaches replay.
                    todo: list[Suggestion] = []
                    primary_by_key: dict[Any, Suggestion] = {}
                    duplicates_by_key: dict[Any, list[Suggestion]] = {}
                    for suggestion in suggestions:
                        backend = suggestion.selection["backend"]
                        if backend not in branch.supported_backends.get(
                            suggestion.parallel_config, frozenset()
                        ):
                            sampler.observe_infeasible(
                                suggestion,
                                f"backend {backend!r} does not support this parallel config",
                            )
                            _record("unsupported", None)
                            continue

                        key = _suggestion_cache_key(suggestion, cache_context)
                        cached = replay_cache.get(key)
                        if cached is not None:
                            _, cached_metrics = cached
                            sampler.observe(suggestion, cached_metrics)
                            tally["cache_hit"] += 1
                            continue
                        if key in primary_by_key:
                            duplicates_by_key.setdefault(key, []).append(suggestion)
                            continue
                        primary_by_key[key] = suggestion
                        todo.append(suggestion)

                    for suggestion, (
                        candidate,
                        observe_metrics,
                        outcome,
                        reason,
                    ) in _eval_batch(todo):
                        key = _suggestion_cache_key(suggestion, cache_context)
                        duplicates = duplicates_by_key.get(key, [])
                        if outcome in ("failed", "infeasible"):
                            sampler.observe_infeasible(suggestion, reason)
                            for duplicate in duplicates:
                                sampler.observe_infeasible(duplicate, reason)
                            if outcome == "failed":
                                failure_reasons[reason] = (
                                    failure_reasons.get(reason, 0) + 1
                                )
                            _record(outcome, None)
                            continue

                        assert candidate is not None and observe_metrics is not None
                        sampler.observe(suggestion, observe_metrics)
                        replay_cache[key] = (candidate, dict(observe_metrics))
                        for duplicate in duplicates:
                            sampler.observe(duplicate, observe_metrics)
                            tally["cache_hit"] += 1
                        _record(outcome, candidate)
                        unique_this_round += 1
                round_no += 1
                if on_round is not None:
                    on_round(round_no, list(candidates))
                if unique_this_round < per_round:
                    branch_stalled = True
                    if show_progress:
                        tqdm.write(
                            f"smart-sweep {branch.deployment_mode} stopped early: projection stalled after "
                            f"{trial_attempts} Vizier trial(s), with {unique_this_round}/{per_round} "
                            "new replay configuration(s) in the round"
                        )
                    break
            if branch_stalled:
                continue

    # Single-objective -> rank best-first by score; pareto -> the non-dominated front.
    result = (
        pareto_front(candidates, goal.resolved_pareto_objectives)
        if goal.is_pareto
        else rank(candidates)
    )
    if show_progress:
        replay_attempts = tally["feasible"] + tally["infeasible"] + tally["failed"]
        summary = (
            f"smart-sweep done: {tally['feasible']}/{replay_attempts} replay attempt(s) feasible, "
            f"{tally['infeasible']} gated, {tally['unsupported']} backend-unsupported, "
            f"{tally['failed']} replay-failed, {tally['cache_hit']} cache hit(s)"
        )
        if not candidates:
            summary += " — NO feasible candidate (check backends / SLA / gpu_budget / replay errors)"
        elif goal.is_pareto:
            summary += f"; pareto front: {len(result)} non-dominated candidate(s)"
        else:
            summary += f"; best {goal.target.value}={_best():.4g}"
        tqdm.write(summary)
        if failure_reasons:
            displayed = []
            for reason, count in list(failure_reasons.items())[:3]:
                displayed.append(f"{reason} (x{count})" if count > 1 else reason)
            remaining = len(failure_reasons) - len(displayed)
            suffix = f" | +{remaining} more distinct reason(s)" if remaining else ""
            tqdm.write(
                f"smart-sweep failure reason(s): {' | '.join(displayed)}{suffix}"
            )
    return result
