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
unroll->materialize ReplaySpec->runner->score path and never touch the study (the
Vizier trial handle never crosses the process boundary).

The replay implementation is always injected as a :class:`RunnerFactory`. Optional
feature adapters are resolved explicitly or through the ``aisimulate.sweep_config_providers``
entry-point group.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from multiprocessing.util import Finalize
from numbers import Real
from typing import Any

from tqdm import tqdm

from .config import Candidate, OptimizationGoal, OptimizationTarget, SmartSearchConfig
from .deploy import build_backend_deployment
from .discovery import resolve_providers
from .kv_estimate import resolve_backend_version
from .kv_load import InfeasibleKVCapacity, resolve_kv_load
from .provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepConfigProvider,
    SweepContext,
)
from .replay import (
    REPLAY_SPEC_API_VERSION,
    ReplayReport,
    ReplaySpec,
    Runner,
    RunnerFactory,
    canonical_json,
    validate_json_value,
)
from .sample import unroll_sample
from .sampler import BranchSampler, Suggestion, make_branch_sampler
from .score import is_feasible, make_candidate, pareto_front, rank
from .search_space import BranchSpace, enumerate_branches

logger = logging.getLogger(__name__)


# Result of evaluating one suggestion (no Vizier here): (candidate|None, observe_metrics|None,
# outcome, reason). observe_metrics is the dict fed to sampler.observe — {"objective": score}
# for a single-objective sweep, or {obj_name: raw_value, ...} under a pareto goal. outcome in
# {"feasible","infeasible","failed"}. Both "failed" (replay error) and "infeasible" (over
# gpu_budget) carry a reason and no metrics -> the loop tells the sampler observe_infeasible
# for them (a gated trial is never fed back as a high score). "unsupported" is decided on the
# main process before evaluation and never reaches the worker.
_EvalResult = tuple[Candidate | None, dict[str, float] | None, str, str]
_ReplayResult = tuple[dict[str, float] | None, str, str]


@dataclass(frozen=True)
class _PreparedCandidate:
    sample: dict[str, Any]
    replay_spec: ReplaySpec


_ADAPTER_PARAM_PREFIX = "adapter::"
_ADAPTER_PARAM_SEPARATOR = "::"


def _adapter_param(adapter_name: str, local_name: str) -> str:
    return (
        f"{_ADAPTER_PARAM_PREFIX}{adapter_name}{_ADAPTER_PARAM_SEPARATOR}{local_name}"
    )


def _adapter_selection(
    selection: Mapping[str, Any], adapter_name: str
) -> dict[str, Any]:
    prefix = _adapter_param(adapter_name, "")
    return {
        key.removeprefix(prefix): deepcopy(value)
        for key, value in selection.items()
        if key.startswith(prefix)
    }


def _prepare_providers(
    config: SmartSearchConfig,
    *,
    injected: Mapping[str, SweepConfigProvider] | None,
    show_progress: bool,
) -> tuple[dict[str, SweepConfigProvider], dict[str, AdapterSearchPlan]]:
    invalid_names = [
        name for name in config.adapters if _ADAPTER_PARAM_SEPARATOR in name
    ]
    if invalid_names:
        raise ValueError(
            f"adapter names cannot contain reserved separator "
            f"{_ADAPTER_PARAM_SEPARATOR!r}: {invalid_names}"
        )
    providers = resolve_providers(config.adapters, injected=injected)
    base_context = SweepContext(
        core_search_space=config.search_space.model_dump(mode="json"),
        workload=config.workload.model_dump(mode="json"),
        goal=config.goal.model_dump(mode="json"),
        show_progress=show_progress,
    )
    plans: dict[str, AdapterSearchPlan] = {}
    for name, provider in providers.items():
        context = SweepContext(
            core_search_space=deepcopy(base_context.core_search_space),
            workload=deepcopy(base_context.workload),
            goal=deepcopy(base_context.goal),
            show_progress=base_context.show_progress,
        )
        plan = provider.generate_search_space(
            deepcopy(config.adapters[name].search_space), context
        )
        _validate_search_plan(name, plan)
        # The adapter owns the object it returned and may reuse internal buffers
        # later. Take a complete core-owned snapshot at the ABI boundary.
        plans[name] = deepcopy(plan)
    configured_modes = set(config.search_space.deployment_mode)
    for name, plan in plans.items():
        unknown = (
            set(plan.fragment.choices_by_branch)
            | set(plan.fragment.float_ranges_by_branch)
        ) - configured_modes
        if unknown:
            raise ValueError(
                f"adapter {name!r} returned unknown deployment branch(es): "
                f"{sorted(unknown)}"
            )
    return providers, plans


def _validate_search_plan(name: str, plan: Any) -> None:
    if not isinstance(plan, AdapterSearchPlan):
        raise TypeError(
            f"adapter {name!r} generate_search_space must return AdapterSearchPlan"
        )
    if not isinstance(plan.fragment, SearchSpaceFragment):
        raise TypeError(f"adapter {name!r} returned an invalid SearchSpaceFragment")
    try:
        if type(plan.diagnostics) is not dict:
            raise TypeError("search diagnostics must be a dictionary")
        if type(plan.potential_runtime_hooks) is not tuple:
            raise TypeError("potential_runtime_hooks must be a tuple")
        _validate_search_fragment(plan.fragment)
        validate_json_value(plan.state, path=f"adapter {name!r} search plan state")
        validate_json_value(
            plan.diagnostics, path=f"adapter {name!r} search diagnostics"
        )
        for index, hook in enumerate(plan.potential_runtime_hooks):
            _validate_runtime_hook(
                hook,
                path=f"adapter {name!r} potential hook {index}",
            )
        canonical_json(plan)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"adapter {name!r} returned an invalid/non-JSON search plan: {exc}"
        ) from exc


def _validate_search_fragment(fragment: SearchSpaceFragment) -> None:
    if type(fragment.choices_by_branch) is not dict:
        raise TypeError("choices_by_branch must be a dictionary")
    for branch, parameters in fragment.choices_by_branch.items():
        if type(branch) is not str or not branch:
            raise TypeError("categorical branch names must be non-empty strings")
        if type(parameters) is not dict:
            raise TypeError(f"categorical branch {branch!r} must be a dictionary")
        for parameter, values in parameters.items():
            if type(parameter) is not str or not parameter:
                raise TypeError("categorical parameter names must be non-empty strings")
            if type(values) is not list:
                raise TypeError(
                    f"categorical parameter {parameter!r} choices must be a list"
                )
            validate_json_value(
                values, path=f"categorical parameter {parameter!r} choices"
            )

    if type(fragment.float_ranges_by_branch) is not dict:
        raise TypeError("float_ranges_by_branch must be a dictionary")
    for branch, parameters in fragment.float_ranges_by_branch.items():
        if type(branch) is not str or not branch:
            raise TypeError("continuous branch names must be non-empty strings")
        if type(parameters) is not dict:
            raise TypeError(f"continuous branch {branch!r} must be a dictionary")
        for parameter, bounds in parameters.items():
            if type(parameter) is not str or not parameter:
                raise TypeError("continuous parameter names must be non-empty strings")
            if type(bounds) is not tuple or len(bounds) != 2:
                raise TypeError(
                    f"continuous parameter {parameter!r} bounds must be a pair"
                )
            if any(type(bound) not in (int, float) for bound in bounds) or not all(
                math.isfinite(float(bound)) for bound in bounds
            ):
                raise ValueError(
                    f"continuous parameter {parameter!r} bounds must be finite numbers"
                )


def _validate_runtime_hook(hook: Any, *, path: str) -> None:
    if not isinstance(hook, RuntimeHookSpec):
        raise TypeError(f"{path} must be a RuntimeHookSpec")
    if type(hook.provider) is not str or not hook.provider:
        raise TypeError(f"{path} provider must be a non-empty string")
    if type(hook.kind) is not str or not hook.kind:
        raise TypeError(f"{path} kind must be a non-empty string")
    if type(hook.api_version) is not int or hook.api_version < 1:
        raise TypeError(f"{path} api_version must be a positive integer")
    if type(hook.config) is not dict:
        raise TypeError(f"{path} config must be a dictionary")
    validate_json_value(hook.config, path=f"{path} config")


def _validate_provider_replay_spec(name: str, spec: Any) -> None:
    if not isinstance(spec, AdapterReplaySpec):
        raise TypeError(
            f"adapter {name!r} materialize_replay must return AdapterReplaySpec"
        )
    try:
        if type(spec.config) is not dict:
            raise TypeError("replay config must be a dictionary")
        if type(spec.runtime_hooks) is not tuple:
            raise TypeError("runtime_hooks must be a tuple")
        validate_json_value(spec.config, path=f"adapter {name!r} replay config")
        for index, hook in enumerate(spec.runtime_hooks):
            _validate_runtime_hook(
                hook,
                path=f"adapter {name!r} runtime hook {index}",
            )
        canonical_json(spec)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"adapter {name!r} returned an invalid/non-JSON replay spec: {exc}"
        ) from exc


def _merge_adapter_spaces(
    branches: list[BranchSpace],
    plans: Mapping[str, AdapterSearchPlan],
) -> list[BranchSpace]:
    """Namespace and merge every adapter fragment into each core branch."""
    merged: list[BranchSpace] = []
    for branch in branches:
        choices = dict(branch.knob_choices)
        float_ranges = dict(branch.float_ranges)
        for name, plan in plans.items():
            local_choices = plan.fragment.choices_by_branch.get(
                branch.deployment_mode, {}
            )
            local_ranges = plan.fragment.float_ranges_by_branch.get(
                branch.deployment_mode, {}
            )
            overlap = set(local_choices).intersection(local_ranges)
            if overlap:
                raise ValueError(
                    f"adapter {name!r} defined parameters as both categorical and "
                    f"continuous: {sorted(overlap)}"
                )
            for local_name, values in local_choices.items():
                if _ADAPTER_PARAM_SEPARATOR in local_name:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} contains "
                        f"reserved separator {_ADAPTER_PARAM_SEPARATOR!r}"
                    )
                if not values:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} "
                        f"has no choices in branch {branch.deployment_mode!r}"
                    )
                choices[_adapter_param(name, local_name)] = list(values)
            for local_name, bounds in local_ranges.items():
                if _ADAPTER_PARAM_SEPARATOR in local_name:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} contains "
                        f"reserved separator {_ADAPTER_PARAM_SEPARATOR!r}"
                    )
                low, high = bounds
                if low >= high:
                    raise ValueError(
                        f"adapter {name!r} search parameter {local_name!r} needs "
                        f"low < high, got {bounds!r}"
                    )
                float_ranges[_adapter_param(name, local_name)] = (low, high)
        merged.append(replace(branch, knob_choices=choices, float_ranges=float_ranges))
    return merged


def _freeze(value: Any) -> Any:
    """Convert a nested suggestion/context value into a stable hashable key."""
    if is_dataclass(value) and not isinstance(value, type):
        return ("dataclass", type(value).__qualname__, _freeze(asdict(value)))
    if isinstance(value, Enum):
        return ("enum", type(value).__qualname__, _freeze(value.value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (_freeze(key), _freeze(item))
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if isinstance(value, set):
        return ("set", tuple(sorted((_freeze(item) for item in value), key=repr)))
    if isinstance(value, frozenset):
        return (
            "frozenset",
            tuple(sorted((_freeze(item) for item in value), key=repr)),
        )
    if value is None:
        return ("none",)
    if type(value) in (str, int, float, bool):
        return (type(value).__name__, value)
    try:
        hash(value)
    except TypeError:
        return ("repr", type(value).__qualname__, repr(value))
    return ("hashable", type(value).__qualname__, value)


def _suggestion_cache_key(suggestion: Suggestion, context: Any) -> Any:
    """A run-local full-sample key; parallel equality alone is not a cache hit."""
    return (context, _freeze(suggestion.selection), _freeze(suggestion.parallel_config))


def _materialize_one(
    selection: dict[str, Any],
    parallel_config: Any,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    providers: Mapping[str, SweepConfigProvider],
    provider_plans: Mapping[str, AdapterSearchPlan],
    runner_factory: RunnerFactory,
) -> tuple[_PreparedCandidate | None, _EvalResult | None]:
    """Build a complete replay specification on the main process."""
    try:
        sample = unroll_sample(
            search_space=config.search_space,
            selection=selection,
            parallel_config=parallel_config,
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
        backend_deployment = build_backend_deployment(
            sample, backend_version=backend_version
        )
        adapter_specs: dict[str, AdapterReplaySpec] = {}
        for name, provider in providers.items():
            candidate_context = CandidateContext(
                sample=deepcopy(sample),
                backend_deployment=deepcopy(backend_deployment),
                concurrency=concurrency,
            )
            adapter_spec = provider.materialize_replay(
                deepcopy(provider_plans[name]),
                _adapter_selection(selection, name),
                candidate_context,
            )
            _validate_provider_replay_spec(name, adapter_spec)
            # Frozen dataclasses do not freeze nested JSON containers. Snapshot
            # the return value so an adapter can safely reuse an output buffer
            # without mutating candidates already prepared in this round.
            adapter_specs[name] = deepcopy(adapter_spec)
        replay_spec = ReplaySpec(
            backend_deployment=backend_deployment,
            workload=config.workload.model_dump(mode="json"),
            goal=goal.model_dump(mode="json"),
            concurrency=concurrency,
            adapters=adapter_specs,
        )
        canonical_json(replay_spec)
        runner_factory.capabilities().require_compatible(replay_spec)
        if adapter_specs:
            sample["adapters"] = {
                name: deepcopy(adapter_spec.config)
                for name, adapter_spec in adapter_specs.items()
            }
    except InfeasibleKVCapacity as exc:
        return None, (
            None,
            None,
            "infeasible",
            f"candidate KV capacity infeasible: {exc}",
        )
    except Exception as exc:
        logger.exception("Sweeper candidate build failed")
        return None, (
            None,
            None,
            "failed",
            f"candidate build failed: {type(exc).__name__}: {exc}",
        )
    return _PreparedCandidate(sample=sample, replay_spec=replay_spec), None


def _run_replay(spec: ReplaySpec, runner: Runner) -> _ReplayResult:
    """Worker-only boundary: run one fully materialized replay specification."""
    try:
        report = runner.run(spec)
        if not isinstance(report, ReplayReport):
            raise TypeError(
                f"runner.run must return ReplayReport, got {type(report).__name__}"
            )
        if type(report.metrics) is not dict:
            raise TypeError("runner report metrics must be a dictionary")
        if type(report.metadata) is not dict:
            raise TypeError("runner report metadata must be a dictionary")
        metrics: dict[str, float] = {}
        for name, value in report.metrics.items():
            if type(name) is not str:
                raise TypeError(f"runner metric names must be strings, got {name!r}")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"runner metric {name!r} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"runner metric {name!r} must be finite")
            metrics[name] = normalized
        validate_json_value(report.metadata, path="runner report metadata")
        return metrics, "replayed", ""
    except Exception as exc:  # one candidate failing must not abort the sweep
        logger.exception("Sweeper candidate replay failed")
        return None, "failed", f"replay failed: {type(exc).__name__}: {exc}"


def _score_prepared(
    prepared: _PreparedCandidate,
    replay_result: _ReplayResult,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
) -> _EvalResult:
    """Score a runner result on the main process."""
    report, outcome, reason = replay_result
    if outcome == "failed":
        return None, None, outcome, reason
    if report is None:
        return (
            None,
            None,
            "failed",
            "runner contract violation: a successful replay returned no metrics",
        )
    effective_targets = (
        set(goal.resolved_pareto_objectives) if goal.is_pareto else {goal.target}
    )
    if (
        effective_targets.intersection(
            {OptimizationTarget.GOODPUT, OptimizationTarget.GOODPUT_PER_GPU}
        )
        and "goodput_output_throughput_tok_s" not in report
    ):
        return (
            None,
            None,
            "failed",
            (
                "runner contract violation: goodput objective requires "
                "goodput_output_throughput_tok_s; aggregate latency cannot be used "
                "as a fallback"
            ),
        )
    sample = prepared.sample
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


# Worker-process plumbing: shared read-only state is sent once via the pool
# initializer; each process creates and reuses one runner.
_WORKER_CTX: dict[str, Any] = {}


def _init_worker(
    runner_factory: RunnerFactory,
) -> None:
    identity = getattr(mp.current_process(), "_identity", ())
    worker_id = int(identity[0]) if identity else 0
    runner = runner_factory.create(worker_id)
    # multiprocessing runs Finalize callbacks during a normal child-process
    # shutdown.  Unlike a plain atexit handler, this matches the ProcessPool
    # worker lifecycle and lets a runtime release worker-local resources.
    Finalize(None, runner.close, exitpriority=0)
    _WORKER_CTX.update(runner=runner)


def _worker_eval(spec: ReplaySpec) -> _ReplayResult:
    return _run_replay(spec, _WORKER_CTX["runner"])


class Sweeper:
    """Compose and execute isolated backend-neutral configuration sweeps.

    The constructor owns stable dependencies. Every :meth:`run` call creates
    fresh studies, caches, runners, and worker pools, so mutable sweep state never
    leaks between runs.
    """

    def __init__(
        self,
        *,
        runner_factory: RunnerFactory,
        providers: Mapping[str, SweepConfigProvider] | None = None,
        sampler_factory: Callable[..., BranchSampler] = make_branch_sampler,
        show_progress: bool = True,
    ) -> None:
        self._runner_factory = runner_factory
        self._providers = dict(providers or {})
        self._sampler_factory = sampler_factory
        self._show_progress = show_progress

    def run(
        self,
        config: SmartSearchConfig,
        *,
        on_round: Callable[[int, list[Candidate]], None] | None = None,
    ) -> list[Candidate]:
        """Run the sweep and return feasible candidates sorted best-first.

        The configured runner factory is the only replay runtime injection point.
        Providers can be injected directly; otherwise configured providers are
        discovered from package entry points. Within a round, suggestions are evaluated
        across spawned worker processes when ``parallel_evals > 1``. Such callers must
        guard script entrypoints with ``if __name__ == "__main__":``.
        """
        runner_factory = self._runner_factory
        providers = self._providers
        sampler_factory = self._sampler_factory
        show_progress = self._show_progress

        goal = config.goal
        capabilities = runner_factory.capabilities()
        capabilities.require_replay_spec_version(REPLAY_SPEC_API_VERSION)

        # Preserve the legacy preflight order: reject an impossible backend/topology
        # search before adapters perform any potentially expensive preparation.
        branches = enumerate_branches(
            config,
            max_seq_len=config.search_space.context_length,
            runner_capabilities=capabilities,
        )
        resolved_providers, provider_plans = _prepare_providers(
            config, injected=providers, show_progress=show_progress
        )
        for name, plan in provider_plans.items():
            unsupported = [
                hook
                for hook in plan.potential_runtime_hooks
                if not capabilities.supports_hook(hook)
            ]
            if unsupported:
                labels = ", ".join(
                    f"{hook.provider}:{hook.kind}@{hook.api_version}"
                    for hook in unsupported
                )
                raise ValueError(
                    f"runner is incompatible with configured adapter {name!r}; "
                    f"unsupported runtime hook(s): {labels}"
                )

        branches = _merge_adapter_spaces(branches, provider_plans)

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
                "adapters": {
                    name: request.model_dump(mode="python")
                    for name, request in config.adapters.items()
                },
                "workload": config.workload.model_dump(mode="python"),
                "goal": goal.model_dump(mode="python"),
                "provider_plans": provider_plans,
            }
        )
        replay_cache: dict[Any, tuple[Candidate, dict[str, float]]] = {}

        def _best() -> float | None:
            return max((c.score for c in candidates), default=None)

        # Parallel across worker processes when parallel_evals > 1. Spawn keeps
        # runner runtimes isolated and lets each worker reuse one runner instance.
        use_pool = sweep.parallel_evals > 1 and per_round > 1
        max_eval_seconds = sweep.max_eval_seconds
        worker_count = min(sweep.parallel_evals, per_round)
        sequential_runner = None if use_pool else runner_factory.create(0)

        def _new_pool() -> ProcessPoolExecutor:
            return ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(runner_factory,),
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
            except BaseException:
                # Do not wait forever for an unrelated hung replay when orchestration,
                # scoring, observation, or cancellation aborts the sweep.
                _terminate_pool(pool_box[0])
                pool_box[0] = None
                if sequential_runner is not None:
                    sequential_runner.close()
                raise
            else:
                # A completed sweep shuts workers down normally so their Runner
                # finalizers execute.  Only the timeout-recovery path above sends a
                # terminate signal, where cleanup is necessarily best-effort.
                if pool_box[0] is not None:
                    pool_box[0].shutdown(wait=True, cancel_futures=True)
                pool_box[0] = None
                if sequential_runner is not None:
                    sequential_runner.close()

        def _pool_error(detail: str) -> RuntimeError:
            return RuntimeError(
                f"Sweeper worker pool failed while {detail}. parallel_evals>1 uses "
                "spawned processes; guard a script entrypoint with `if __name__ == "
                '"__main__":`, or set sweep.parallel_evals=1 to evaluate sequentially.'
            )

        def _eval_batch(todo: list[tuple[Suggestion, _PreparedCandidate]]):
            """Yield ``(suggestion, _EvalResult)`` for each supported suggestion — across worker
            processes when a pool is set, else sequentially in-process. On the pool path it
            evaluates waves no larger than the worker count so queued work never consumes a
            candidate's ``max_eval_seconds`` budget. A replay that overruns is reported
            infeasible ("exceed runtime") and the wave's workers are force-killed (a shared
            pool can't cancel a running task), then a fresh pool handles later waves.
            """
            if pool_box[0] is None:
                assert sequential_runner is not None
                for suggestion, prepared in todo:
                    yield (
                        suggestion,
                        _score_prepared(
                            prepared,
                            _run_replay(prepared.replay_spec, sequential_runner),
                            config=config,
                            goal=goal,
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
                        pool.submit(_worker_eval, prepared.replay_spec): (
                            suggestion,
                            prepared,
                        )
                        for suggestion, prepared in wave
                    }
                except BrokenProcessPool as exc:
                    raise _pool_error("submitting a candidate wave") from exc

                pending = set(futures)
                deadline = (
                    time.monotonic() + max_eval_seconds if max_eval_seconds else None
                )
                while pending:
                    remaining = (
                        None
                        if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    done, pending = wait(
                        pending, timeout=remaining, return_when=FIRST_COMPLETED
                    )
                    if not done:
                        break
                    for future in done:
                        try:
                            replay_result = future.result()
                        except BrokenProcessPool as exc:
                            raise _pool_error("collecting a candidate result") from exc
                        except Exception as exc:
                            raise _pool_error(
                                f"collecting a candidate result ({type(exc).__name__}: {exc})"
                            ) from exc
                        suggestion, prepared = futures[future]
                        yield (
                            suggestion,
                            _score_prepared(
                                prepared,
                                replay_result,
                                config=config,
                                goal=goal,
                            ),
                        )

                if pending:
                    seconds = max_eval_seconds or 0.0
                    for future in pending:
                        suggestion, _prepared = futures[future]
                        yield (
                            suggestion,
                            (
                                None,
                                None,
                                "infeasible",
                                f"exceed runtime: replay > {seconds:.0f}s",
                            ),
                        )
                    _replace_pool()

        with (
            _pool_lifecycle(),
            tqdm(
                total=total, desc="sweeper", unit="eval", disable=not show_progress
            ) as bar,
        ):

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
                    study_id=f"sweeper_{branch.deployment_mode}_{run_nonce}",
                    objectives=sampler_objectives,
                )
                bar.set_description(f"Sweeper {branch.deployment_mode}")
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
                        todo: list[tuple[Suggestion, _PreparedCandidate]] = []
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

                        # Materialization stays on the main process: adapters see the
                        # resolved backend candidate and workers receive ReplaySpec only.
                        for key, suggestion in primary_by_key.items():
                            prepared, build_result = _materialize_one(
                                suggestion.selection,
                                suggestion.parallel_config,
                                config=config,
                                goal=goal,
                                providers=resolved_providers,
                                provider_plans=provider_plans,
                                runner_factory=runner_factory,
                            )
                            if build_result is not None:
                                (
                                    candidate,
                                    observe_metrics,
                                    outcome,
                                    reason,
                                ) = build_result
                                assert candidate is None and observe_metrics is None
                                duplicates = duplicates_by_key.get(key, [])
                                sampler.observe_infeasible(suggestion, reason)
                                for duplicate in duplicates:
                                    sampler.observe_infeasible(duplicate, reason)
                                if outcome == "failed":
                                    failure_reasons[reason] = (
                                        failure_reasons.get(reason, 0)
                                        + 1
                                        + len(duplicates)
                                    )
                                _record(outcome, None)
                                for _duplicate in duplicates:
                                    _record(outcome, None)
                                continue
                            assert prepared is not None
                            todo.append((suggestion, prepared))

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
                                        failure_reasons.get(reason, 0)
                                        + 1
                                        + len(duplicates)
                                    )
                                _record(outcome, None)
                                for _duplicate in duplicates:
                                    _record(outcome, None)
                                continue

                            if candidate is None or observe_metrics is None:
                                raise RuntimeError(
                                    "Sweeper runner contract violation: a feasible outcome "
                                    "must include both a candidate and observation metrics"
                                )
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
                                f"Sweeper {branch.deployment_mode} stopped early: projection stalled after "
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
                f"Sweeper done: {tally['feasible']}/{replay_attempts} replay attempt(s) feasible, "
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
                    f"Sweeper failure reason(s): {' | '.join(displayed)}{suffix}"
                )
        return result
