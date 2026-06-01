# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Saturation discovery test — finds C_max for an EXISTING deployment under
# a latency SLA. Does NOT deploy/teardown; runs aiperf against the FE
# service in the chosen namespace.
#
# Three sweep modes:
#   1. ladder         — fixed ascending concurrency list, stop at first SLA violation
#   2. adaptive       — headroom-scaled step size, bisects toward the SLA boundary
#   3. aiperf-builtin — delegate to aiperf's own `--search-space / --search-sla`
#                       (one aiperf job; planner is monotonic-sla or bayesian)
#
# Modes 1 and 2 are our own driver: one aiperf job per rung, framework parses
# the per-rung summary. Mode 3 is a single aiperf job; the recommended
# concurrency is parsed from aiperf's search output.
#
# Example invocation (must be run AFTER a deployment is up in <namespace>):
#
#   uv run pytest test_saturation_discovery.py \
#       --namespace neelays-e1-arm5 \
#       --sat-served-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
#       --sat-concurrency-ladder 200,400,800,1200,1600,2000,2400 \
#       --sat-rung-seconds 90 \
#       --sat-sla-ms 5000 \
#       --sat-sla-quantile p95 \
#       --storage-class dgxc-enterprise-file --log-pvc dynamo-ft-logs \
#       --skip-service-restart -s -v
#
# For mocker deployments running at --speedup-ratio 5.0, an SLA of 5000 ms
# (=30000 / 5x speedup) is the canonical "production-equivalent" target.
# For real-vLLM workloads the SLA is the production p95 latency budget
# (e.g., 30000 ms).

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.utils.managed_load import LoadConfig, ManagedLoad

_LOG = logging.getLogger(__name__)


def _normalize_quantile(q: str) -> str:
    """Map p50/p95/p99/avg/min/max → the key in aiperf summary JSON."""
    if q in ("avg", "p50", "p95", "p99", "min", "max"):
        return q
    raise ValueError(f"unknown SLA quantile: {q}")


def _extract_quantile(summary: dict, metric: str, quantile: str) -> float | None:
    """Read summary[metric][quantile] from the aiperf JSON shape."""
    m = summary.get(metric, {})
    return m.get(quantile)


async def _run_aiperf_builtin(
    *,
    deployment,
    namespace: str,
    image: str | None,
    model: str,
    log_dir: Path,
    search_lo: int,
    search_hi: int,
    search_planner: str,
    search_max_iter: int,
    sla_metric: str,
    sla_quantile: str,
    sla_ms: float,
    rung_seconds: int,
    warmup_seconds: int,
    request_timeout_seconds: int = 600,
) -> None:
    """Single-shot aiperf invocation with --search-space / --search-sla.

    Delegates the entire sweep to aiperf's own search planner. Each search
    iteration is one internal aiperf profile run (gated by --search-max-iter).
    We don't parse per-iteration p95 here — aiperf writes its own
    search-summary artifact under the load job's results directory; we just
    log the path so the operator can read it directly.

    aiperf wires search-stat from the SLA quantile (e.g. 'p95'). For
    request_latency, aiperf reports milliseconds, so --search-sla takes
    the SLA in ms directly. Direction is 'maximize' for concurrency
    sweeps under a latency ceiling.
    """
    # aiperf wants the SLA in the metric's native unit (ms for *_latency)
    # and the search-stat in lowercase quantile form ('p95', 'avg').
    extra_args = [
        "--search-space",
        f"phases.profiling.concurrency:{search_lo},{search_hi}:int",
        "--search-metric",
        sla_metric,
        "--search-stat",
        sla_quantile,
        "--search-direction",
        "maximize",
        "--search-sla",
        f"{sla_metric}:{sla_quantile}:lt:{int(sla_ms)}",
        "--search-planner",
        search_planner,
        "--search-max-iterations",
        str(search_max_iter),
    ]

    # Concurrency is the variable aiperf will tune via --search-space; we
    # still need to set a starting value on the LoadConfig (aiperf uses
    # --concurrency as the search-space seed when --search-space is set).
    load_config = LoadConfig(
        model_name=model,
        tokenizer=model,
        concurrency=search_lo,
        duration_minutes=rung_seconds / 60.0,
        warmup_duration=warmup_seconds,
        seq_dist=(
            "100,20:1;100,80:2;100,130:2;100,180:1;100,200:1;"
            "500,20:2;500,80:5;500,130:5;500,180:2;500,200:1;"
            "1000,20:3;1000,80:5;1000,130:5;1000,180:3;1000,200:2;"
            "1600,20:5;1600,80:8;1600,130:8;1600,180:5;1600,200:3;"
            "3400,20:3;3400,80:6;3400,130:6;3400,180:3;3400,200:2;"
            "7000,20:2;7000,80:3;7000,130:3;7000,180:2;7000,200:1"
        ),
        num_prefix_prompts=5,
        prefix_prompt_length=740,
        streaming=True,
        ignore_eos=True,
        # VERY relaxed client timeout, deliberately >> the latency SLA. If the
        # per-request timeout equals the SLA (both 30s), an overloaded request
        # is cancelled at 30s and counted as an ERROR, so aiperf's p95 can
        # never exceed the SLA — the search then thinks every concurrency
        # "passes" and picks a garbage-high C_max from a 99%-erroring rung
        # (observed: conc 1013 -> 100% errors but p95 ~28s "passed"). With a
        # relaxed timeout, overload instead shows up as genuine high-latency
        # successes, so p95 climbs past the SLA and the search finds the real
        # saturation point. Default 600s (10 min); see --sat-request-timeout-seconds.
        request_timeout_seconds=request_timeout_seconds,
        connection_reuse_strategy="never",
        extra_aiperf_args=extra_args,
    )

    job_name = "sat-aiperf-search"
    ml = ManagedLoad(
        namespace=namespace,
        load_config=load_config,
        pvc_name=deployment.get_log_pvc_name(),
        pvc_run_id=deployment.get_log_run_id(),
        endpoint_url=deployment.deployment_spec.get_in_cluster_frontend_url(namespace),
        log_dir=str(log_dir),
        job_name=job_name,
    )
    # aiperf's bayesian search runs up to ``search_max_iter`` concurrency
    # rungs back-to-back (each ~rung_seconds + warmup_seconds) plus planner
    # overhead. ManagedLoad's default completion timeout is derived from a
    # SINGLE rung's duration (~390s), which is far too short for the whole
    # search and trips a spurious TimeoutError. Wait with a search-aware
    # budget instead.
    # Each rung sends for ~rung_seconds then drains in-flight requests, which
    # with the relaxed 300s client timeout can take a while near the cliff;
    # budget generously (this is just an upper bound — aiperf exits when done).
    search_timeout = search_max_iter * (rung_seconds + warmup_seconds) + 1800
    _LOG.info(
        "aiperf-builtin: waiting up to %ds for the %d-iteration search "
        "(rung=%ds, warmup=%ds) to complete",
        search_timeout,
        search_max_iter,
        rung_seconds,
        warmup_seconds,
    )
    async with ml:
        await ml.run(wait_for_completion=False)
        await ml.wait_for_completion(timeout=search_timeout)

    # aiperf's search planner writes a recommendation artifact alongside the
    # per-iteration profile exports. Glob for likely names — schema varies by
    # aiperf release, so we capture paths and let the operator parse.
    job_dirs = sorted((log_dir / "load").glob(f"{job_name}-*"))
    if job_dirs:
        latest = job_dirs[-1]
        candidates = list(latest.rglob("*.json"))
        recommendation_files = [
            p
            for p in candidates
            if any(tok in p.name.lower() for tok in ("search", "recommend"))
        ]
        _LOG.info("aiperf-builtin: artifact dir = %s", latest)
        if recommendation_files:
            _LOG.info(
                "aiperf-builtin: %d candidate recommendation file(s): %s",
                len(recommendation_files),
                [str(p.relative_to(latest)) for p in recommendation_files],
            )
        else:
            _LOG.info(
                "aiperf-builtin: no obvious recommendation JSON. All %d json files: %s",
                len(candidates),
                [str(p.relative_to(latest)) for p in candidates[:20]],
            )

    summary_path = log_dir / "saturation_sweep_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "namespace": namespace,
                "image": image,
                "served_model": model,
                "mode": "aiperf-builtin",
                "search_lo": search_lo,
                "search_hi": search_hi,
                "search_planner": search_planner,
                "search_max_iterations": search_max_iter,
                "sla_metric": sla_metric,
                "sla_quantile": sla_quantile,
                "sla_ms": sla_ms,
                "rung_seconds": rung_seconds,
                "job_name": job_name,
                "note": (
                    "aiperf-builtin: see job_dirs above for per-iteration outputs "
                    "and the recommendation artifact."
                ),
            },
            fh,
            indent=2,
        )
    _LOG.info("aiperf-builtin: summary written: %s", summary_path)


def _build_deploy_spec(
    scenario_name: str, *, image: str | None, model_cache_pvc: str | None
):
    """Resolve a router_memory scenario name to a ready-to-deploy DeploymentSpec.

    Mirrors test_router_modes._run_yaml_scenario's spec-build sequence
    (load_scenario -> _load_template_spec -> apply_deployment/router/admission
    -> apply_pull_secrets -> apply_mocker_planner_profile_fixup) so a deployment
    stood up here is identical to one stood up by the router-mode suite. We do
    NOT call run_scenario — this test owns its own ManagedDeployment lifecycle.

    Returns (spec, scenario) so the caller can read scenario.deployment.log_pvc.
    """
    from tests.fault_tolerance.deploy.scenario_lib._loader import (
        SCENARIOS_DIR,
        load_scenario,
    )
    from tests.fault_tolerance.deploy.scenario_lib._runtime import (
        apply_admission,
        apply_deployment,
        apply_mocker_planner_profile_fixup,
        apply_pull_secrets,
        apply_router,
    )
    from tests.fault_tolerance.deploy.test_router_modes import _load_template_spec

    name = scenario_name
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    scenario_path = SCENARIOS_DIR / "router_memory" / name
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"--sat-deploy-scenario: scenario not found: {scenario_path}"
        )
    scenario = load_scenario(scenario_path)

    # CLI overrides (parity with test_router_modes): --model-cache-pvc and
    # --image trump the YAML's values when non-empty.
    if model_cache_pvc:
        scenario.deployment.model_cache_pvc = model_cache_pvc
    if image:
        scenario.deployment.image = image

    spec = _load_template_spec(scenario.deployment)
    apply_deployment(spec, scenario.deployment)
    apply_router(spec, scenario.router)
    apply_admission(spec, scenario.admission)
    apply_pull_secrets(spec)
    apply_mocker_planner_profile_fixup(spec)
    return spec, scenario


def _write_c_max_json(
    log_dir: Path,
    *,
    search_lo: int,
    search_hi: int,
    sla_ms: float,
    max_error_pct: float = 5.0,
) -> None:
    """Parse aiperf's search output for C_max and write <log_dir>/c_max.json.

    Extraction is best-effort and defensive. aiperf's search-summary schema
    varies by release, so we try several known shapes in order and log exactly
    what we found. If nothing parses, we write c_max=0 with a "source" marker
    so the wrapper script can detect the failure and fall back to grepping.

    JSON shape written: {"c_max": int, "sla_ms": float,
                         "search_lo": int, "search_hi": int, "source": str}
    """
    c_max = 0
    source = "unparsed"
    ERR_PCT_MAX = (
        max_error_pct  # a rung "passes" only if <= this %% of requests errored
    )

    # aiperf-builtin search writes one dir per iteration:
    #   <log_dir>/load/sat-aiperf-search/search_iter_*/profile_runs/run_*/profile_export_aiperf.json
    # Each summary carries the tested concurrency
    # (input_config.phases.profiling.concurrency), request_latency.p95 (display
    # unit = ms, same unit as the SLA), and request vs error counts. C_max =
    # the HIGHEST concurrency whose p95 <= sla_ms AND whose error rate is
    # acceptable.
    #
    # The error gate is ESSENTIAL: under a tight client timeout an overloaded
    # rung can report p95 < SLA computed over a handful of survivors while ~all
    # requests error out — without the gate the search would pick a garbage-high
    # C_max (observed: concurrency 1013 -> 100% errors but p95 ~28s "passed").
    search_dir = log_dir / "load" / "sat-aiperf-search"
    summaries = sorted(
        search_dir.glob("search_iter_*/profile_runs/run_*/profile_export_aiperf.json")
    )
    _LOG.info(
        "c_max parse: scanning %d iteration summaries under %s",
        len(summaries),
        search_dir,
    )
    passing: list[int] = []
    for p in summaries:
        try:
            with open(p) as fh:
                data = json.load(fh)
        except Exception as e:  # noqa: BLE001 - defensive parse
            _LOG.warning("c_max parse: could not read %s: %s", p, e)
            continue
        ic = data.get("input_config") or {}
        # ``phases`` is a LIST of phase dicts; the tested concurrency is on the
        # entry named "profiling" (the "warmup" entry carries the search seed).
        conc = None
        try:
            for ph in ic.get("phases") or []:
                if isinstance(ph, dict) and ph.get("name") == "profiling":
                    conc = int(ph["concurrency"])
                    break
        except Exception:  # noqa: BLE001
            conc = None
        p95 = (data.get("request_latency") or {}).get("p95")
        ok = (data.get("request_count") or {}).get("avg") or 0
        err = (data.get("error_request_count") or {}).get("avg") or 0
        tot = ok + err
        err_pct = (100.0 * err / tot) if tot else 100.0
        if conc is None or p95 is None:
            continue
        ok_sla = p95 <= sla_ms
        ok_err = err_pct <= ERR_PCT_MAX
        verdict = (
            "PASS"
            if (ok_sla and ok_err)
            else ("FAIL-SLA" if not ok_sla else "FAIL-ERR")
        )
        _LOG.info(
            "c_max parse: conc=%d p95=%.0fms ok=%d err=%d (%.1f%%) -> %s",
            conc,
            p95,
            int(ok),
            int(err),
            err_pct,
            verdict,
        )
        if ok_sla and ok_err:
            passing.append(conc)

    if passing:
        c_max = max(passing)
        source = str(search_dir)
        _LOG.info(
            "c_max parse: C_max=%d (highest concurrency with p95<=%.0fms AND err<=%.1f%%)",
            c_max,
            sla_ms,
            ERR_PCT_MAX,
        )
    else:
        _LOG.warning(
            "c_max parse: NO rung passed (p95<=%.0fms AND err<=%.1f%%); writing "
            "c_max=0. The search range may be entirely above saturation — inspect "
            "the per-iteration summaries under %s.",
            sla_ms,
            ERR_PCT_MAX,
            search_dir,
        )

    out = {
        "c_max": c_max,
        "sla_ms": float(sla_ms),
        "search_lo": int(search_lo),
        "search_hi": int(search_hi),
        "source": source,
    }
    path = log_dir / "c_max.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    _LOG.info("c_max parse: wrote %s = %s", path, out)


def _extract_c_max_from_obj(data) -> int | None:
    """Pull a concurrency recommendation out of an aiperf JSON object.

    Tries, in order, the shapes seen across aiperf releases:
      1. top-level keys: best_concurrency / recommended_concurrency / c_max
      2. nested "search"/"result"/"recommendation" dict with a *concurrency* key
      3. a "concurrency" key whose sibling marks it as the SLA-passing best
    Returns None if nothing matches. Purely defensive — never raises.
    """
    try:
        if isinstance(data, dict):
            for k in (
                "best_concurrency",
                "recommended_concurrency",
                "c_max",
                "max_concurrency",
            ):
                v = data.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    return int(v)
            for container_key in ("search", "result", "recommendation", "summary"):
                sub = data.get(container_key)
                if isinstance(sub, dict):
                    nested = _extract_c_max_from_obj(sub)
                    if nested:
                        return nested
            # Last resort: any int-valued key containing "concurrency".
            for k, v in data.items():
                if (
                    "concurrency" in str(k).lower()
                    and isinstance(v, (int, float))
                    and v > 0
                ):
                    return int(v)
        elif isinstance(data, list):
            best = 0
            for item in data:
                nested = _extract_c_max_from_obj(item)
                if nested and nested > best:
                    best = nested
            return best or None
    except Exception:  # noqa: BLE001 - never let parse kill the test
        return None
    return None


def add_cli_options(parser):
    """Pytest CLI options scoped to saturation discovery."""
    g = parser.getgroup("saturation_discovery")
    g.addoption(
        "--sat-served-model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Served model name passed to aiperf --model + --tokenizer.",
    )
    g.addoption(
        "--sat-concurrency-ladder",
        default="200,400,800,1200,1600,2000,2400",
        help="Comma-separated concurrency levels to sweep (ascending).",
    )
    g.addoption(
        "--sat-rung-seconds",
        type=int,
        default=90,
        help="Seconds of steady load per rung. Default 90.",
    )
    g.addoption(
        "--sat-warmup-seconds",
        type=int,
        default=15,
        help="Warmup seconds at start of each rung (aiperf request_count "
        "stabilization). Counts against rung-seconds.",
    )
    g.addoption(
        "--sat-request-timeout-seconds",
        type=int,
        default=600,
        help="Client per-request timeout for the search load (aiperf "
        "--request-timeout-seconds). DEFAULT 600s (10 min) — deliberately "
        ">> the latency SLA. This is critical for a correct C_max: if the "
        "client timeout is near the SLA, overloaded requests get cancelled "
        "and counted as ERRORS (not high-latency successes), so the measured "
        "p95 can never exceed the SLA and the search picks a garbage-high "
        "C_max from a ~100%-erroring rung. A relaxed timeout lets overload "
        "show up as honest high latency. Pass <=0 for effectively-infinite "
        "(mapped to 24h).",
    )
    g.addoption(
        "--sat-max-error-pct",
        type=float,
        default=5.0,
        help="A search rung only counts toward C_max if its error rate is "
        "<= this percent AND its p95 <= the SLA. The error gate guards "
        "against picking a concurrency where p95 looks fine but it was "
        "computed over a handful of survivors while the rung was failing. "
        "Default 5%%.",
    )
    g.addoption(
        "--sat-sla-metric",
        default="request_latency",
        choices=["request_latency", "time_to_first_token", "inter_token_latency"],
        help="Which aiperf metric to gate on.",
    )
    g.addoption(
        "--sat-sla-quantile",
        default="p95",
        choices=["avg", "p50", "p95", "p99"],
        help="Which quantile of the SLA metric to gate on.",
    )
    g.addoption(
        "--sat-sla-ms",
        type=float,
        default=5000.0,
        help="SLA threshold in milliseconds. Below = pass. Above = saturated. "
        "Default 5000 ms — appropriate for mocker --speedup-ratio=5.0 "
        "(matches prod 30s budget / 5x).",
    )
    g.addoption(
        "--sat-min-success-rate",
        type=float,
        default=0.95,
        help="Minimum aiperf request success rate per rung (1 - errors/total). "
        "Below this counts as saturated.",
    )
    # --- Sweep mode selection ---
    g.addoption(
        "--sat-mode",
        default="ladder",
        choices=["ladder", "adaptive", "aiperf-builtin"],
        help="Sweep strategy. 'ladder' = fixed list, stop at first violation. "
        "'adaptive' = headroom-scaled step + bisection. "
        "'aiperf-builtin' = single aiperf job with --search-space / --search-sla.",
    )
    # Back-compat shortcut: --sat-adaptive is equivalent to --sat-mode=adaptive.
    g.addoption(
        "--sat-adaptive",
        action="store_true",
        default=False,
        help="Shortcut for --sat-mode=adaptive (kept for back-compat).",
    )
    g.addoption(
        "--sat-start-concurrency",
        type=int,
        default=200,
        help="Adaptive mode: initial concurrency level. Default 200.",
    )
    g.addoption(
        "--sat-min-step",
        type=int,
        default=50,
        help="Adaptive mode: minimum step size (per rung). Default 50.",
    )
    g.addoption(
        "--sat-max-step",
        type=int,
        default=2000,
        help="Adaptive mode: maximum step size (per rung). Default 2000.",
    )
    g.addoption(
        "--sat-tolerance",
        type=float,
        default=0.05,
        help="Adaptive mode: stop when |p95 - SLA|/SLA < tolerance. Default 0.05 (±5%).",
    )
    g.addoption(
        "--sat-max-rungs",
        type=int,
        default=12,
        help="Adaptive mode: hard cap on iterations. Default 12.",
    )
    # --- aiperf-builtin mode options ---
    g.addoption(
        "--sat-search-lo",
        type=int,
        default=50,
        help="aiperf-builtin: lower bound of search-space concurrency. Default 50.",
    )
    g.addoption(
        "--sat-search-hi",
        type=int,
        default=2400,
        help="aiperf-builtin: upper bound of search-space concurrency. Default 2400.",
    )
    g.addoption(
        "--sat-search-planner",
        default="monotonic-sla",
        choices=["monotonic-sla", "bayesian"],
        help="aiperf-builtin: which search planner to drive the sweep. Default "
        "monotonic-sla (binary-search-style; fastest for a single threshold). "
        "Use bayesian for noisier metrics.",
    )
    g.addoption(
        "--sat-search-max-iter",
        type=int,
        default=10,
        help="aiperf-builtin: max search iterations (= aiperf profile invocations "
        "inside the search). Default 10.",
    )
    # --- Optional deploy-first mode ---
    g.addoption(
        "--sat-deploy-scenario",
        default=None,
        help="Name of a router_memory scenario YAML (basename, with or without "
        ".yaml) to DEPLOY before sweeping. When set, the test stands up that "
        "DGD (template + replicas/image/router/admission per the YAML), waits "
        "for ready, runs the sweep against it, and tears it down on exit. When "
        "unset (default), behavior is unchanged: the sweep runs against a "
        "deployment that is already standing in --namespace.",
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_saturation_discovery(runtime_env, request, namespace, image):
    """Sweep concurrency, find the highest rung where p95 latency stays under SLA.

    Does not deploy. Assumes a deployment with a Frontend Service is already
    running in the chosen namespace. Each rung is an independent aiperf job
    submitted via ManagedLoad; results are collected per-rung and the test
    reports the saturation boundary.
    """
    cfg = request.config
    model = cfg.getoption("--sat-served-model")
    rung_seconds = cfg.getoption("--sat-rung-seconds")
    warmup_seconds = cfg.getoption("--sat-warmup-seconds")
    request_timeout_seconds = cfg.getoption("--sat-request-timeout-seconds")
    if request_timeout_seconds is None or request_timeout_seconds <= 0:
        request_timeout_seconds = 86400  # <=0 => effectively infinite (24h)
    max_error_pct = cfg.getoption("--sat-max-error-pct")
    sla_metric = cfg.getoption("--sat-sla-metric")
    sla_quantile = _normalize_quantile(cfg.getoption("--sat-sla-quantile"))
    sla_ms = cfg.getoption("--sat-sla-ms")
    min_success = cfg.getoption("--sat-min-success-rate")
    # --sat-adaptive is a shortcut for --sat-mode=adaptive (back-compat).
    mode = cfg.getoption("--sat-mode")
    if cfg.getoption("--sat-adaptive") and mode == "ladder":
        mode = "adaptive"
    start_c = cfg.getoption("--sat-start-concurrency")
    min_step = cfg.getoption("--sat-min-step")
    max_step = cfg.getoption("--sat-max-step")
    tolerance = cfg.getoption("--sat-tolerance")
    max_rungs = cfg.getoption("--sat-max-rungs")
    search_lo = cfg.getoption("--sat-search-lo")
    search_hi = cfg.getoption("--sat-search-hi")
    search_planner = cfg.getoption("--sat-search-planner")
    search_max_iter = cfg.getoption("--sat-search-max-iter")
    adaptive = mode == "adaptive"

    if mode == "aiperf-builtin":
        ladder = None
        _LOG.info(
            "aiperf-builtin sweep: namespace=%s model=%s search=[%d,%d] "
            "planner=%s max_iter=%d sla=%s/%s/%dms rung=%ds",
            namespace,
            model,
            search_lo,
            search_hi,
            search_planner,
            search_max_iter,
            sla_metric,
            sla_quantile,
            sla_ms,
            rung_seconds,
        )
    elif adaptive:
        ladder = None
        _LOG.info(
            "Adaptive sweep: namespace=%s model=%s start=%d sla=%s/%s/%dms "
            "tolerance=%.2f rung=%ds step=[%d,%d]",
            namespace,
            model,
            start_c,
            sla_metric,
            sla_quantile,
            sla_ms,
            tolerance,
            rung_seconds,
            min_step,
            max_step,
        )
    else:
        ladder = [
            int(c)
            for c in cfg.getoption("--sat-concurrency-ladder").split(",")
            if c.strip()
        ]
        _LOG.info(
            "Ladder sweep: namespace=%s model=%s ladder=%s sla=%s/%s/%dms rung=%ds",
            namespace,
            model,
            ladder,
            sla_metric,
            sla_quantile,
            sla_ms,
            rung_seconds,
        )

    log_dir = Path(f"/workspace/test_outputs/test_saturation_discovery_{namespace}")
    log_dir.mkdir(parents=True, exist_ok=True)

    async def _run_sweep(deployment) -> None:
        """The actual concurrency sweep. Runs against whatever deployment is
        standing in ``namespace`` (deployed by us when --sat-deploy-scenario
        is set, or pre-existing otherwise)."""
        if mode == "aiperf-builtin":
            await _run_aiperf_builtin(
                deployment=deployment,
                namespace=namespace,
                image=image,
                model=model,
                log_dir=log_dir,
                search_lo=search_lo,
                search_hi=search_hi,
                search_planner=search_planner,
                search_max_iter=search_max_iter,
                sla_metric=sla_metric,
                sla_quantile=sla_quantile,
                sla_ms=sla_ms,
                rung_seconds=rung_seconds,
                warmup_seconds=warmup_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )
            # Parse aiperf's search output → c_max.json for the wrapper.
            _write_c_max_json(
                log_dir,
                search_lo=search_lo,
                search_hi=search_hi,
                sla_ms=sla_ms,
                max_error_pct=max_error_pct,
            )
            return
        await _run_ladder_sweep(deployment)

    async def _run_ladder_sweep(deployment) -> None:
        results: list[dict] = []
        c_max = 0
        saturated_at: int | None = None

        # Iterator: produces concurrencies. Fixed ladder OR adaptive (state-driven).
        def adaptive_iter():
            c = start_c
            # going_up: True while we believe we're still under SLA
            going_up = True
            for _ in range(max_rungs):
                yield c
                # The for-loop body in the main test sets `last_p95` via closure.
                # We compute the next step here based on previous measurement.
                p95 = last_p95_box[0]
                if p95 is None:
                    break
                headroom = (sla_ms - p95) / sla_ms  # >0 = under SLA, <0 = over
                if abs(headroom) < tolerance:
                    _LOG.info(
                        f"  → within tolerance ({abs(headroom):.3f} < {tolerance}); "
                        f"saturation boundary ≈ {c}"
                    )
                    break
                # step size scales with |headroom|; bigger gap = bigger step
                step = int(max_step * abs(headroom))
                step = max(min_step, min(max_step, step))
                if headroom > 0:
                    if not going_up:
                        # We previously crossed SLA, now back under — halve step
                        # to bisect toward boundary
                        step = max(min_step, step // 2)
                    c += step
                    going_up = True
                else:
                    # Over SLA — step back, but smaller (looking for the boundary)
                    c -= max(min_step, step // 2)
                    going_up = False

        last_p95_box: list[float | None] = [None]
        iterator = adaptive_iter() if adaptive else iter(ladder)

        for level in iterator:
            _LOG.info(f"=== rung concurrency={level} ===")

            load_config = LoadConfig(
                model_name=model,
                tokenizer=model,
                concurrency=level,
                duration_minutes=rung_seconds / 60.0,
                warmup_duration=warmup_seconds,
                # Match the production seq_dist shape, same as E1 arms.
                seq_dist=(
                    "100,20:1;100,80:2;100,130:2;100,180:1;100,200:1;"
                    "500,20:2;500,80:5;500,130:5;500,180:2;500,200:1;"
                    "1000,20:3;1000,80:5;1000,130:5;1000,180:3;1000,200:2;"
                    "1600,20:5;1600,80:8;1600,130:8;1600,180:5;1600,200:3;"
                    "3400,20:3;3400,80:6;3400,130:6;3400,180:3;3400,200:2;"
                    "7000,20:2;7000,80:3;7000,130:3;7000,180:2;7000,200:1"
                ),
                num_prefix_prompts=5,
                prefix_prompt_length=740,
                streaming=True,
                ignore_eos=True,
                request_timeout_seconds=30,
                connection_reuse_strategy="never",
            )

            rung_name = f"sat-c{level}"
            ml = ManagedLoad(
                namespace=namespace,
                load_config=load_config,
                pvc_name=deployment.get_log_pvc_name(),
                pvc_run_id=deployment.get_log_run_id(),
                endpoint_url=deployment.deployment_spec.get_in_cluster_frontend_url(
                    namespace
                ),
                log_dir=str(log_dir),
                job_name=rung_name,
            )
            async with ml:
                await ml.run(wait_for_completion=True)

            # aiperf summary lives at <log_dir>/load/<job-name>/profile_export_aiperf.json
            # ManagedLoad's run() returns a dict; we also can read the file directly.
            try:
                summary_path = next(
                    (log_dir / "load").glob(f"{rung_name}-*/profile_export_aiperf.json")
                )
                with open(summary_path) as fh:
                    summary = json.load(fh)
            except (StopIteration, FileNotFoundError) as e:
                _LOG.error(f"rung c={level}: aiperf summary missing ({e})")
                saturated_at = level
                results.append({"concurrency": level, "error": "no aiperf summary"})
                break

            latency = _extract_quantile(summary, sla_metric, sla_quantile)
            total_count = (summary.get("request_count", {}) or {}).get("avg")
            err_count = (summary.get("error_request_count", {}) or {}).get("avg") or 0
            success_rate = (
                (total_count - err_count) / total_count if total_count else 0.0
            )

            # Feed back to the adaptive iterator so the next step can scale by headroom.
            last_p95_box[0] = latency

            passed_sla = latency is not None and latency <= sla_ms
            passed_success = success_rate >= min_success
            passed = passed_sla and passed_success

            _LOG.info(
                f"  c={level}: {sla_metric}/{sla_quantile}={latency:.0f}ms "
                f"(sla={sla_ms:.0f}) success_rate={success_rate:.3f} "
                f"(min={min_success}) → {'PASS' if passed else 'SATURATED'}"
            )

            results.append(
                {
                    "concurrency": level,
                    f"{sla_metric}_{sla_quantile}_ms": latency,
                    "success_rate": success_rate,
                    "total_requests": total_count,
                    "error_requests": err_count,
                    "passed_sla": passed_sla,
                    "passed_success": passed_success,
                    "passed": passed,
                }
            )

            if passed:
                c_max = level
            else:
                if saturated_at is None or level < saturated_at:
                    saturated_at = level
                _LOG.info(f"  ⚠ c={level} exceeded SLA (best C_max so far={c_max})")
                # Ladder mode: stop at first failure. Adaptive mode: keep searching
                # to bisect the boundary — the iterator's tolerance check ends it.
                if not adaptive:
                    break

        # Write a summary CSV that's easy to plot / share
        summary_path = log_dir / "saturation_sweep_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(
                {
                    "namespace": namespace,
                    "image": image,
                    "served_model": model,
                    "sla_metric": sla_metric,
                    "sla_quantile": sla_quantile,
                    "sla_ms": sla_ms,
                    "min_success_rate": min_success,
                    "rung_seconds": rung_seconds,
                    "ladder": ladder,
                    "c_max_under_sla": c_max,
                    "saturated_at": saturated_at,
                    "rungs": results,
                },
                fh,
                indent=2,
            )
        _LOG.info(f"Summary written: {summary_path}")
        _LOG.info(
            f"FINAL: C_max under SLA={c_max}, saturation observed at={saturated_at}"
        )

        # Ladder/adaptive: C_max is computed directly — write c_max.json so the
        # wrapper consumes the same artifact regardless of sweep mode.
        out = {
            "c_max": int(c_max),
            "sla_ms": float(sla_ms),
            "search_lo": int(ladder[0]) if ladder else int(search_lo),
            "search_hi": int(ladder[-1]) if ladder else int(search_hi),
            "source": str(summary_path),
        }
        cmax_path = log_dir / "c_max.json"
        with open(cmax_path, "w") as fh:
            json.dump(out, fh, indent=2)
        _LOG.info(f"c_max written: {cmax_path} = {out}")

    # --- Optional deploy-first wrapper -------------------------------------- #
    # When --sat-deploy-scenario is set, stand up that DGD here, wait for ready,
    # run the sweep against it, and tear it down on context exit. Otherwise run
    # the sweep directly against the standing deployment in --namespace.
    deploy_scenario = cfg.getoption("--sat-deploy-scenario")
    if not deploy_scenario:
        # The sweep wires ManagedLoad's pvc_name/endpoint_url from a live
        # ManagedDeployment handle, which only the deploy-first path provides.
        raise RuntimeError(
            "test_saturation_discovery requires --sat-deploy-scenario: the "
            "sweep needs a ManagedDeployment handle to wire the load job's "
            "PVC + frontend endpoint. Pass --sat-deploy-scenario <name>."
        )

    from tests.utils.managed_deployment import ManagedDeployment

    cli_cache = cfg.getoption("--model-cache-pvc", default="") or ""
    spec, scenario = _build_deploy_spec(
        deploy_scenario, image=image, model_cache_pvc=cli_cache
    )
    # Mirror run_scenario's spec prep: enable logging + PVC log collection so
    # teardown extracts pod logs the same way the router-mode suite does.
    spec.set_logging(True, "info")
    storage_class = cfg.getoption("--storage-class", default=None)
    log_pvc = scenario.deployment.log_pvc or None
    log_collection_kwargs = {
        "pvc_size": "500Mi",
        "container_log_dir": "/tmp/service_logs",
    }
    if storage_class:
        log_collection_kwargs["storage_class"] = storage_class
    if log_pvc:
        log_collection_kwargs["pvc_name"] = log_pvc
    spec.enable_log_collection(**log_collection_kwargs)

    _LOG.info(
        "deploy-first: standing up scenario=%s in namespace=%s (image=%s, "
        "model_cache_pvc=%s) before sweep",
        deploy_scenario,
        namespace,
        scenario.deployment.image,
        scenario.deployment.model_cache_pvc,
    )
    async with ManagedDeployment(
        namespace=namespace,
        log_dir=str(log_dir),
        deployment_spec=spec,
        skip_service_restart=True,
        reuse_log_pvc=bool(log_pvc),
    ) as deployment:
        await deployment.wait_for_ready(timeout=2400)
        _LOG.info("deploy-first: deployment ready; starting sweep")
        await _run_sweep(deployment)
    _LOG.info("deploy-first: sweep complete; deployment torn down")
