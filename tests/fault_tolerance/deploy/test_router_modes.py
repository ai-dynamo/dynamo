# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# YAML-driven router-mode / leak / admission test suite.
#
# Each scenario YAML in scenarios/<kind>/ describes a complete cell:
# deployment (backend, topology, units, image) × router config
# (mode + DYN_ROUTER_* knobs) × workload (shape + rungs) × reports +
# checks + expectations.
#
# A scenario's ``kind:`` field selects which test function consumes it:
#   scenarios/router_memory/    → test_router_memory
#   scenarios/admission_control/ → test_admission_control (future)
#   scenarios/endurance/        → test_endurance (future)
#
# The Event/Report/Check classes referenced by YAML are auto-discovered
# from the framework's class hierarchy at import time — no registry
# maintenance. Adding a new Event/Report/Check class anywhere in
# tests/fault_tolerance/deploy/{events,reports,checks}.py automatically
# makes it available in scenarios.

import os

import pytest

from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.fault_tolerance.deploy.scenario_lib._loader import (
    discover_scenarios,
    load_scenario,
)
from tests.fault_tolerance.deploy.scenario_lib._runtime import (
    apply_admission,
    apply_deployment,
    apply_mocker_planner_profile_fixup,
    apply_pull_secrets,
    apply_router,
    build_checks,
    build_reports,
    build_scenario_events,
)
from tests.utils.managed_deployment import DeploymentSpec

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _assert_scenario_applied(spec: DeploymentSpec, scenario) -> None:
    """Pre-deploy validation that the framework's apply_* hooks actually
    landed the YAML's declared values on the spec. Caches a clear failure
    BEFORE the long DGD-deploy cycle.

    Asserts:
      1. ``deployment.replicas`` overrides match the live spec
      2. ``deployment.image`` (if set) is on every service
      3. ``router.knobs`` are all present on the Frontend env (with str values)
      4. ``admission.knobs`` (if set) are present on every worker service
    """
    import logging

    log = logging.getLogger(scenario.path or "_assert_scenario_applied")

    # Replicas
    if scenario.deployment.replicas:
        for svc, want in scenario.deployment.replicas.items():
            got = spec[svc].replicas
            assert got == int(want), (
                f"{scenario.path}: deployment.replicas[{svc!r}]={want} but "
                f"spec[{svc!r}].replicas={got} after apply_deployment"
            )

    # Image — applied to every service when deployment.image is set
    if scenario.deployment.image:
        for svc in ["Frontend"] + spec.worker_services():
            got = spec[svc].image
            assert got == scenario.deployment.image, (
                f"{scenario.path}: deployment.image={scenario.deployment.image!r} "
                f"but spec[{svc!r}].image={got!r}"
            )

    # Router knobs — every entry must be on Frontend env, plus DYN_ROUTER_MODE
    fe_envs = {e.get("name"): str(e.get("value")) for e in spec["Frontend"].envs}
    assert fe_envs.get("DYN_ROUTER_MODE") == scenario.router.mode, (
        f"{scenario.path}: router.mode={scenario.router.mode!r} but "
        f"Frontend DYN_ROUTER_MODE={fe_envs.get('DYN_ROUTER_MODE')!r}"
    )
    for k, v in scenario.router.knobs.items():
        got = fe_envs.get(k)
        assert got == str(v), (
            f"{scenario.path}: router.knobs[{k!r}]={v!r} but "
            f"Frontend env[{k!r}]={got!r}"
        )

    # Admission knobs — every entry on every worker service
    if scenario.admission and scenario.admission.knobs:
        for svc in spec.worker_services():
            envs = {e.get("name"): str(e.get("value")) for e in spec[svc].envs}
            for k, v in scenario.admission.knobs.items():
                got = envs.get(k)
                assert got == str(v), (
                    f"{scenario.path}: admission.knobs[{k!r}]={v!r} but "
                    f"{svc} env[{k!r}]={got!r}"
                )

    # Log a summary so the test output shows what was actually applied —
    # helps diagnose "why did this arm do X?" without re-reading the YAML.
    log.info(
        "scenario applied: backend=%s topology=%s replicas=%s image=%s router.mode=%s router.knobs=%s",
        scenario.deployment.backend,
        scenario.deployment.topology,
        {svc: spec[svc].replicas for svc in ["Frontend"] + spec.worker_services()},
        scenario.deployment.image,
        scenario.router.mode,
        dict(scenario.router.knobs),
    )


def _load_template_spec(deployment) -> DeploymentSpec:
    """Resolve scenario.deployment.template + backend to a DeploymentSpec.

    Template lookup order:
      1. If deployment.template is set, use <template_dir>/<backend>/<template>.yaml
      2. Otherwise, fall back to backend-default templates:
         - disagg → <backend>/disagg_qwen3_30b_unit_prod.yaml
         - agg    → <backend>/agg_qwen3_30b_unit_prod.yaml
      3. If no local template exists, fall back to the in-repo example
         deploy yaml via DeploymentSpec.from_backend(backend, topology) —
         this is how mocker (and any other backend without local prod
         overlays) gets a DGD.
    """
    if deployment.template:
        path = os.path.join(
            _TEMPLATE_DIR, deployment.backend, f"{deployment.template}.yaml"
        )
    elif deployment.topology == "disagg":
        path = os.path.join(
            _TEMPLATE_DIR, deployment.backend, "disagg_qwen3_30b_unit_prod.yaml"
        )
    elif deployment.topology == "agg":
        path = os.path.join(
            _TEMPLATE_DIR, deployment.backend, "agg_qwen3_30b_unit_prod.yaml"
        )
    else:
        raise ValueError(f"unknown topology {deployment.topology!r}")
    if os.path.exists(path):
        return DeploymentSpec(path)

    # Fallback to /workspace/examples/backends/<backend>/deploy/<topology>.yaml
    return DeploymentSpec.from_backend(deployment.backend, deployment.topology)


async def _run_yaml_scenario(
    scenario_path, *, runtime_env=None, test_name=None, request=None
):
    """Generic YAML scenario runner. Used by every test function in this
    file; per-kind tests just pre-select a subset of scenario YAMLs and
    call this with each.

    Pre-deploy validation asserts the framework's apply_* hooks actually
    landed the YAML's declared values on the spec. This catches typos,
    schema drift, and apply_* regressions BEFORE the 10+ minute DGD
    deployment cycle.
    """
    scenario = load_scenario(scenario_path)

    # CLI override: --model-cache-pvc trumps the YAML value if non-empty.
    # Lets pytest invocations mount the HF cache PVC across all scenarios
    # without per-YAML wiring (R3 GPU sanity hit this when Qwen3-30B
    # crashlooped trying to download from HF).
    if request is not None:
        cli_cache = request.config.getoption("--model-cache-pvc", default="") or ""
        if cli_cache:
            scenario.deployment.model_cache_pvc = cli_cache
        # --image CLI override trumps the YAML's deployment.image. Same intent:
        # one pytest invocation per image candidate without rewriting YAMLs.
        cli_image = request.config.getoption("--image", default=None) or None
        if cli_image:
            scenario.deployment.image = cli_image

    spec = _load_template_spec(scenario.deployment)
    apply_deployment(spec, scenario.deployment)
    apply_router(spec, scenario.router)
    apply_admission(spec, scenario.admission)
    # Shared-cluster pull-secret wiring. Idempotent; safe on clusters
    # that don't define the secret (k8s silently ignores unknowns).
    apply_pull_secrets(spec)
    # Mocker template has a stale planner-profile-data path; override to
    # the runtime-image-shipped location. No-op for non-mocker backends.
    apply_mocker_planner_profile_fixup(spec)

    _assert_scenario_applied(spec, scenario)

    worker_names = spec.worker_services()
    if not worker_names:
        raise RuntimeError(f"DGD {scenario.deployment.backend} has no worker services")
    served_model = spec[worker_names[0]].model
    events = build_scenario_events(scenario, served_model)
    reports = build_reports(scenario)
    checks = build_checks(scenario)

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=reports,
        runtime_env=runtime_env,
        test_name=test_name,
    )


# --------------------------------------------------------------------------- #
# Per-kind test functions
# --------------------------------------------------------------------------- #
#
# Each per-kind test function pre-selects its scenarios by directory
# name. The parametrize id is the YAML filename, so test_outputs and
# pytest -k filtering are intuitive.


_ROUTER_MEMORY_SCENARIOS = discover_scenarios("router_memory")


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "scenario_path",
    _ROUTER_MEMORY_SCENARIOS,
    ids=[p.name for p in _ROUTER_MEMORY_SCENARIOS],
)
async def test_router_memory(runtime_env, request, scenario_path):
    """Memory-leak characterization across router modes × workload shapes
    × deployment topologies. One scenario YAML per cell. Reports cover
    FE memory growth, per-worker balance, request-level latency.

    Scenarios live in scenarios/router_memory/. See INDEX.md there for
    a human-readable list."""
    await _run_yaml_scenario(
        scenario_path,
        runtime_env=runtime_env,
        test_name=request.node.name,
        request=request,
    )


_ENDURANCE_SCENARIOS = discover_scenarios("endurance")


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "scenario_path",
    _ENDURANCE_SCENARIOS,
    ids=[p.name for p in _ENDURANCE_SCENARIOS],
)
async def test_endurance(runtime_env, request, scenario_path):
    """Long-running endurance scenarios. Mocker-backed by default so
    cluster cost is low and the test can run for hours.

    Designed to surface slow drift (memory leak, queue backlog) that
    only becomes visible after extended uptime. The scenario uses the
    ``PeriodicSnapshot`` event to drop timestamped artifact bundles
    into ``<log_dir>/snapshots/`` every N minutes, so analysis can
    proceed against in-progress data.

    To pair two scenarios (e.g. router-mode A vs B), launch this test
    twice in two different namespaces with the two YAML names. No
    pairing framework is provided — manual two-namespace launch keeps
    the framework simple and resource accounting clear."""
    await _run_yaml_scenario(
        scenario_path,
        runtime_env=runtime_env,
        test_name=request.node.name,
        request=request,
    )
