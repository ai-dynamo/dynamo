# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `tests.fault_tolerance.deploy.*`
# imports resolve when this harness runs as the standalone dynamo-ft uv
# project (where pytest's rootdir is this directory). Path is computed from
# __file__ so it survives moves of this file or any parent dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The harness uses repo-root-relative paths (e.g.
# ``examples/backends/{backend}/deploy/agg.yaml`` in scenarios.py). Chdir
# so those resolve whether pytest was invoked from the repo root or from
# this directory under the dynamo-ft uv project. Side-effects on CWD-
# sensitive code (e.g. ``DYN_TEST_OUTPUT_PATH`` default below) are
# intentional and match the previous repo-root-invocation behavior.
os.chdir(_REPO_ROOT)

import pytest  # noqa: E402

from tests.fault_tolerance.deploy.scenarios import scenarios  # noqa: E402


def pytest_configure(config):
    """Route this dir's tests' outputs to ``<cwd>/test_outputs/<test>/`` and
    inherit the marker registry from the root pyproject.

    The shared ``resolve_test_output_path`` defaults to
    ``/tmp/dynamo_tests/`` to keep outputs out of the git tree, which is
    fine when pytest runs on the host. Inside a dev container with the
    worktree mounted at ``/workspace``, ``/tmp`` is container-local and
    artifacts disappear from the host. We narrow the override to this
    conftest so non-k8s tests on the host keep their existing default,
    and k8s tests get one host-visible per-test directory holding load,
    services, pod yaml, test.log.txt, and the report together.

    The user-set ``DYN_TEST_OUTPUT_PATH`` always wins.
    """
    if "DYN_TEST_OUTPUT_PATH" not in os.environ:
        os.environ["DYN_TEST_OUTPUT_PATH"] = os.path.join(os.getcwd(), "test_outputs")

    # Inherit the marker registry from the root pyproject so --strict-markers
    # in the dynamo-ft uv project stays useful without duplicating the marker
    # list here. Single source of truth: root pyproject.toml.
    try:
        import tomllib  # py3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore

    root_pp = _REPO_ROOT / "pyproject.toml"
    try:
        with open(root_pp, "rb") as f:
            for marker in (
                tomllib.load(f)
                .get("tool", {})
                .get("pytest", {})
                .get("ini_options", {})
                .get("markers", [])
            ):
                config.addinivalue_line("markers", marker)
    except FileNotFoundError:
        pass  # standalone runs without the surrounding repo still ok


@pytest.fixture(autouse=True)
def _refresh_kubeconfig_if_requested():
    """Refresh the in-container kubeconfig before each test if requested.

    Gated on ``DYN_TEST_REFRESH_KUBECONFIG`` for two reasons:
      * Most environments don't need it (k3s local kubeconfigs don't
        rotate; gcloud / Nebius creds use a different model).
      * Only AWS-dev via Teleport hands out short-lived (≤7h) certs that
        a multi-hour parametrized sweep can outrun mid-run.

    Set ``DYN_TEST_REFRESH_KUBECONFIG=/path/to/script`` to run that
    script before every test. The script is expected to refresh
    ``/root/.kube/aws-dev-stripped`` (or whatever kubeconfig the dev
    container is using) in-place — see
    ``~/dynamo-dev/scripts/refresh-aws-kubeconfig.sh`` for the AWS-dev
    case. Failure is logged but does not block the test (so a
    no-network probe doesn't hard-fail the suite).
    """
    script = os.environ.get("DYN_TEST_REFRESH_KUBECONFIG")
    if script:
        import logging
        import subprocess

        logger = logging.getLogger(__name__)
        try:
            r = subprocess.run([script], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                logger.warning(
                    f"DYN_TEST_REFRESH_KUBECONFIG script {script!r} failed "
                    f"(rc={r.returncode}): {r.stderr.strip()[:500]}"
                )
            else:
                logger.info(
                    f"kubeconfig refreshed via {script}: "
                    f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'ok'}"
                )
        except Exception as e:
            logger.warning(f"DYN_TEST_REFRESH_KUBECONFIG hook failed: {e}")
    yield


@pytest.fixture(autouse=True)
def logger(request):
    """Per-test ``test.log.txt`` file handler on the root logger.

    Overrides the root ``tests/conftest.py`` fixture of the same name (pytest
    resolves the nearest definition), so the harness works identically whether
    run from the repo root or as the standalone dynamo-ft uv project. The
    output dir comes from ``DYN_TEST_OUTPUT_PATH`` (set in ``pytest_configure``
    above), so the handler always lands next to scenario logs / reports.
    """
    log_dir = Path(os.environ["DYN_TEST_OUTPUT_PATH"]) / request.node.name
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "test.log.txt", mode="w")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        handler.close()
        root.removeHandler(handler)


@pytest.fixture
def fault_pod(request):
    """Resolve the pod-selection for fault-injection tests.

    Source order: ``--fault-pod`` CLI flag → ``DYN_TEST_FAULT_POD`` env →
    default ``"0"``. Returns the raw string; tests interpret it via
    ``_parse_index_env`` (or the equivalent) so ``"all"``, ``"random"``,
    and a specific integer all work.
    """
    cli = request.config.getoption("--fault-pod")
    if cli is not None:
        return cli
    return os.environ.get("DYN_TEST_FAULT_POD", "0")


@pytest.fixture
def fault_rank(request):
    """Resolve the rank-selection for fault-injection tests. See ``fault_pod``."""
    cli = request.config.getoption("--fault-rank")
    if cli is not None:
        return cli
    return os.environ.get("DYN_TEST_FAULT_RANK", "0")


def _parse_concurrency_list(value):
    """Parse a ``--fault-concurrency`` value (CSV of ints) → list[int]."""
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out = [int(p) for p in parts]
    if not out:
        raise pytest.UsageError("--fault-concurrency cannot be empty")
    if any(n < 1 for n in out):
        raise pytest.UsageError(f"--fault-concurrency values must be >= 1, got {out}")
    return out


@pytest.fixture
def fault_concurrency(request):
    """Single concurrency value for a fault-injection scenario run.

    If ``--fault-concurrency=N1,N2,...`` is passed, tests that consume
    this fixture are auto-parametrized one-per-value via the
    ``pytest_generate_tests`` hook below (each parametrization gets its
    own ``test_outputs/<test>[N]/`` directory). The fixture itself
    just returns the int that pytest dispatched for this run.
    """
    return (
        request.param
        if hasattr(request, "param")
        else _resolve_fault_concurrency_default(request.config)
    )


def _resolve_fault_concurrency_default(config):
    cli = config.getoption("--fault-concurrency")
    raw = cli if cli is not None else os.environ.get("DYN_TEST_FAULT_CONCURRENCY", "18")
    return _parse_concurrency_list(raw)[0]


def _resolve_float_opt(config, cli_name: str, env_name: str, default: float | None):
    """CLI > env > default. Returns float or None."""
    cli = config.getoption(cli_name)
    if cli is not None:
        return float(cli)
    env = os.environ.get(env_name)
    if env is not None and env.strip():
        try:
            return float(env)
        except ValueError:
            pass
    return default


@pytest.fixture
def request_timeout_seconds(request):
    """Per-request aiperf timeout, in seconds. CLI > env > 300s default."""
    return _resolve_float_opt(
        request.config,
        "--request-timeout-seconds",
        "DYN_TEST_REQUEST_TIMEOUT_SECONDS",
        300.0,
    )


@pytest.fixture
def goodput_slos(request):
    """Resolved list of aiperf --goodput SLO strings, or None.

    Builds from (CLI > env > none) for each of:
      - request_latency:<ms>          via --goodput-request-latency-ms
      - time_to_first_token:<ms>      via --goodput-ttft-ms
      - inter_token_latency:<ms>      via --goodput-itl-ms
    Returns None when no SLOs are set (so aiperf doesn't compute goodput).
    """
    slos = []
    rl = _resolve_float_opt(
        request.config,
        "--goodput-request-latency-ms",
        "DYN_TEST_GOODPUT_REQUEST_LATENCY_MS",
        None,
    )
    if rl is not None:
        slos.append(f"request_latency:{rl}")
    ttft = _resolve_float_opt(
        request.config,
        "--goodput-ttft-ms",
        "DYN_TEST_GOODPUT_TTFT_MS",
        None,
    )
    if ttft is not None:
        slos.append(f"time_to_first_token:{ttft}")
    itl = _resolve_float_opt(
        request.config,
        "--goodput-itl-ms",
        "DYN_TEST_GOODPUT_ITL_MS",
        None,
    )
    if itl is not None:
        slos.append(f"inter_token_latency:{itl}")
    return slos or None


def _add_or_skip(parser, name, **kwargs):
    """Register an addoption, no-op if already registered by another conftest.

    Lets us declare shared options here (so the harness runs standalone under
    the dynamo-ft uv project) without conflicting when the root
    ``tests/conftest.py`` is also loaded (e.g., when pytest is invoked from the
    repo root).
    """
    try:
        parser.addoption(name, **kwargs)
    except ValueError:
        pass  # already registered upstream


# Shared CLI options (--image, --namespace, --skip-service-restart) are also
# defined in tests/conftest.py. We re-declare them here (via _add_or_skip)
# so the harness works standalone under the dynamo-ft uv project, where the
# root conftest is not loaded.
def pytest_addoption(parser):
    # ---- shared options (also in root tests/conftest.py) ----
    _add_or_skip(
        parser,
        "--image",
        type=str,
        default=None,
        help="Container image to use for deployment (overrides YAML default).",
    )
    _add_or_skip(
        parser,
        "--namespace",
        type=str,
        default=None,
        help="Kubernetes namespace for deployment.",
    )
    _add_or_skip(
        parser,
        "--skip-service-restart",
        action="store_true",
        default=None,
        help="Skip restarting NATS and etcd services before deployment.",
    )
    # ---- per-test-file CLI options ----
    # Each test file that exposes `add_cli_options(parser)` gets its options
    # registered here. Keeps each file's flags scoped (own pytest --help
    # group) and avoids one giant addoption block.
    try:
        from tests.fault_tolerance.deploy import test_overload as _overload_mod

        _overload_mod.add_cli_options(parser)
    except Exception:
        # Test module not importable in this environment — skip gracefully.
        pass
    try:
        from tests.fault_tolerance.deploy import (
            test_worker_death_kv_transfer as _wdkt_mod,
        )

        _wdkt_mod.add_cli_options(parser)
    except Exception:
        pass
    try:
        from tests.fault_tolerance.deploy import test_endurance as _endurance_mod

        _endurance_mod.add_cli_options(parser)
    except Exception:
        pass

    # ---- fault_tolerance-specific options ----
    parser.addoption(
        "--client-type",
        type=str,
        default=None,
        choices=["aiperf", "legacy"],
        help="Client type for load generation: 'aiperf' (default) or 'legacy'",
    )
    parser.addoption(
        "--baseline-concurrency",
        type=int,
        default=8,
        help="Initial concurrency for the cascade-console long-running driver "
        "(test_cascade_console.py). Adjust on the fly via "
        "`python -m tests.utils.cascade_inject load <N>`.",
    )
    parser.addoption(
        "--include-custom-build",
        action="store_true",
        default=False,
        help="Include tests that require custom builds (e.g., MoE models). "
        "By default, these tests are excluded.",
    )
    parser.addoption(
        "--storage-class",
        type=str,
        default=None,
        help="Storage class for PVC log collection (must support RWX). "
        "If not specified, uses cluster default.",
    )
    parser.addoption(
        "--log-pvc",
        type=str,
        default=None,
        help="Reuse a named RWX PVC for log collection across multiple "
        "tests / DGDs. If the PVC does not exist it is created on first "
        "use; once created the framework never deletes it. Each test "
        "writes to a unique sub-path inside the PVC, which is wiped "
        "in-place after a successful extract. Useful on clusters where "
        "PVC provisioning is slow (e.g. AWS FSx ≈ 7 min). Without this "
        "flag the framework creates a fresh per-test PVC and deletes "
        "it at teardown (current default behaviour).",
    )
    parser.addoption(
        "--recreate-log-pvc",
        action="store_true",
        default=False,
        help="When --log-pvc is set, delete the existing PVC first and "
        "create a fresh one. Use this when the standing PVC is stuck "
        "in Terminating, has accumulated other tests' data you no "
        "longer want to keep, or you want to change its storage "
        "class. Requires --log-pvc.",
    )
    parser.addoption(
        "--model-pvc",
        type=str,
        default=None,
        help="Reuse an existing RWX PVC as the HuggingFace model cache "
        "(mounted at /model-cache, exported as HF_HOME on workers). "
        "Skips re-downloading large model weights between runs. The "
        "PVC must already exist in the test namespace and be RWX. "
        "Many shared clusters auto-provision a 'shared-model-cache' "
        "PVC per namespace.",
    )
    parser.addoption(
        "--prefetch-model",
        action="store_true",
        default=False,
        help="Before applying the DGD, schedule a one-shot Job that "
        "downloads every non-frontend service's model into the model "
        "cache. Avoids the concurrent-download lock thrash that "
        "happens when N worker pods all hit HuggingFace simultaneously "
        "on a cold cache. Idempotent — skips models already present. "
        "Requires --model-pvc.",
    )
    parser.addoption(
        "--restart-services",
        action="store_true",
        default=False,
        help="Restart NATS and etcd before each test. "
        "Default is to skip restart (faster iteration).",
    )
    parser.addoption(
        "--fault-pod",
        type=str,
        default=None,
        help="Target pod selection for fault-injection scenarios. "
        "Accepts an integer index (e.g. '0'), 'random', or 'all'. "
        "Overrides the DYN_TEST_FAULT_POD env var; default is '0' "
        "(pod-0 of the targeted service).",
    )
    parser.addoption(
        "--fault-rank",
        type=str,
        default=None,
        help="Target rank/process selection within each pod for rank-level "
        "fault-injection scenarios. Same grammar as --fault-pod: integer, "
        "'random', or 'all'. Overrides DYN_TEST_FAULT_RANK; default is '0'.",
    )
    parser.addoption(
        "--fault-concurrency",
        type=str,
        default=None,
        help="Sustained-load concurrency for fault-injection scenarios. "
        "Accepts a single integer (e.g. '4096') or a comma-separated list "
        "('18,256,4096') for chained runs of the same scenario at "
        "different load levels. Overrides DYN_TEST_FAULT_CONCURRENCY; "
        "default is '18' (near-saturation on the N=3 prod-mirror DGD).",
    )
    parser.addoption(
        "--request-timeout-seconds",
        type=float,
        default=None,
        help="Per-request timeout passed to aiperf (--request-timeout-seconds). "
        "Requests exceeding this are cancelled client-side and counted in "
        "error_request_count. Overrides DYN_TEST_REQUEST_TIMEOUT_SECONDS.",
    )
    parser.addoption(
        "--goodput-ttft-ms",
        type=float,
        default=None,
        help="TTFT goodput SLO in ms (--goodput time_to_first_token:<ms>). "
        "Requests are counted as 'good' only if TTFT ≤ this AND every other "
        "configured goodput SLO holds. Overrides DYN_TEST_GOODPUT_TTFT_MS.",
    )
    parser.addoption(
        "--goodput-itl-ms",
        type=float,
        default=None,
        help="Inter-token latency goodput SLO in ms (--goodput "
        "inter_token_latency:<ms>). Overrides DYN_TEST_GOODPUT_ITL_MS.",
    )
    parser.addoption(
        "--goodput-request-latency-ms",
        type=float,
        default=None,
        help="End-to-end request_latency goodput SLO in ms (--goodput "
        "request_latency:<ms>). Overrides DYN_TEST_GOODPUT_REQUEST_LATENCY_MS.",
    )


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests and apply markers based on scenario properties.

    This hook applies markers to individual test instances based on their scenario:
    - @pytest.mark.custom_build: For MoE models and other tests requiring custom builds
    """
    if "scenario" in metafunc.fixturenames:
        scenario_names = list(scenarios.keys())
        argvalues = []
        ids = []

        for scenario_name in scenario_names:
            scenario_obj = scenarios[scenario_name]
            marks = []

            if getattr(scenario_obj, "requires_custom_build", False):
                marks.append(pytest.mark.custom_build)

            # Always use pytest.param for type consistency (even with empty marks)
            argvalues.append(pytest.param(scenario_name, marks=marks))
            ids.append(scenario_name)

        metafunc.parametrize("scenario_name", argvalues, ids=ids)

    # Expand --fault-concurrency=N1,N2,... into one parametrization per
    # value for any test that asks for the ``fault_concurrency`` fixture.
    # Single-value (default) still produces one test invocation as
    # ``test[18]``; multi-value chains as ``test[18]/[256]/[4096]`` etc.
    if "fault_concurrency" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--fault-concurrency")
        if raw is None:
            raw = os.environ.get("DYN_TEST_FAULT_CONCURRENCY", "18")
        values = _parse_concurrency_list(raw)
        metafunc.parametrize(
            "fault_concurrency", values, indirect=True, ids=[str(n) for n in values]
        )


def pytest_collection_modifyitems(config, items):
    """Automatically deselect custom_build tests unless --include-custom-build is specified.

    This allows users to run tests without any special flags and automatically excludes
    tests that require custom builds. To include them, use --include-custom-build.

    Note: If user explicitly uses -m marker filtering, we respect that and don't
    auto-deselect, allowing them to run custom_build tests with -m "custom_build".
    """
    # If --include-custom-build flag is set, include all tests
    if config.getoption("--include-custom-build"):
        return

    # If user explicitly used -m marker filtering, let pytest handle it
    # Don't auto-deselect in this case
    if config.option.markexpr:
        return

    # Default case: auto-deselect custom_build tests
    deselected = []
    selected = []

    for item in items:
        if "custom_build" in item.keywords:
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture
def image(request):
    return request.config.getoption("--image")


@pytest.fixture
def namespace(request):
    """Get Kubernetes namespace from CLI option, with fault-tolerance-specific default."""
    value = request.config.getoption("--namespace")
    return value if value is not None else "fault-tolerance-test"


@pytest.fixture
def client_type(request):
    """Get client type from command line or use scenario default."""
    return request.config.getoption("--client-type")


@pytest.fixture
def skip_service_restart(request):
    """Whether to skip restarting NATS and etcd services.

    Default: True (skip restart — services are assumed to be running).
    Pass --restart-services to opt in to a clean-state restart.
    The legacy --skip-service-restart flag is still honored if explicitly set.
    """
    if request.config.getoption("--restart-services", default=False):
        return False  # don't skip = do restart
    skip = request.config.getoption("--skip-service-restart", default=None)
    if skip is not None:
        return skip
    return True  # default: skip restart


@pytest.fixture
def storage_class(request):
    """Storage class for PVC log collection (must support RWX)."""
    return request.config.getoption("--storage-class")


@pytest.fixture
def log_pvc(request):
    """Existing RWX PVC name to reuse for log collection (skips create/delete)."""
    return request.config.getoption("--log-pvc")


@pytest.fixture
def recreate_log_pvc(request):
    """Drop + recreate the --log-pvc PVC before this run."""
    return request.config.getoption("--recreate-log-pvc")


@pytest.fixture
def model_pvc(request):
    """Existing RWX PVC name to reuse as the HF model cache (mounted on workers)."""
    return request.config.getoption("--model-pvc")


@pytest.fixture
def prefetch_model(request):
    """Run a one-shot model-download Job before applying the DGD."""
    return request.config.getoption("--prefetch-model")


@pytest.fixture
def runtime_env(
    namespace,
    image,
    skip_service_restart,
    storage_class,
    log_pvc,
    recreate_log_pvc,
    model_pvc,
    prefetch_model,
):
    """Bundle of every CLI-driven runtime knob a scenario needs.

    Tests take just this one fixture and forward it to
    ``run_scenario(runtime_env=...)`` instead of declaring seven
    individual fixtures. Adding a new CLI flag means: define an
    addoption + fixture above, add the field below, and update
    ``run_scenario`` to read it from the bundle. Test signatures
    stay unchanged.
    """
    from tests.fault_tolerance.deploy.scenario import RuntimeEnv

    return RuntimeEnv(
        namespace=namespace,
        image=image,
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
        log_pvc=log_pvc,
        recreate_log_pvc=recreate_log_pvc,
        model_pvc=model_pvc,
        prefetch_model=prefetch_model,
    )
