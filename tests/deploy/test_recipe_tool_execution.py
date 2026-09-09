# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tool execution against a live DynamoGraphDeployment.

Runs the same scenarios as ``tests/frontend/test_tool_calling_sglang.py``, but
against a recipe deployed on a real cluster instead of locally spawned
processes. Both the recipe and the cluster are chosen at run time:

```bash
KUBECONFIG=/path/to/kubeconfig python -m pytest \\
  tests/deploy/test_recipe_tool_execution.py \\
  --recipe recipes/gpt-oss-120b/vllm/agg-h200-agentic/deploy.yaml \\
  --namespace dynamo -m "k8s and deploy" -v -s
```

Nothing about the scenarios is Kubernetes-specific -- they need only an
OpenAI-compatible client and a model name (see ``tests/utils/tool_calling.py``).
The tools they invoke run as subprocesses **in the pytest process**, not in the
cluster, which is what keeps them meaningful regardless of where Dynamo runs.

Two gates decide whether this module can say anything at all about a recipe:

* **Precondition** -- the manifest must configure a tool-call parser. Without
  one the frontend cannot emit ``tool_calls`` and a failure would say nothing
  about the model or the deployment. Missing parser => skip, not fail.
* **Capability** -- chaining two tools is a property of the *model*, not of the
  serving stack. Qwen3-0.6B and Qwen3-8B both fail it while handling the
  protocol perfectly; Kimi-K2.5 passes. It is therefore recorded as a
  capability result rather than asserted, so pointing this module at a weaker
  recipe reports the limitation instead of manufacturing a red build.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from typing import Any, Callable, NamedTuple, Optional

import pytest
import requests
import yaml

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.utils.client import wait_for_model_availability
from tests.utils.tool_calling import (
    assert_chained_tools_thread_real_output,
    assert_executes_real_tool_and_uses_output,
)

openai = pytest.importorskip("openai")
OpenAI = openai.OpenAI

logger = logging.getLogger(__name__)

# Budgets for everything this test waits on *after* the DGD reports Ready. The
# outer pytest timeout is the sum of these plus the readiness budget, computed
# in ``tests/deploy/conftest.py`` -- see ``POST_READY_BUDGET`` below.
_MODEL_DISCOVERY_BUDGET = 300.0
# Passed straight to ``wait_for_model_availability``; its own worst case is
# these timeouts plus its fixed inter-attempt sleeps. Kept as data rather than
# an attempt count so the budget is checkable -- see
# ``test_availability_probe_stays_within_its_budget``.
_AVAILABILITY_ATTEMPT_TIMEOUTS = [20.0] * 10
_AVAILABILITY_BUDGET = 300.0
# The two tool scenarios, plus port-forward setup and teardown.
_SCENARIO_BUDGET = 240.0
_TIMEOUT_SLACK = 120.0

#: Seconds this module can spend after readiness. ``pytest_collection_modifyitems``
#: reads this off the module and sets ``timeout = --recipe-deploy-timeout + this``.
POST_READY_BUDGET = int(
    _MODEL_DISCOVERY_BUDGET + _AVAILABILITY_BUDGET + _SCENARIO_BUDGET + _TIMEOUT_SLACK
)

# Flags that make a Dynamo frontend/worker emit OpenAI `tool_calls`. Either the
# frontend declares the parser directly, or the worker declares it and the
# frontend picks it up from the runtime config registered at discovery time.
_TOOL_PARSER_FLAGS = ("--dyn-tool-call-parser", "--tool-call-parser")

# Matches the flag whether it is a discrete argv token (`["--dyn-tool-call-parser",
# "qwen25"]`), an `=`-joined token, or embedded in a shell-style command string
# (`sh -c "python3 -m dynamo.vllm --dyn-tool-call-parser deepseek_v4 \\ ..."`).
# The last form is not exotic: 34 of the 101 parser-bearing recipe manifests use
# it, disproportionately the `-agentic` profiles this module exists to exercise.
# Scanning argv tokens alone reported those as having no parser at all.
_TOOL_PARSER_RE = re.compile(
    r"(?:{})[=\s]+([^\s\\'\"]+)".format("|".join(_TOOL_PARSER_FLAGS))
)


class ParserScan(NamedTuple):
    """Outcome of looking for a tool-call parser in a manifest.

    Three-valued on purpose. "No parser configured" and "could not read this
    manifest's arguments" are different claims, and collapsing them produces a
    skip message that asserts something false about the recipe.
    """

    parser: Optional[str]
    unreadable: tuple[str, ...]  # services whose args could not be parsed

    @property
    def undetermined(self) -> bool:
        return self.parser is None and bool(self.unreadable)


def _declared_tool_call_parser(spec: DeploymentSpec) -> ParserScan:
    """Find the tool-call parser a manifest configures.

    Scans every service, since which component carries the flag differs between
    the frontend-parser and worker-declared topologies. Matches both argv-token
    and shell-string forms -- see ``_TOOL_PARSER_RE``.

    ``ServiceSpec._get_args()`` shlex-splits the container command and raises on
    manifests with unbalanced quotes, which real recipes contain. Rather than
    dropping such a service (and then reporting the recipe as having no parser),
    fall back to scanning that service's raw container spec, and record the
    service as unreadable so the caller can say "undetermined" instead of "absent".
    """
    unreadable: list[str] = []
    for service in spec.services:
        name = getattr(service, "name", "<unnamed>")
        try:
            haystack = " ".join(str(arg) for arg in (service._get_args() or []))
        except Exception:  # noqa: BLE001 - unbalanced quotes are real
            # Degrade to the raw container spec rather than dropping the service.
            unreadable.append(name)
            try:
                haystack = str(service._main_container() or "")
            except Exception:  # noqa: BLE001
                continue
        match = _TOOL_PARSER_RE.search(haystack)
        if match:
            return ParserScan(match.group(1), tuple(unreadable))
    return ParserScan(None, tuple(unreadable))


# ``--model $MODEL_PATH`` / ``--model "${MODEL_ID}"``: the manifest names an
# environment variable the deployment expands at pod start, so the literal text
# is not a model id. 47 of the 178 recipes that declare a DGD do this.
_UNRESOLVED_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _resolve_model(
    explicit: Optional[str],
    discover: Callable[[], Optional[str]],
    manifest: Optional[str],
) -> Optional[str]:
    """Decide which model id to send, in order of authority.

    1. ``--recipe-model``, because the operator said so.
    2. ``/v1/models``, because it reports what the frontend actually advertises
       after the deployment expanded whatever the manifest left unresolved.
    3. the manifest, and only when it is a literal rather than a ``${VAR}``.

    The manifest used to come first. A truthy ``"${MODEL_PATH}"`` then
    suppressed the endpoint lookup and every request named a model the frontend
    had never heard of.

    ``discover`` is a callable rather than a value so the authority order is
    real: passing ``_model_from_endpoint(base_url)`` positionally still *ran*
    the poll before this function could prefer ``explicit`` over it, spending
    the whole discovery budget on a result that was then discarded.
    """
    if explicit:
        return explicit
    discovered = discover()
    if discovered:
        return discovered
    if manifest and not _UNRESOLVED_VAR.search(manifest):
        return manifest
    return None


def _resolve_served_model(
    model_hint: Optional[str],
    base_url: str,
    manifest_model: Optional[str],
    budget: float = _MODEL_DISCOVERY_BUDGET,
) -> Optional[str]:
    """The call site's model resolution, as one callable.

    Named so it can be tested without a cluster: the property that matters --
    an explicit ``--recipe-model`` never pays for endpoint discovery -- is a
    property of *this composition*, not of ``_resolve_model`` alone.
    """
    return _resolve_model(
        model_hint, lambda: _model_from_endpoint(base_url, budget), manifest_model
    )


def _served_model(spec: DeploymentSpec) -> Optional[str]:
    """Best-effort model name from the manifest's worker args.

    Weaker than ``/v1/models`` in two ways, so it is only a fallback. Only about
    a third of the corpus passes ``--model`` on the command line at all; and of
    those that do, many pass an unexpanded ``${MODEL_PATH}``. Returning that
    string would be worse than returning nothing, because a truthy value
    suppresses the endpoint lookup and the test then sends requests naming a
    model that does not exist.
    """
    for service in spec.services:
        try:
            if service.model:
                return service.model
        except Exception:  # noqa: BLE001
            continue
    return None


def _model_from_endpoint(
    base_url: str, budget: float = _MODEL_DISCOVERY_BUDGET, delay: float = 10.0
) -> Optional[str]:
    """Read the served model id back off a running frontend's /v1/models.

    Polls, because the frontend answers before any worker has registered and
    reports an empty list until one has.

    Bounded by wall clock, not by an attempt count. Each attempt costs its
    request timeout *plus* the sleep, so ``attempts * delay`` understates the
    real worst case by the entire request budget: the previous 30 attempts of a
    30s request and a 10s sleep was 1200s, half the outer timeout on its own,
    while reading as if it were 300.
    """
    deadline = time.monotonic() + budget
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                timeout=max(1.0, min(30.0, deadline - time.monotonic())),
            )
            response.raise_for_status()
            entries = (response.json() or {}).get("data") or []
        except (requests.RequestException, ValueError) as exc:
            logger.debug("attempt %d: /v1/models not ready: %s", attempt, exc)
            entries = []
        for entry in entries:
            model_id = (entry or {}).get("id")
            if model_id:
                logger.info("discovered served model %r from /v1/models", model_id)
                return model_id
        time.sleep(max(0.0, min(delay, deadline - time.monotonic())))
    logger.warning("/v1/models reported no model within %.0fs", budget)
    return None


@pytest.mark.framework_only
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.e2e
# No @pytest.mark.timeout here on purpose. pytest-timeout resolves a marker
# ahead of the --timeout command line option, so a static marker is a ceiling
# nobody can raise: with a hardcoded 2400 the readiness budget alone
# (--recipe-deploy-timeout, default 1800) plus the post-ready waits could exceed
# it, and any --recipe-deploy-timeout above 2400 could never finish. The timeout
# is derived from both budgets in tests/deploy/conftest.py.
async def test_recipe_executes_tools_end_to_end(
    request: pytest.FixtureRequest,
    image: Optional[str],
    namespace: str,
    skip_service_restart: bool,
    record_property: Any,
):
    """Deploy --recipe, then prove the model actually uses real tool output."""
    recipe = request.config.getoption("--recipe")
    if not recipe:
        pytest.skip("--recipe not provided; nothing to deploy")

    deployment_spec = DeploymentSpec(recipe)

    scan = _declared_tool_call_parser(deployment_spec)
    if scan.undetermined:
        pytest.skip(
            f"could not determine whether {recipe} configures a tool-call "
            f"parser: the arguments of service(s) {', '.join(scan.unreadable)} "
            "could not be parsed. Skipping without claiming the recipe lacks a "
            "parser -- pass --recipe-model and re-run, or fix the manifest."
        )
    if scan.parser is None:
        pytest.skip(
            f"{recipe} configures no tool-call parser "
            f"(looked for {' / '.join(_TOOL_PARSER_FLAGS)}), so the frontend "
            "cannot emit tool_calls. This is a deployment precondition, not a "
            "defect -- re-run against a recipe that enables tool calling."
        )
    parser = scan.parser

    # Only an explicit --recipe-model overrides the endpoint. The manifest is a
    # weaker source: 47 of the 178 recipes pass --model "${MODEL_PATH}" or
    # similar, and the deployment resolves that from the environment at pod
    # start. Letting the manifest win means sending requests with the literal
    # string "${MODEL_PATH}" as the model id.
    model_hint = request.config.getoption("--recipe-model")
    manifest_model = _served_model(deployment_spec)

    if image:
        deployment_spec.set_image(image)

    # Unique names so concurrent runs against one cluster do not collide. This
    # has to cover the companions too, not just the DGD: they are applied and
    # deleted by name, 14 of the names in recipes/ are shared between manifests,
    # and __aenter__ deletes before it creates -- so a run starting up would
    # otherwise delete a running one's ConfigMaps and ComputeDomains.
    renames = deployment_spec.uniquify(f"-tx-{uuid.uuid4().hex[:6]}")
    logger.info("Per-run resource names: %s", renames)

    record_property("recipe", recipe)
    record_property("tool_call_parser", parser)

    logger.info(
        "Deploying recipe=%s name=%s namespace=%s model=%s parser=%s",
        recipe,
        deployment_spec.name,
        namespace,
        model_hint or manifest_model or "<from /v1/models>",
        parser,
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        readiness_timeout=request.config.getoption("--recipe-deploy-timeout"),
        # ManagedDeployment defaults this to False, which restarts the
        # namespace-wide dynamo-platform-{nats,etcd} StatefulSets and deletes
        # their PVCs -- interrupting every other deployment sharing the
        # namespace. The shared fixture defaults to True for deploy tests; take
        # it rather than inheriting the destructive default.
        skip_service_restart=skip_service_restart,
    ) as deployment:
        frontend_pods = deployment.get_pods([deployment.frontend_service_name]).get(
            deployment.frontend_service_name, []
        )
        assert frontend_pods, f"no frontend pods for {deployment_spec.name}"

        port_forward = deployment.port_forward(frontend_pods[0], deployment_spec.port)
        assert port_forward is not None, (
            f"failed to port-forward to {frontend_pods[0].name}:"
            f"{deployment_spec.port}"
        )

        base_url = f"http://localhost:{port_forward.local_port}"
        logger.info("Frontend reachable at %s", base_url)

        # Resolution is lazy inside: an explicit --recipe-model must not pay
        # for a poll whose answer it outranks.
        model = _resolve_served_model(model_hint, base_url, manifest_model)
        assert model, (
            f"could not determine the served model for {recipe}: /v1/models "
            f"never reported one and the manifest says {manifest_model!r}. "
            "Pass --recipe-model explicitly."
        )
        record_property("model", model)

        assert wait_for_model_availability(
            url=base_url,
            endpoint=deployment_spec.endpoint,
            model=model,
            logger=logger,
            max_attempts=len(_AVAILABILITY_ATTEMPT_TIMEOUTS),
            attempt_timeouts=_AVAILABILITY_ATTEMPT_TIMEOUTS,
        ), f"model {model} never became available at {base_url}"

        client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")

        # Requirement: a real subprocess runs and the model reports back a
        # secret that appears in no prompt. Any deployment serving a
        # tool-calling model must satisfy this.
        assert_executes_real_tool_and_uses_output(client, model)
        logger.info("single-tool execution: PASS")

        # Capability probe: threading one tool's real output into the next
        # call. Recorded, not asserted -- see the module docstring.
        try:
            assert_chained_tools_thread_real_output(client, model)
            chained = "pass"
            logger.info("chained-tool execution: PASS")
        except AssertionError as exc:
            chained = f"unsupported: {exc}"
            logger.warning(
                "chained-tool execution: NOT SUPPORTED by %s -- %s", model, exc
            )
        record_property("chained_tool_capability", chained)


# ---------------------------------------------------------------------------
# Unit coverage for the precondition scan (no cluster required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "args, expected",
    [
        (["--model", "m", "--dyn-tool-call-parser", "qwen25"], "qwen25"),
        (["--dyn-tool-call-parser=qwen25"], "qwen25"),
        (["--tool-call-parser", "hermes"], "hermes"),
        # Shell-style: the whole command is one token. 34 of 101 parser-bearing
        # recipe manifests look like this; scanning argv tokens alone missed them.
        (
            ["python3 -m dynamo.vllm --model m --dyn-tool-call-parser deepseek_v4 \\"],
            "deepseek_v4",
        ),
        (["--model", "m"], None),
        ([], None),
    ],
)
def test_tool_call_parser_scan_matches_argv_and_shell_forms(tmp_path, args, expected):
    """The scan must see the flag in argv-token, `=`-joined and shell-string form."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "scan-test"},
        "spec": {
            "components": [
                {
                    "name": "VllmDecodeWorker",
                    "type": "worker",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "img",
                                    "command": ["python3"],
                                    "args": list(args),
                                }
                            ]
                        }
                    },
                }
            ]
        },
    }
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(manifest))

    scan = _declared_tool_call_parser(DeploymentSpec(str(path)))

    assert scan.parser == expected
    assert not scan.undetermined


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_unparseable_args_report_undetermined_not_absent(tmp_path):
    """An unparseable service must not be reported as "configures no parser".

    `ServiceSpec._get_args()` shlex-splits and raises on unbalanced quotes, which
    real recipes contain. Claiming absence there asserts something false about
    the recipe -- the exact false-green this module's precondition exists to avoid.
    """
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "unparseable"},
        "spec": {
            "components": [
                {
                    "name": "VllmDecodeWorker",
                    "type": "worker",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "img",
                                    "command": ["sh", "-c"],
                                    # Unbalanced quote, and no parser anywhere.
                                    "args": ['python3 -m dynamo.vllm --model "m'],
                                }
                            ]
                        }
                    },
                }
            ]
        },
    }
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(manifest))

    scan = _declared_tool_call_parser(DeploymentSpec(str(path)))

    assert scan.parser is None
    assert scan.undetermined, "must say 'undetermined', not 'no parser configured'"
    assert "VllmDecodeWorker" in scan.unreadable


# ---------------------------------------------------------------------------
# Unit coverage for model resolution (no cluster required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "value, unresolved",
    [
        ("${MODEL_PATH}", True),
        ("$MODEL_PATH", True),
        ("$model_path", True),
        ("${MODEL_ID}", True),
        ("Qwen/Qwen3-0.6B", False),
        ("moonshotai/Kimi-K3", False),
        ("nvidia/model-v1.2", False),
    ],
)
def test_unexpanded_variables_are_recognised(value, unresolved):
    """A manifest value naming an environment variable is not a model id.

    47 of the 178 recipes that declare a DGD pass one; the deployment expands it
    at pod start, so the literal text never identifies a model.
    """
    assert bool(_UNRESOLVED_VAR.search(value)) is unresolved


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_the_endpoint_outranks_the_manifest():
    """`/v1/models` is authoritative; the manifest is a last resort.

    Previously the manifest value was tried first, so a truthy `"${MODEL_PATH}"`
    suppressed the endpoint lookup entirely and the test then asked the frontend
    for a model named `"${MODEL_PATH}"`.
    """

    def resolve(explicit, endpoint, manifest):
        return _resolve_model(explicit, lambda: endpoint, manifest)

    # The case the reviewer found: Kimi K3 declares ${MODEL_PATH} but advertises
    # its real id through ${SERVED_MODEL_NAME}.
    assert resolve(None, "moonshotai/Kimi-K3", "${MODEL_PATH}") == "moonshotai/Kimi-K3"
    # An explicit --recipe-model still wins over everything.
    assert resolve("override/model", "moonshotai/Kimi-K3", None) == "override/model"
    # The manifest is used only when the endpoint says nothing and it is literal.
    assert resolve(None, None, "Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"
    # An unexpanded variable is never used, even as a last resort.
    assert resolve(None, None, "${MODEL_PATH}") is None


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_the_deployment_test_takes_the_shared_restart_fixture():
    """`ManagedDeployment` defaults `skip_service_restart` to False, which
    restarts namespace-wide NATS and etcd and deletes their PVCs. Every other
    deploy test takes the shared fixture, which defaults to True; this one must
    too, or one recipe run interrupts every other deployment in the namespace.
    """
    import inspect

    params = inspect.signature(test_recipe_executes_tools_end_to_end).parameters
    assert "skip_service_restart" in params


# ---------------------------------------------------------------------------
# Unit coverage for the wait budgets (no cluster required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_an_explicit_model_skips_endpoint_discovery(monkeypatch):
    """`--recipe-model` must not pay for a poll whose answer it outranks.

    Exercised through `_resolve_served_model`, the composition the deploy test
    actually calls, because the defect lived in the composition: passing
    `_model_from_endpoint(base_url)` as a positional argument ran the poll
    before `_resolve_model` could prefer the explicit value, so an operator who
    named the model still waited out the whole discovery budget and the result
    was then discarded.
    """
    calls: list[str] = []

    def spy(url, **kwargs):
        calls.append(url)
        raise AssertionError("endpoint discovery must not run")

    monkeypatch.setattr(requests, "get", spy)

    assert (
        _resolve_served_model("override/model", "http://frontend", "${MODEL_PATH}")
        == "override/model"
    )
    assert calls == [], f"discovery ran anyway: {calls}"


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_model_discovery_stays_within_its_budget(monkeypatch):
    """Discovery must be bounded by wall clock, not by an attempt count.

    Counting attempts hid the request timeout: 30 attempts of a 30s request plus
    a 10s sleep reads like 300s and costs 1200s. Simulate the worst case -- every
    request burning its full timeout, every sleep taken -- on a virtual clock.
    """
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda s: now.__setitem__(0, now[0] + s))

    def timing_out(url, timeout=None, **kwargs):
        now[0] += timeout  # the request consumed its entire allowance
        raise requests.ConnectionError("frontend not up")

    monkeypatch.setattr(requests, "get", timing_out)

    budget = 300.0
    assert _model_from_endpoint("http://frontend", budget=budget) is None
    assert now[0] <= budget + 1.0, f"discovery overran its budget: {now[0]}s"


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_availability_probe_stays_within_its_budget(monkeypatch):
    """The availability probe must fit the budget the outer timeout reserves.

    Measured against the real `wait_for_model_availability` rather than against
    a restatement of its schedule: its inter-attempt sleeps are internal
    constants, so a locally computed worst case would silently stop matching it.
    """
    import tests.utils.client as client_module

    elapsed = [0.0]
    monkeypatch.setattr(
        client_module.time, "sleep", lambda s: elapsed.__setitem__(0, elapsed[0] + s)
    )

    def timing_out(url, json=None, timeout=None, headers=None, **kwargs):
        elapsed[0] += timeout
        raise requests.ConnectionError("no worker yet")

    monkeypatch.setattr(client_module.requests, "post", timing_out)

    assert not wait_for_model_availability(
        url="http://frontend",
        endpoint="/v1/chat/completions",
        model="m",
        logger=logger,
        max_attempts=len(_AVAILABILITY_ATTEMPT_TIMEOUTS),
        attempt_timeouts=_AVAILABILITY_ATTEMPT_TIMEOUTS,
    )
    assert elapsed[0] <= _AVAILABILITY_BUDGET, (
        f"availability probe worst case is {elapsed[0]}s, over its "
        f"{_AVAILABILITY_BUDGET}s budget"
    )


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_the_outer_timeout_covers_every_inner_budget():
    """The derived outer timeout must exceed readiness plus every post-ready wait.

    The regression: a hardcoded `timeout(2400)` sat under readiness (1800) +
    discovery (1200) + availability (680) = 3680, so a valid slow recipe was
    killed mid-wait, and because pytest-timeout prefers a marker over the
    `--timeout` option, no command line could raise the ceiling.
    """
    from tests.deploy.conftest import pytest_collection_modifyitems

    for deploy_timeout in (600, 1800, 5400):
        markers: list[Any] = []
        item = type(
            "Item",
            (),
            {
                "module": sys.modules[__name__],
                "add_marker": lambda self, m: markers.append(m),
            },
        )()
        config = type(
            "Config", (), {"getoption": lambda self, name, default=None: deploy_timeout}
        )()

        pytest_collection_modifyitems(config, [item])

        assert len(markers) == 1, "the recipe test must get a derived timeout"
        outer = markers[0].args[0]
        inner = (
            deploy_timeout
            + _MODEL_DISCOVERY_BUDGET
            + _AVAILABILITY_BUDGET
            + _SCENARIO_BUDGET
        )
        assert outer > inner, (
            f"--recipe-deploy-timeout={deploy_timeout} yields outer={outer}s "
            f"but the inner waits can reach {inner}s"
        )
