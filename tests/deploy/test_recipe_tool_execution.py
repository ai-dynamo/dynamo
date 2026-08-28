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
import time
import uuid
from typing import Any, Optional

import pytest
import requests

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.utils.client import wait_for_model_availability
from tests.utils.tool_calling import (
    assert_chained_tools_thread_real_output,
    assert_executes_real_tool_and_uses_output,
)

openai = pytest.importorskip("openai")
OpenAI = openai.OpenAI

logger = logging.getLogger(__name__)

# Flags that make a Dynamo frontend/worker emit OpenAI `tool_calls`. Either the
# frontend declares the parser directly, or the worker declares it and the
# frontend picks it up from the runtime config registered at discovery time.
_TOOL_PARSER_FLAGS = ("--dyn-tool-call-parser", "--tool-call-parser")


def _declared_tool_call_parser(spec: DeploymentSpec) -> Optional[str]:
    """Return the tool-call parser a manifest configures, or None.

    Scans every service's args, since which component carries the flag differs
    between the frontend-parser and worker-declared topologies.
    """
    for service in spec.services:
        try:
            args = service._get_args() or []
        except Exception:  # noqa: BLE001 - a malformed service must not abort the scan
            continue
        for i, arg in enumerate(args):
            for flag in _TOOL_PARSER_FLAGS:
                if arg == flag and i + 1 < len(args):
                    return args[i + 1]
                if arg.startswith(f"{flag}="):
                    return arg.split("=", 1)[1]
    return None


def _served_model(spec: DeploymentSpec) -> Optional[str]:
    """Best-effort model name from the manifest's worker args.

    Only about a third of the recipe corpus passes ``--model`` on the command
    line; the rest configure it through an engine config file or ConfigMap. For
    those, ``_model_from_endpoint`` reads it back off the running deployment
    instead, which is authoritative anyway.
    """
    for service in spec.services:
        try:
            if service.model:
                return service.model
        except Exception:  # noqa: BLE001
            continue
    return None


def _model_from_endpoint(
    base_url: str, attempts: int = 30, delay: float = 10.0
) -> Optional[str]:
    """Read the served model id back off a running frontend's /v1/models.

    Polls, because the frontend answers before any worker has registered and
    reports an empty list until one has.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=30)
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
        time.sleep(delay)
    return None


@pytest.mark.framework_only
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.e2e
@pytest.mark.timeout(2400)
async def test_recipe_executes_tools_end_to_end(
    request: pytest.FixtureRequest,
    image: Optional[str],
    namespace: str,
    record_property: Any,
):
    """Deploy --recipe, then prove the model actually uses real tool output."""
    recipe = request.config.getoption("--recipe")
    if not recipe:
        pytest.skip("--recipe not provided; nothing to deploy")

    deployment_spec = DeploymentSpec(recipe)

    parser = _declared_tool_call_parser(deployment_spec)
    if parser is None:
        pytest.skip(
            f"{recipe} configures no tool-call parser "
            f"(looked for {' / '.join(_TOOL_PARSER_FLAGS)}), so the frontend "
            "cannot emit tool_calls. This is a deployment precondition, not a "
            "defect -- re-run against a recipe that enables tool calling."
        )

    model_hint = request.config.getoption("--recipe-model") or _served_model(
        deployment_spec
    )

    if image:
        deployment_spec.set_image(image)

    # Unique name so concurrent runs against one cluster do not collide.
    deployment_spec.name = f"{deployment_spec.name}-tx-{uuid.uuid4().hex[:6]}"

    record_property("recipe", recipe)
    record_property("tool_call_parser", parser)

    logger.info(
        "Deploying recipe=%s name=%s namespace=%s model=%s parser=%s",
        recipe,
        deployment_spec.name,
        namespace,
        model_hint or "<from /v1/models>",
        parser,
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        readiness_timeout=request.config.getoption("--recipe-deploy-timeout"),
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

        model = model_hint or _model_from_endpoint(base_url)
        assert model, (
            f"could not determine the served model for {recipe}: it is not in "
            "the manifest args and /v1/models never reported one. Pass "
            "--recipe-model explicitly."
        )
        record_property("model", model)

        assert wait_for_model_availability(
            url=base_url,
            endpoint=deployment_spec.endpoint,
            model=model,
            logger=logger,
            max_attempts=30,
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
