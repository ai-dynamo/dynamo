# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upgrade acceptance for unchanged requests across SGLang releases."""

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion

from tests.deploy.dgd_utils import ManagedDeployment
from tests.deploy.n2_utils import (
    DEFAULT_RELEASE_LINE,
    MODELS,
    VersionPair,
    compatibility_spec,
    runtime_version,
    version_matrix,
)
from tests.utils.client import wait_for_model_availability
from tests.utils.output_paths import resolve_test_output_path

ROOT = Path(__file__).resolve().parents[2]
FLAG = "DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION"
logger = logging.getLogger(__name__)


def _validate_images(baseline, target):
    versions = []
    for role, image in (("baseline", baseline), ("target", target)):
        if not image:
            raise pytest.UsageError(f"The {role} SGLang runtime image is required")
        if "@" in image and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", image.partition("@")[2]
        ):
            raise pytest.UsageError(
                f"Invalid {role} image digest; expected sha256:<64 hex digits>"
            )
        try:
            versions.append(tuple(map(int, runtime_version(image).split("."))))
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
    if versions[0] >= versions[1]:
        raise pytest.UsageError("Target runtime version must be newer than baseline")
    baseline_digest = baseline.partition("@")[2]
    if baseline_digest and baseline_digest == target.partition("@")[2]:
        raise pytest.UsageError("Baseline and target must have different image digests")


def pytest_generate_tests(metafunc):
    config = metafunc.config
    if "upgrade_images" not in metafunc.fixturenames or not config.getoption(
        "--run-strict-schema-upgrade"
    ):
        return
    if any(
        config.getoption(f"--schema-upgrade-{role}-image")
        for role in ("baseline", "target")
    ):
        return
    candidate = config.getoption("--image")
    if not candidate:
        raise pytest.UsageError("N-2 upgrade acceptance requires --image")
    releases = json.loads((ROOT / "tests/deploy/n2/releases.json").read_text())
    line = os.environ.get("N2_RELEASE_LINE") or DEFAULT_RELEASE_LINE
    pairs = version_matrix(releases, line, candidate, candidate)[1::2]
    metafunc.parametrize(
        "upgrade_images",
        [(pair.worker, candidate) for pair in pairs],
        indirect=True,
        ids=[pair.name for pair in pairs],
    )


def _model_cache(config):
    pvc = config.getoption("--schema-upgrade-model-cache-pvc")
    mount = config.getoption("--schema-upgrade-model-cache-mount")
    return (
        pvc or config.getoption("--model-cache-pvc", default=None),
        mount or config.getoption("--model-cache-mount", default="/models"),
    )


@pytest.fixture
def upgrade_images(request):
    config = request.config
    if not config.getoption("--run-strict-schema-upgrade"):
        pytest.skip(
            "Requires explicit --run-strict-schema-upgrade and versioned images"
        )
    images = getattr(request, "param", None) or tuple(
        config.getoption(f"--schema-upgrade-{version}-image")
        for version in ("baseline", "target")
    )
    _validate_images(*images)
    if not config.getoption("--namespace"):
        raise pytest.UsageError("Upgrade acceptance requires an explicit --namespace")
    if not _model_cache(config)[0]:
        raise pytest.UsageError(
            "Upgrade acceptance requires a prepared model-cache PVC"
        )
    return images


def _upgrade_spec(image, enabled, name, pvc, mount):
    spec = compatibility_spec(
        ROOT, VersionPair(name, image, image), "chat", name, pvc, mount
    )
    frontend = next(
        c for c in spec.spec()["spec"]["components"] if c["name"] == "Frontend"
    )
    # Authored args replace the operator's defaults, including -m dynamo.frontend.
    frontend["podTemplate"]["spec"]["containers"][0]["command"] = [
        "python3",
        "-m",
        "dynamo.frontend",
    ]
    for flag, value in (
        ("--dyn-chat-processor", "sglang"),
        ("--tool-call-parser", "qwen25"),
        ("--reasoning-parser", "qwen3"),
    ):
        spec.add_arg_to_service("Frontend", flag, value)
    spec.set_service_env_var("Frontend", FLAG, "1" if enabled else "0")
    spec.add_arg_to_service("decode", "--max-total-tokens", "2048")
    spec.add_arg_to_service("decode", "--mem-fraction-static", "0.9")
    return spec


def _assert_completion(result):
    assert result["status"] == 200, f"HTTP {result['status']}: {result['body'][:500]}"
    body = json.loads(result["body"])
    assert "error" not in body, body
    completion = ChatCompletion.model_validate(body, strict=True)
    assert completion.model == MODELS["chat"][0], "Unexpected served model"
    assert len(completion.choices) == 1, "Expected one completion"
    assert completion.usage is not None, "Missing usage"
    assert completion.usage.completion_tokens > 0, "No generated tokens"


async def _probe(image, enabled, payload, name, namespace, pvc, mount, output):
    output.mkdir(parents=True)
    spec = _upgrade_spec(image, enabled, name, pvc, mount)
    spec.save(str(output / "deployment.yaml"))
    managed = ManagedDeployment(
        str(output),
        spec,
        namespace,
        skip_service_restart=True,
        readiness_timeout=1200,
        fail_fast_startup=True,
    )
    async with managed as deployment:
        pods = await asyncio.to_thread(deployment.get_pods, ["Frontend"])
        assert len(pods["Frontend"]) == 1, "Expected one frontend Pod"
        pod = pods["Frontend"][0]
        forward = await asyncio.to_thread(deployment.port_forward, pod, 8000)
        assert forward is not None, "Frontend port forwarding failed"
        endpoint = "/v1/chat/completions"
        assert await asyncio.to_thread(
            wait_for_model_availability,
            f"http://localhost:{forward.local_port}",
            endpoint,
            payload["model"],
            logger,
        ), "Model did not become available"
        response, _ = await asyncio.to_thread(
            deployment.send_request_with_port_forward_retry,
            pod=pod,
            remote_port=8000,
            endpoint=endpoint,
            payload=payload,
            timeout=120,
            port_forward=forward,
        )
        try:
            result = {"status": response.status_code, "body": response.text}
        finally:
            response.close()
        (output / "response.json").write_text(json.dumps(result, indent=2) + "\n")
    # ManagedDeployment logs deletion timeouts; do not start the next image
    # while an old worker still holds the GPU.
    remaining = await asyncio.to_thread(managed.get_pods)
    assert not any(remaining.values()), "Previous deployment still has remaining Pods"
    return result


@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.core
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.timeout(4500)
async def test_strict_schema_upgrade(request, upgrade_images):
    baseline, target = upgrade_images
    namespace = request.config.getoption("--namespace")
    pvc, mount = _model_cache(request.config)
    run_id = uuid.uuid4().hex[:8]
    output = Path(resolve_test_output_path(request.node.name)) / run_id
    output.mkdir(parents=True)
    payload = {
        "model": MODELS["chat"][0],
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": 64,
        "temperature": 0,
    }
    (output / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = {"model_revision": MODELS["chat"][1], "stages": {}}
    failures = []
    for stage, image, enabled in (
        ("baseline", baseline, False),
        ("default", target, False),
        ("compat", target, True),
    ):
        probe_finished = False
        try:
            result = await _probe(
                image,
                enabled,
                payload,
                f"schema-upgrade-{run_id}-{stage}",
                namespace,
                pvc,
                mount,
                output / stage,
            )
            probe_finished = True
        finally:
            if not probe_finished:
                # Isolation is uncertain after a deployment or cleanup failure.
                request.session.shouldstop = (
                    "Schema upgrade deployment failed; "
                    "remaining pairs are not validated"
                )
        report["stages"][stage] = {"image": image, "enabled": enabled, **result}
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            _assert_completion(result)
        except (AssertionError, ValueError) as error:
            failures.append(f"{stage}: {error}")
            if stage == "baseline":
                break
    assert (
        not failures
    ), f"Upgrade compatibility failed; evidence: {output}\n" + "\n".join(failures)
