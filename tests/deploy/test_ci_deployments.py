# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI deployment tests - pytest version of pr.yaml deploy tests.

These tests replicate the functionality of the shell-based deploy tests in
.github/workflows/pr.yaml (deploy-test-vllm, deploy-test-sglang, deploy-test-trtllm jobs).

Each test:
1. Applies a DynamoGraphDeployment manifest
2. Waits for all pods to be ready
3. Tests model availability via /v1/models API
4. Tests inference via /v1/chat/completions API
5. Validates response format and content
6. Cleans up the deployment

Run these tests with:
    pytest -m "deploy and pre_merge" tests/deploy/

Or for a specific framework:
    pytest -m "deploy and pre_merge and vllm" tests/deploy/
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import requests
import yaml

from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

logger = logging.getLogger(__name__)


@dataclass
class DeployConfig:
    """Configuration for a deployment test scenario."""

    name: str
    """Unique identifier for this test (e.g., 'vllm-agg')"""

    framework: str
    """Framework name: vllm, sglang, or trtllm"""

    profile: str
    """Profile name: agg, agg_router, disagg, or disagg_router"""

    deployment_file: str
    """Relative path to deployment YAML from repo root"""

    min_pods: int
    """Minimum expected number of pods (for validation)"""

    marks: List
    """Pytest marks for this configuration"""

    description: str = ""
    """Human-readable description"""


def _get_workspace_dir() -> Path:
    """Get workspace directory (repo root)."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find workspace directory (no pyproject.toml found)")


# Define all deployment configurations
# This replaces the matrix strategy in pr.yaml
DEPLOY_CONFIGS: Dict[str, DeployConfig] = {
    # vLLM configurations
    "vllm-agg": DeployConfig(
        name="vllm-agg",
        framework="vllm",
        profile="agg",
        deployment_file="examples/backends/vllm/deploy/agg.yaml",
        min_pods=2,  # 1 frontend + 1 worker
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.vllm,
            pytest.mark.gpu_1,
        ],
        description="vLLM Aggregated: Single worker handling prefill and decode",
    ),
    "vllm-agg_router": DeployConfig(
        name="vllm-agg_router",
        framework="vllm",
        profile="agg_router",
        deployment_file="examples/backends/vllm/deploy/agg_router.yaml",
        min_pods=3,  # 1 frontend + 2 workers
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.vllm,
            pytest.mark.gpu_1,
        ],
        description="vLLM Aggregated with Router: Multiple replicas with KV routing",
    ),
    "vllm-disagg": DeployConfig(
        name="vllm-disagg",
        framework="vllm",
        profile="disagg",
        deployment_file="examples/backends/vllm/deploy/disagg.yaml",
        min_pods=3,  # 1 frontend + 1 prefill + 1 decode
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.vllm,
            pytest.mark.gpu_2,
        ],
        description="vLLM Disaggregated: Separate prefill and decode workers",
    ),
    "vllm-disagg_router": DeployConfig(
        name="vllm-disagg_router",
        framework="vllm",
        profile="disagg_router",
        deployment_file="examples/backends/vllm/deploy/disagg_router.yaml",
        min_pods=5,  # 1 frontend + 2 prefill + 2 decode
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.vllm,
            pytest.mark.gpu_4,
        ],
        description="vLLM Disaggregated with Router: Multiple replicas with KV routing",
    ),
    # SGLang configurations
    "sglang-agg": DeployConfig(
        name="sglang-agg",
        framework="sglang",
        profile="agg",
        deployment_file="examples/backends/sglang/deploy/agg.yaml",
        min_pods=2,
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.sglang,
            pytest.mark.gpu_1,
        ],
        description="SGLang Aggregated: Single worker",
    ),
    "sglang-agg_router": DeployConfig(
        name="sglang-agg_router",
        framework="sglang",
        profile="agg_router",
        deployment_file="examples/backends/sglang/deploy/agg_router.yaml",
        min_pods=3,
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.sglang,
            pytest.mark.gpu_1,
        ],
        description="SGLang Aggregated with Router",
    ),
    # TensorRT-LLM configurations
    "trtllm-agg": DeployConfig(
        name="trtllm-agg",
        framework="trtllm",
        profile="agg",
        deployment_file="examples/backends/trtllm/deploy/agg.yaml",
        min_pods=2,
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.gpu_1,
        ],
        description="TensorRT-LLM Aggregated",
    ),
    "trtllm-agg_router": DeployConfig(
        name="trtllm-agg_router",
        framework="trtllm",
        profile="agg_router",
        deployment_file="examples/backends/trtllm/deploy/agg_router.yaml",
        min_pods=3,
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.gpu_1,
        ],
        description="TensorRT-LLM Aggregated with Router",
    ),
    "trtllm-disagg": DeployConfig(
        name="trtllm-disagg",
        framework="trtllm",
        profile="disagg",
        deployment_file="examples/backends/trtllm/deploy/disagg.yaml",
        min_pods=3,
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.gpu_2,
        ],
        description="TensorRT-LLM Disaggregated",
    ),
    "trtllm-disagg_router": DeployConfig(
        name="trtllm-disagg_router",
        framework="trtllm",
        profile="disagg_router",
        deployment_file="examples/backends/trtllm/deploy/disagg_router.yaml",
        min_pods=5,
        marks=[
            pytest.mark.deploy,
            pytest.mark.k8s,
            pytest.mark.e2e,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.gpu_4,
        ],
        description="TensorRT-LLM Disaggregated with Router",
    ),
}


async def wait_for_model_available(
    url: str,
    model_name: str,
    max_attempts: int = 30,
    sleep_seconds: int = 5,
) -> bool:
    """Wait for model to appear in /v1/models endpoint.

    Args:
        url: Base URL (e.g., http://localhost:8000)
        model_name: Expected model name
        max_attempts: Maximum polling attempts
        sleep_seconds: Sleep duration between attempts

    Returns:
        True if model is available

    Raises:
        TimeoutError: If model doesn't become available within max_attempts
    """
    models_url = f"{url}/v1/models"

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                models_url,
                timeout=10,
                headers={"accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            # Check if model is in the response
            if "data" in data:
                for model_entry in data["data"]:
                    if model_entry.get("id") == model_name:
                        logger.info(f"Model {model_name} is available in /v1/models")
                        return True

            logger.info(
                f"Waiting for model {model_name} to be available in /v1/models... "
                f"(attempt {attempt}/{max_attempts})"
            )

        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt}/{max_attempts}): {e}")

        if attempt < max_attempts:
            await asyncio.sleep(sleep_seconds)

    raise TimeoutError(
        f"Model {model_name} not found in /v1/models after {max_attempts} attempts"
    )


async def validate_chat_completion(
    url: str,
    model_name: str,
    min_content_length: int = 100,
) -> Dict:
    """Send chat completion request and validate response.

    Replicates the validation logic from pr.yaml deploy tests (lines 431-466).

    Args:
        url: Base URL (e.g., http://localhost:8000)
        model_name: Model name to use
        min_content_length: Minimum expected content length in response

    Returns:
        Response JSON dictionary

    Raises:
        AssertionError: If validation fails
        requests.RequestException: If request fails
    """
    completion_url = f"{url}/v1/chat/completions"

    # Payload matches pr.yaml lines 434-445
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, "
                    "lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria "
                    "was buried beneath the shifting sands of time, lost to the world for centuries. You are "
                    "an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled "
                    "upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has "
                    "the potential to reshape the very fabric of reality. Your journey will take you through "
                    "treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: "
                    "Character Background: Develop a detailed background for your character. Describe their "
                    "motivations for seeking out Aeloria, their skills and weaknesses, and any personal "
                    "connections to the ancient city or its legends. Are they driven by a quest for knowledge, "
                    "a search for lost familt clue is hidden."
                ),
            }
        ],
        "stream": False,
        "max_tokens": 30,
        "temperature": 0.0,
    }

    # Send request with retry (matches pr.yaml --retry flags)
    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = requests.post(
                completion_url,
                json=payload,
                headers={
                    "accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Request failed, retrying in {retry_delay}s: {e}")
                await asyncio.sleep(retry_delay)
            else:
                raise

    data = response.json()
    logger.info(f"Response: {json.dumps(data, indent=2)}")

    # Validation checks (matches pr.yaml lines 448-466)

    # 1. Check if response is valid JSON (already done by response.json())

    # 2. Check message role is "assistant"
    assert (
        data.get("choices", [{}])[0].get("message", {}).get("role") == "assistant"
    ), f"Expected message role 'assistant', got: {data.get('choices', [{}])[0].get('message', {}).get('role')}"

    # 3. Check model name matches
    assert (
        data.get("model") == model_name
    ), f"Expected model '{model_name}', got: {data.get('model')}"

    # 4. Check content length
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    assert (
        len(content) > min_content_length
    ), f"Expected content length > {min_content_length}, got: {len(content)}"

    logger.info("✅ Test passed: Response matches expected format and content")
    return data


@pytest.mark.parametrize(
    "config_name",
    list(DEPLOY_CONFIGS.keys()),
    ids=list(DEPLOY_CONFIGS.keys()),
)
async def test_deploy_inference(
    request,
    config_name: str,
    deploy_namespace: str,
    runtime_image: Optional[str],
    model_name: str,
    kubeconfig_path,  # Ensures kubeconfig is set up
):
    """Test deployment and inference for various profiles.

    This test replicates the functionality of the deploy-test-* jobs in pr.yaml.

    Steps:
    1. Load deployment YAML
    2. Update image and namespace
    3. Apply deployment via ManagedDeployment
    4. Wait for pods to be ready
    5. Port forward to frontend
    6. Wait for model availability
    7. Test chat completion
    8. Validate response
    9. Clean up (automatic via ManagedDeployment context manager)

    Args:
        config_name: Configuration name from DEPLOY_CONFIGS
        deploy_namespace: Kubernetes namespace
        runtime_image: Runtime container image
        model_name: Model name for inference
        kubeconfig_path: Path to kubeconfig (fixture ensures it's set)
    """
    config = DEPLOY_CONFIGS[config_name]

    # Apply marks from config
    for mark in config.marks:
        request.node.add_marker(mark)

    logger.info(f"=" * 80)
    logger.info(f"Starting test: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"Framework: {config.framework}")
    logger.info(f"Profile: {config.profile}")
    logger.info(f"Namespace: {deploy_namespace}")
    logger.info(f"Runtime Image: {runtime_image}")
    logger.info(f"Model: {model_name}")
    logger.info(f"=" * 80)

    # Get workspace directory
    workspace = _get_workspace_dir()
    deployment_file = workspace / config.deployment_file

    if not deployment_file.exists():
        pytest.skip(f"Deployment file not found: {deployment_file}")

    # Load and configure deployment spec
    deployment_spec = DeploymentSpec(str(deployment_file))

    # Update namespace
    deployment_spec.namespace = deploy_namespace

    # Update runtime image if provided
    if runtime_image:
        logger.info(f"Setting runtime image: {runtime_image}")
        deployment_spec.set_image(runtime_image)
    else:
        logger.warning("No runtime image provided, using image from YAML")

    # Create log directory for this test
    log_dir = Path(request.node.name.replace("::", "_"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Deploy and test
    async with ManagedDeployment(
        namespace=deploy_namespace,
        log_dir=str(log_dir),
        deployment_spec=deployment_spec,
    ) as deployment:
        logger.info(f"Deployment {deployment_spec.name} applied successfully")

        # Get frontend URL (port forwarding is handled by ManagedDeployment)
        frontend_url = f"http://localhost:{deployment_spec.port}"
        logger.info(f"Frontend URL: {frontend_url}")

        # Wait for model to be available
        logger.info(f"Waiting for model {model_name} to be available...")
        await wait_for_model_available(frontend_url, model_name)

        # Test chat completion and validate response
        logger.info("Testing chat completion...")
        response = await validate_chat_completion(frontend_url, model_name)

        logger.info(f"✅ Test {config.name} completed successfully")

    # ManagedDeployment context manager handles cleanup
    logger.info(f"Deployment {deployment_spec.name} cleaned up")
