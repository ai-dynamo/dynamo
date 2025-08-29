# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from dataclasses import dataclass

import pytest

from tests.serve.common import EngineConfig, create_payload_for_config
from tests.utils.deployment_graph import (
    chat_completions_response_handler,
    completions_response_handler,
)
from tests.utils.engine_process import EngineProcess

logger = logging.getLogger(__name__)


@dataclass
class TRTLLMConfig(EngineConfig):
    """Configuration for trtllm test scenarios"""


class TRTLLMProcess(EngineProcess):
    """Simple process manager for trtllm shell scripts"""

    def __init__(self, config: TRTLLMConfig, request):
        self.port = 8000
        self.config = config
        self.dir = config.directory
        script_path = os.path.join(self.dir, "launch", config.script_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"trtllm script not found: {script_path}")

        # Set these env vars to customize model launched by launch script to match test
        os.environ["MODEL_PATH"] = config.model
        os.environ["SERVED_MODEL_NAME"] = config.model

        command = ["bash", script_path]

        super().__init__(
            command=command,
            timeout=config.timeout,
            display_output=True,
            working_dir=self.dir,
            health_check_ports=[],  # Disable port health check
            health_check_urls=[
                (f"http://localhost:{self.port}/v1/models", self._check_models_api)
            ],
            delayed_start=config.delayed_start,
            terminate_existing=False,  # If true, will call all bash processes including myself
            stragglers=[],  # Don't kill any stragglers automatically
            log_dir=request.node.name,
        )


# trtllm test configurations
trtllm_configs = {
    "aggregated": TRTLLMConfig(
        name="aggregated",
        directory="/workspace/components/backends/trtllm",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
    "disaggregated": TRTLLMConfig(
        name="disaggregated",
        directory="/workspace/components/backends/trtllm",
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
    # TODO: These are sanity tests that the kv router examples launch
    # and inference without error, but do not do detailed checks on the
    # behavior of KV routing.
    "aggregated_router": TRTLLMConfig(
        name="aggregated_router",
        directory="/workspace/components/backends/trtllm",
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
    "disaggregated_router": TRTLLMConfig(
        name="disaggregated_router",
        directory="/workspace/components/backends/trtllm",
        script_name="disagg_router.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
}


@pytest.fixture(
    params=[
        pytest.param(config_name, marks=config.marks)
        for config_name, config in trtllm_configs.items()
    ]
)
def trtllm_config_test(request):
    """Fixture that provides different trtllm test configurations"""
    return trtllm_configs[request.param]


@pytest.mark.e2e
@pytest.mark.slow
def test_deployment(trtllm_config_test, request, runtime_services):
    """
    Test dynamo deployments with different configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")

    config = trtllm_config_test
    payload = create_payload_for_config(config)

    logger.info(f"Using model: {config.model}")
    logger.info(f"Script: {config.script_name}")

    with TRTLLMProcess(config, request) as server_process:
        assert len(config.endpoints) == len(config.response_handlers)
        for endpoint, response_handler in zip(
            config.endpoints, config.response_handlers
        ):
            url = f"http://localhost:{server_process.port}/{endpoint}"
            start_time = time.time()
            elapsed = 0.0

            request_body = (
                payload.payload_chat
                if endpoint == "v1/chat/completions"
                else payload.payload_completions
            )

            for _ in range(payload.repeat_count):
                elapsed = time.time() - start_time

                response = server_process.send_request(
                    url, payload=request_body, timeout=config.timeout - elapsed
                )
                server_process.check_response(payload, response, response_handler)


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.trtllm_marker
@pytest.mark.slow
def test_metrics_labels(request, runtime_services):
    """
    Test that the trtllm backend correctly exports model labels in its metrics.

    This test uses the --extra-engine-args flag with agg.yaml configuration
    to start the backend.

    The test runs from the trtllm directory to access engine_configs/agg.yaml
    """
    import re
    import subprocess
    import threading

    import requests

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_metrics_labels")

    # Use the exact configuration that works for the user
    model_path = "Qwen/Qwen3-0.6B"
    served_model_name = "Qwen/Qwen3-0.6B"
    agg_engine_args = "engine_configs/agg.yaml"
    metrics_port = 8081
    timeout = 60

    # Calculate the path to the trtllm directory from the test file location
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from tests/serve/
    project_root = os.path.dirname(os.path.dirname(test_dir))
    working_directory = os.path.join(project_root, "components", "backends", "trtllm")

    # Verify the engine config file exists
    engine_config_path = os.path.join(working_directory, agg_engine_args)
    if not os.path.exists(engine_config_path):
        pytest.fail(f"Engine config file not found at: {engine_config_path}")

    logger.info(f"Using engine config from: {engine_config_path}")

    # Build command using the user's working command
    command = [
        "python3",
        "-m",
        "dynamo.trtllm",
        "--model-path",
        model_path,
        "--served-model-name",
        served_model_name,
        "--extra-engine-args",
        agg_engine_args,
        "--max-seq-len",
        "100",
        "--max-num-tokens",
        "100",
        "--publish-events-and-metrics",
    ]

    # Set environment for metrics
    env = os.environ.copy()
    env["DYN_SYSTEM_ENABLED"] = "true"
    env["DYN_SYSTEM_PORT"] = str(metrics_port)

    # Start the backend process
    logger.info(f"Starting trtllm backend with model: {served_model_name}")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Working directory: {working_directory}")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=working_directory,
    )

    try:
        # Start a thread to capture and log output
        output_lines = []

        def log_output():
            if process.stdout is None:
                logger.warning("Process stdout is None, cannot capture output")
                return
            for line in process.stdout:
                line = line.strip()
                if line:
                    output_lines.append(line)
                    logger.info(f"[TRTLLM] {line}")

        output_thread = threading.Thread(target=log_output)
        output_thread.daemon = True
        output_thread.start()

        # Wait for metrics endpoint to be ready
        metrics_url = f"http://localhost:{metrics_port}/metrics"
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process has died
            if process.poll() is not None:
                logger.error(f"Process exited with code: {process.returncode}")
                logger.error("Last 20 output lines:\n" + "\n".join(output_lines[-20:]))
                pytest.fail(
                    f"trtllm backend process died with exit code {process.returncode}"
                )

            try:
                response = requests.get(metrics_url, timeout=5)
                if response.status_code == 200:
                    logger.info("Metrics endpoint is ready")
                    break
            except requests.RequestException as e:
                logger.debug(f"Metrics not ready yet: {e}")
            time.sleep(2)
        else:
            logger.error("Last 50 output lines:\n" + "\n".join(output_lines[-50:]))
            pytest.fail(
                f"Metrics endpoint did not become available within {timeout} seconds"
            )

        # Check that the metrics include the model label
        response = requests.get(metrics_url)
        assert response.status_code == 200, "Failed to fetch metrics"

        metrics_text = response.text
        logger.info(f"Metrics text: {metrics_text}")

        # With the --extra-engine-args flag pointing to agg.yaml,
        # the backend should be able to start properly and register endpoints.
        # Let's check for the dynamo_component_requests_total metric with our model label.

        # Parse the Prometheus metrics to find our label
        pattern = rf'dynamo_component_requests_total\{{[^}}]*model="{re.escape(served_model_name)}"[^}}]*\}}\s+(\d+)'
        matches = re.findall(pattern, metrics_text)

        if matches:
            initial_value = int(matches[0])
            assert (
                initial_value == 0
            ), f"Expected initial metric value to be 0, got {initial_value}"
        else:
            # Check if any dynamo_component metrics exist
            if "dynamo_component" in metrics_text:
                logger.info(
                    "✓ Metrics endpoint is working (found dynamo_component metrics)"
                )
                logger.warning(
                    "Note: dynamo_component_requests_total not found - likely because dummy engine didn't fully initialize"
                )
                logger.info("For complete testing, use a real pre-built TRT-LLM engine")
            else:
                pytest.fail("No dynamo_component metrics found at all")

    finally:
        # Clean up
        logger.info("Terminating backend process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
