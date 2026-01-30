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

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Pattern

from typing_extensions import Required, TypedDict

from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

if TYPE_CHECKING:
    from tests.fault_tolerance.deploy.base_checker import BaseChecker


# Import checker factory (actual import, not TYPE_CHECKING)
def _get_checkers_for_scenario(
    scenario_name: str, scenario: "Scenario"
) -> List["BaseChecker"]:
    """Lazy import to avoid circular dependencies during module initialization."""
    from tests.fault_tolerance.deploy.checker_factory import get_checkers_for_scenario

    return get_checkers_for_scenario(scenario_name, scenario)


class TestPhase(Enum):
    """Enum representing different test phases in fault tolerance testing."""

    STANDARD = auto()
    OVERFLOW = auto()
    RECOVERY = auto()


class DeploymentInfo(TypedDict, total=False):
    """Information about a deployment configuration.

    Attributes:
        spec: DeploymentSpec object defining the deployment configuration
        backend: Backend type - "vllm", "sglang", or "trtllm"
        model: Optional model identifier (e.g., "deepseek-ai/DeepSeek-V2-Lite")
        is_moe: Optional flag indicating if this is a Mixture-of-Experts model
    """

    spec: Required[DeploymentSpec]
    backend: Required[str]
    model: str
    is_moe: bool


# Test phase suffixes derived from TestPhase enum
OVERFLOW_SUFFIX = f"_{TestPhase.OVERFLOW.name.lower()}"
RECOVERY_SUFFIX = f"_{TestPhase.RECOVERY.name.lower()}"

# Worker name mapping for different backends
WORKER_MAP = {
    "vllm": {
        "decode": "VllmDecodeWorker",
        "prefill": "VllmPrefillWorker",
    },
    "sglang": {
        "decode": "decode",
        "prefill": "prefill",
    },
    "trtllm": {
        "decode": "TRTLLMDecodeWorker",
        "decode_agg": "TRTLLMWorker",  # Aggregated uses different name
        "prefill": "TRTLLMPrefillWorker",
    },
}

# Process ready patterns for recovery detection
WORKER_READY_PATTERNS: Dict[str, Pattern] = {
    # Frontend
    "Frontend": re.compile(r"added model"),
    # vLLM workers
    "VllmDecodeWorker": re.compile(
        r"VllmWorker for (?P<model_name>.*?) has been initialized"
    ),
    "VllmPrefillWorker": re.compile(
        r"VllmWorker for (?P<model_name>.*?) has been initialized"
    ),
    # SGLang workers - look for their specific initialization messages
    "decode": re.compile(
        r"Model registration succeeded|Decode worker handler initialized|Worker handler initialized"
    ),
    "prefill": re.compile(
        r"Model registration succeeded|Prefill worker handler initialized|Worker handler initialized"
    ),
    # TensorRT-LLM workers
    "TRTLLMWorker": re.compile(
        r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
    ),
    "TRTLLMDecodeWorker": re.compile(
        r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
    ),
    "TRTLLMPrefillWorker": re.compile(
        r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
    ),
}


def get_all_worker_types() -> list[str]:
    """Get all worker type names for both vLLM and SGLang."""
    worker_types = ["Frontend"]
    for backend in WORKER_MAP.values():
        worker_types.extend(backend.values())
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for x in worker_types:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def get_worker_ready_pattern(worker_name: str) -> Optional[Pattern]:
    """Get the ready pattern for a specific worker type."""
    return WORKER_READY_PATTERNS.get(worker_name)


def get_backend_workers(backend: str) -> Dict[str, str]:
    """Get worker mapping for a specific backend."""
    return WORKER_MAP.get(backend, {})


@dataclass
class Load:
    clients: int = 10
    requests_per_client: int = 150
    input_token_length: int = 100
    output_token_length: int = 100
    max_retries: int = 3  # Increased for fault tolerance
    sla: Optional[float] = None
    client_type: str = "aiperf"  # "aiperf" or "legacy"
    max_request_rate: float = 1.0  # Rate limiting for legacy client (requests/sec)
    success_threshold: float = 90.0  # Success rate threshold for tests

    # For mixed token testing (overflow + recovery)
    mixed_token_test: bool = False
    overflow_token_length: Optional[int] = None  # Tokens for overflow requests
    overflow_request_count: int = 15  # Number of overflow requests
    normal_request_count: int = 15  # Number of normal requests after overflow

    continuous_load: bool = (
        False  # If True, use continuous load instead of fixed request count
    )


@dataclass
class Failure(ABC):
    """Base class for all failure types."""

    # time to wait in seconds before the failure is injected
    time: int

    # names of DGD services to inject the failure into the corresponding pods for
    service_names: list[str]

    @abstractmethod
    async def execute(
        self, deployment: ManagedDeployment, logger: logging.Logger
    ) -> list[str]:
        """Execute the failure injection.

        Args:
            deployment: The managed deployment to inject the failure into
            logger: Logger instance for logging failure injection

        Returns: List of affected pod names
        """
        pass

    @abstractmethod
    def get_failure_key(self) -> str:
        """Get the failure key for the failure."""
        pass


@dataclass
class RollingUpgradeFailure(Failure):
    """Failure type for triggering rolling upgrades."""

    async def execute(
        self, deployment: ManagedDeployment, logger: logging.Logger
    ) -> list[str]:
        """Execute rolling upgrade failure injection."""
        await deployment.trigger_rolling_upgrade(self.service_names)

        # Need to wait for the deployment to be unready so we know the rolling upgrade has started
        await deployment.wait_for_unready(timeout=60, log_interval=10)

        await deployment._wait_for_ready(timeout=1800)  # 30 minute timeout

        await asyncio.sleep(
            self.time
        )  # have some requests processed after the rolling upgrade has completed

        return await deployment.get_pod_names(self.service_names)

    def get_failure_key(self) -> str:
        """Get the failure key for the rolling upgrade failure."""
        return f"rolling_upgrade:{','.join(self.service_names)}"


@dataclass
class DeletePodFailure(Failure):
    """Failure type for deleting pods."""

    async def execute(
        self, deployment: ManagedDeployment, logger: logging.Logger
    ) -> list[str]:
        """Execute pod deletion failure injection."""
        service_pod_dict = deployment.get_pods(self.service_names)
        pod_names: list[str] = []
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                deployment.get_pod_manifest_logs_metrics(
                    service_name, pod, ".before_delete"
                )
                pod.delete(force=True)  # force means no graceful termination
                pod_names.append(pod.name)

        return pod_names

    def get_failure_key(self) -> str:
        """Get the failure key for the delete pod failure."""
        return f"delete_pod:{','.join(self.service_names)}"


class TerminateProcessFailure(Failure):
    """Failure type for terminating specific processes by name."""

    def __init__(
        self,
        time: int,
        service_names: list[str],
        signal: str = "SIGINT",
        process_name: str = "",
    ):
        """Initialize TerminateProcessFailure.

        Args:
            time: Time to wait in seconds before the failure is injected
            service_names: Names of DGD services to inject the failure into
            signal: Signal to send (default: "SIGINT")
            process_name: Name of the process to terminate (required)
            end_condition: End condition for failure (e.g., "dgd_ready")
        """
        super().__init__(
            time=time,
            service_names=service_names,
        )
        if not process_name or not signal:
            raise ValueError(
                "process_name and signal are required for TerminateProcessFailure"
            )
        self.process_name = process_name
        self.signal = signal

    async def execute(
        self, deployment: ManagedDeployment, logger: logging.Logger
    ) -> list[str]:
        """Execute process termination failure injection."""
        service_pod_dict = deployment.get_pods(self.service_names)
        pod_names: list[str] = []
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                processes = deployment.get_processes(pod)
                for process in processes:
                    if self.process_name in process.command:
                        logger.info(
                            f"Terminating {service_name} pod {pod} Pid {process.pid} Command {process.command}"
                        )
                        process.kill(self.signal)
                pod_names.append(pod.name)

        return pod_names

    def get_failure_key(self) -> str:
        """Get the failure key for the terminate process failure."""
        return f"terminate_process:{','.join(self.service_names)}:{self.process_name}:{self.signal}"


@dataclass
class TokenOverflowFailure(Failure):
    """
    Failure type for injecting token overflow (prompt > max_seq_len)
    """

    overflow_multiplier: float = 2.0  # How much to exceed max_seq_len (e.g., 2.0 = 2x)
    max_seq_len: int = 1024

    def __init__(
        self,
        time: int,
        max_seq_len: int = 1024,
        overflow_multiplier: float = 2.0,
    ):
        super().__init__(
            time=time,
            service_names=["Client"],
        )
        self.max_seq_len = max_seq_len
        self.overflow_multiplier = overflow_multiplier
        self.overflow_token_count = int(max_seq_len * overflow_multiplier)

    async def execute(
        self, deployment: ManagedDeployment, logger: logging.Logger
    ) -> list[str]:
        """Token overflow is handled client-side, so this is a no-op."""
        # The actual overflow is handled by the client configuration
        # which uses the input_token_length from the Load config
        # This is just a placeholder for the abstract method
        return []

    def get_failure_key(self) -> str:
        """Get the failure key for the token overflow failure."""
        return f"token_overflow:{self.overflow_token_count}"


@dataclass
class Scenario:
    deployment: DeploymentSpec
    load: Load
    failures: list[Failure]
    model: Optional[str] = None
    backend: str = "vllm"  # Backend type for tracking
    # When set to True, the test will be automatically marked with @pytest.mark.custom_build
    # and excluded from default test runs unless --include-custom-build flag is used
    requires_custom_build: bool = False  # Flag for tests needing custom builds/setup
    # List of checkers to run for validation (scenario + results checkers)
    # If None, factory will determine checkers based on scenario name and deployment
    checkers: Optional[List["BaseChecker"]] = field(default=None)


# Helper functions to create deployment specs
def _create_deployment_info(backend: str, yaml_path: str) -> DeploymentInfo:
    """Create a deployment spec with backend information.

    Args:
        backend: Backend type ("vllm", "sglang", or "trtllm")
        yaml_path: Path to the deployment YAML file

    Returns:
        DeploymentInfo dictionary with spec and backend
    """
    return DeploymentInfo(spec=DeploymentSpec(yaml_path), backend=backend)


def _set_replicas(deployment_spec, backend, deploy_type, replicas):
    """Set replicas for all components in a deployment based on backend type."""
    spec = deployment_spec["spec"]

    # Frontend is common for all backends
    spec["Frontend"].replicas = replicas

    if backend in WORKER_MAP:
        # For trtllm agg deployments, use different worker name
        if backend == "trtllm" and deploy_type == "agg":
            decode_worker = WORKER_MAP[backend]["decode_agg"]
        else:
            decode_worker = WORKER_MAP[backend]["decode"]

        # always scale decode
        spec[decode_worker].replicas = replicas
        # scale prefill only for disagg
        if deploy_type == "disagg":
            spec[WORKER_MAP[backend]["prefill"]].replicas = replicas


def _set_tensor_parallel(
    deployment_spec: DeploymentInfo, backend: str, deploy_type: str, tp_size: int
):
    """Set tensor parallel size for worker components."""
    spec = deployment_spec["spec"]

    if backend in WORKER_MAP:
        # For trtllm agg deployments, use different worker name
        if backend == "trtllm" and deploy_type == "agg":
            decode_worker = WORKER_MAP[backend]["decode_agg"]
        else:
            decode_worker = WORKER_MAP[backend]["decode"]
        prefill_worker = WORKER_MAP[backend]["prefill"]

        if deploy_type == "agg":
            if hasattr(spec, "set_tensor_parallel"):
                spec.set_tensor_parallel(tp_size, [decode_worker])
            else:
                spec[decode_worker].tensor_parallel_size = tp_size
        elif deploy_type == "disagg":
            spec[prefill_worker].tensor_parallel_size = tp_size
            spec[decode_worker].tensor_parallel_size = tp_size


def _create_deployments_for_backend(backend: str) -> Dict[str, DeploymentInfo]:
    """Create all deployment specifications for a given backend.

    Args:
        backend: Backend type ("vllm", "sglang", or "trtllm")

    Returns:
        Dictionary mapping deployment names to DeploymentInfo objects
    """
    deployments: Dict[str, DeploymentInfo] = {}

    # Define the yaml files for agg and disagg deployments
    yaml_files = {
        "agg": f"examples/backends/{backend}/deploy/agg.yaml",
        "disagg": f"examples/backends/{backend}/deploy/disagg.yaml",
    }

    # Define the different configurations to test
    configurations = [
        {"tp": 1, "dp": 1},
        {"tp": 1, "dp": 2},
        {"tp": 2, "dp": 1},
        {"tp": 4, "dp": 1},
    ]

    for deploy_type in ["agg", "disagg"]:
        for config in configurations:
            tp_size = config["tp"]
            dp_replicas = config["dp"]
            # Skip creating disagg scenarios for TP > 1 if DP is also > 1 (uncommon case)
            if deploy_type == "disagg" and tp_size > 1 and dp_replicas > 1:
                continue

            # Construct the scenario name
            name_parts = [backend, deploy_type]

            if deploy_type == "agg":
                name_parts.append(f"tp-{tp_size}")
            elif deploy_type == "disagg":
                name_parts.append(f"prefill-tp-{tp_size}-decode-tp-{tp_size}")

            name_parts.append(f"dp-{dp_replicas}")

            scenario_name = "-".join(name_parts)

            # Create and configure the deployment
            deployment = _create_deployment_info(backend, yaml_files[deploy_type])
            if tp_size > 1:
                _set_tensor_parallel(deployment, backend, deploy_type, tp_size)
            if dp_replicas > 1:
                _set_replicas(deployment, backend, deploy_type, dp_replicas)

            deployments[scenario_name] = deployment

    return deployments


def _create_moe_deployments_for_backend(
    backend: str = "vllm",
) -> Dict[str, DeploymentInfo]:
    """Create MoE-specific deployment configurations for DeepSeek-V2-Lite.

    Args:
        backend: Backend type (default: "vllm")

    Returns:
        Dictionary mapping deployment names to DeploymentInfo objects
    """
    deployments: Dict[str, DeploymentInfo] = {}

    # Only test tp=1, dp=2 for now
    tp_size = 1
    dp_replicas = (
        2  # Note: this is handled internally by vLLM with --data-parallel-size
    )

    template_dir = "tests/fault_tolerance/deploy/templates"
    yaml_files = {
        "agg": f"{template_dir}/{backend}/moe_agg.yaml",
        "disagg": f"{template_dir}/{backend}/moe_disagg.yaml",
    }

    for deploy_type in ["agg", "disagg"]:
        scenario_name = f"{backend}-moe-{deploy_type}-tp-{tp_size}-dp-{dp_replicas}"
        deployment = DeploymentInfo(
            spec=DeploymentSpec(yaml_files[deploy_type]),
            backend=backend,
            model="deepseek-ai/DeepSeek-V2-Lite",
            is_moe=True,
        )

        deployments[scenario_name] = deployment

    return deployments


# Create all deployment specifications
DEPLOYMENT_SPECS: Dict[str, DeploymentInfo] = {}
DEPLOYMENT_SPECS.update(_create_deployments_for_backend("vllm"))
DEPLOYMENT_SPECS.update(_create_deployments_for_backend("sglang"))
DEPLOYMENT_SPECS.update(_create_deployments_for_backend("trtllm"))

# Add MoE deployments for vLLM only
DEPLOYMENT_SPECS.update(_create_moe_deployments_for_backend("vllm"))


# Each failure scenaro contains a list of failure injections
# Each failure injection has a time in seconds after the pervious injection and
# a list of failures to inject including the number of failures for each type.
# Failures are currently process termination or pod deletion
#
# Example:
#
#   "prefill_worker": [Failure(30, "VllmPrefillWorker", "dynamo.vllm", "SIGKILL")],
#
# terminates 1 prefill worker after 30 seconds
def _create_backend_failures(backend, deploy_type="disagg"):
    """Generate backend-specific failure scenarios.

    Args:
        backend: Backend type (vllm, sglang, trtllm)
        deploy_type: Deployment type (agg or disagg)
    """
    workers = WORKER_MAP[backend]

    # Use correct worker name based on deployment type
    if backend == "trtllm" and deploy_type == "agg":
        decode_worker = workers["decode_agg"]
    else:
        decode_worker = workers["decode"]

    prefill_worker = workers["prefill"]
    process_name = f"dynamo.{backend}"

    failures = {
        "frontend": [
            TerminateProcessFailure(
                30, ["Frontend"], "SIGINT", process_name="dynamo.frontend"
            )
        ],
        "frontend_pod": [DeletePodFailure(30, ["Frontend"])],
        "decode_worker": [
            TerminateProcessFailure(
                30, [decode_worker], "SIGKILL", process_name=process_name
            )
        ],
        "decode_worker_pod": [DeletePodFailure(30, [decode_worker])],
        "prefill_worker": [
            TerminateProcessFailure(
                30, [prefill_worker], "SIGKILL", process_name=process_name
            )
        ],
        "prefill_worker_pod": [DeletePodFailure(30, [prefill_worker])],
        "none": [],
    }

    if backend == "vllm":
        failures["vllm_decode_engine_core"] = [
            TerminateProcessFailure(
                30, [decode_worker], "SIGKILL", process_name="VLLM::EngineCore"
            )
        ]
        failures["vllm_prefill_engine_core"] = [
            TerminateProcessFailure(
                30, [prefill_worker], "SIGKILL", process_name="VLLM::EngineCore"
            )
        ]
    elif backend == "sglang":
        failures["sglang_decode_scheduler"] = [
            TerminateProcessFailure(
                30, [decode_worker], "SIGKILL", process_name="sglang::scheduler"
            )
        ]
        failures["sglang_decode_detokenizer"] = [
            TerminateProcessFailure(
                30, [decode_worker], "SIGKILL", process_name="sglang::detokenizer"
            )
        ]
        failures["sglang_prefill_scheduler"] = [
            TerminateProcessFailure(
                30, [prefill_worker], "SIGKILL", process_name="sglang::scheduler"
            )
        ]
        failures["sglang_prefill_detokenizer"] = [
            TerminateProcessFailure(
                30, [prefill_worker], "SIGKILL", process_name="sglang::detokenizer"
            )
        ]
    elif backend == "trtllm":
        failures["trtllm_decode_engine_core"] = [
            TerminateProcessFailure(
                30, [decode_worker], "SIGKILL", process_name="TRTLLM::EngineCore"
            )
        ]
        failures["trtllm_prefill_engine_core"] = [
            TerminateProcessFailure(
                30, [prefill_worker], "SIGKILL", process_name="TRTLLM::EngineCore"
            )
        ]

    return failures


def create_aiperf_load(
    clients: int = 10,
    requests_per_client: int = 150,
    input_token_length: int = 100,
    output_token_length: int = 100,
    max_retries: int = 3,
    sla: Optional[float] = None,
    max_request_rate: float = 1.0,
    success_threshold: float = 90.0,
) -> Load:
    """Create a Load configuration for AI-Perf client.

    Args:
        clients: Number of concurrent clients (default: 10)
        requests_per_client: Number of requests per client (default: 150)
        input_token_length: Input token count (default: 100)
        output_token_length: Output token count (default: 100)
        max_retries: Maximum retry attempts - AI-Perf retries entire test (default: 3)
        sla: Optional SLA threshold for latency (default: None)
        max_request_rate: Rate limiting for requests/sec (default: 1.0)
        success_threshold: Success rate threshold for pass/fail (default: 90.0)

    Returns:
        Load instance configured for AI-Perf client

    Example:
        >>> load = create_aiperf_load(clients=20, requests_per_client=200)
    """
    return Load(
        clients=clients,
        requests_per_client=requests_per_client,
        input_token_length=input_token_length,
        output_token_length=output_token_length,
        max_retries=max_retries,
        sla=sla,
        client_type="aiperf",
        max_request_rate=max_request_rate,
        success_threshold=success_threshold,
    )


def create_legacy_load(
    clients: int = 10,
    requests_per_client: int = 100,
    input_token_length: int = 100,
    output_token_length: int = 100,
    max_retries: int = 1,
    sla: Optional[float] = None,
    max_request_rate: float = 1.0,
    success_threshold: float = 90.0,
) -> Load:
    """Create a Load configuration for legacy custom client.

    Args:
        clients: Number of concurrent clients (default: 10)
        requests_per_client: Number of requests per client (default: 100, fewer than AI-Perf)
        input_token_length: Input token count (default: 100)
        output_token_length: Output token count (default: 100)
        max_retries: Maximum retry attempts - legacy retries per request (default: 1)
        sla: Optional SLA threshold for latency (default: None)
        max_request_rate: Rate limiting for requests/sec (default: 1.0)
        success_threshold: Success rate threshold for pass/fail (default: 90.0)

    Returns:
        Load instance configured for legacy client

    Example:
        >>> load = create_legacy_load(clients=10, max_request_rate=2.0)
    """
    return Load(
        clients=clients,
        requests_per_client=requests_per_client,
        input_token_length=input_token_length,
        output_token_length=output_token_length,
        max_retries=max_retries,
        sla=sla,
        client_type="legacy",
        max_request_rate=max_request_rate,
        success_threshold=success_threshold,
    )


# Default load configuration (using AI-Perf)
load = Load()

# MoE-specific load configuration
moe_load = Load(
    clients=3,  # Fewer clients for MoE testing
    requests_per_client=30,  # Reduced for MoE complexity
    input_token_length=100,
    output_token_length=100,
    max_retries=3,
    sla=None,
    client_type="aiperf",
    max_request_rate=0.5,  # Lower rate for MoE
)

# model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

model = None

# Populate Scenarios

scenarios: dict[str, Scenario] = {}

# Map of backend+deploy_type to failure definitions
backend_failure_map = {}
for backend in ["vllm", "sglang", "trtllm"]:
    backend_failure_map[f"{backend}_agg"] = _create_backend_failures(backend, "agg")
    backend_failure_map[f"{backend}_disagg"] = _create_backend_failures(
        backend, "disagg"
    )

for deployment_name, deployment_info in DEPLOYMENT_SPECS.items():
    backend = deployment_info["backend"]

    # Check if this is an MoE deployment
    is_moe = deployment_info.get("is_moe", False)

    # Determine deployment type from deployment name
    deploy_type = (
        "agg"
        if ("agg" in deployment_name and "disagg" not in deployment_name)
        else "disagg"
    )

    # Get the appropriate failure set for this backend+deploy_type
    failure_map_key = f"{backend}_{deploy_type}"
    if failure_map_key not in backend_failure_map:
        raise ValueError(
            f"Unsupported backend+deploy_type: {failure_map_key}. Available: {list(backend_failure_map.keys())}"
        )

    failure_set = backend_failure_map[failure_map_key]

    for failure_name, failure in failure_set.items():
        # Skip prefill failures for aggregated deployments
        if "prefill" in failure_name and deploy_type == "agg":
            continue

        scenario_name = f"{deployment_name}-{failure_name}"

        # Use MoE-specific load configuration if it's an MoE model
        load_config = moe_load if is_moe else load

        # Get model from deployment info or use the global model
        scenario_model = deployment_info.get("model", model)

        # Create scenario first (without checkers)
        scenario = Scenario(
            deployment=deployment_info["spec"],
            load=load_config,
            failures=failure,
            model=scenario_model,
            backend=backend,
            checkers=None,  # Will be populated below
            requires_custom_build=is_moe,  # MoE models require custom builds
        )

        # Generate checkers for this scenario
        # This uses the checker factory to determine appropriate validation checks
        scenario.checkers = _get_checkers_for_scenario(scenario_name, scenario)

        scenarios[scenario_name] = scenario


# Add token overflow test scenarios
def add_token_overflow_scenarios():
    """
    Add test scenarios for token overflow (prompt > max_seq_len) failures
    """
    overflow_test_configs = [
        # vLLM tests
        {
            "name": "vllm_agg_token_overflow_2x",
            "deployment_key": "vllm-agg-tp-1-dp-1",
            "backend": "vllm",
        },
        {
            "name": "vllm_disagg_token_overflow_2x",
            "deployment_key": "vllm-disagg-prefill-tp-2-decode-tp-2-dp-1",
            "backend": "vllm",
        },
        # TRT-LLM tests
        {
            "name": "trtllm_agg_token_overflow_2x",
            "deployment_key": "trtllm-agg-tp-1-dp-1",
            "backend": "trtllm",
        },
        {
            "name": "trtllm_disagg_token_overflow_2x",
            "deployment_key": "trtllm-disagg-prefill-tp-2-decode-tp-2-dp-1",
            "backend": "trtllm",
        },
        # SGLang tests
        {
            "name": "sglang_agg_token_overflow_2x",
            "deployment_key": "sglang-agg-tp-1-dp-1",
            "backend": "sglang",
        },
        {
            "name": "sglang_disagg_token_overflow_2x",
            "deployment_key": "sglang-disagg-prefill-tp-2-decode-tp-2-dp-1",
            "backend": "sglang",
        },
    ]

    # Common configuration for all tests
    MAX_SEQ_LEN = 1024
    OVERFLOW_MULTIPLIER = 2.0
    OVERFLOW_REQUESTS = 15  # Number of oversized requests to send
    NORMAL_REQUESTS = 15  # Number of normal requests to send after overflow

    for config in overflow_test_configs:
        # Skip if deployment doesn't exist
        if config["deployment_key"] not in DEPLOYMENT_SPECS:
            continue

        overflow_scenario_name = config["name"]
        deployment_info = DEPLOYMENT_SPECS[config["deployment_key"]]

        scenario_model = deployment_info.get("model", model)

        deployment_spec = deployment_info["spec"]

        backend = config["backend"]
        is_agg = (
            "disagg" not in config["deployment_key"]
        )  # If not disaggregated, then it's aggregated

        workers = WORKER_MAP[backend]

        # Get the correct decode worker name
        if backend == "trtllm" and is_agg:
            decode_worker = workers["decode_agg"]
        else:
            decode_worker = workers["decode"]

        prefill_worker = workers["prefill"]

        # Determine argument name based on backend
        if backend == "trtllm":
            arg_name = "--max-seq-len"
        elif backend == "sglang":
            arg_name = "--context-length"
        else:  # vllm
            arg_name = "--max-model-len"

        # Add arguments to appropriate workers
        if is_agg:
            # For aggregated, add only to decode worker
            deployment_spec.add_arg_to_service(
                decode_worker, arg_name, str(MAX_SEQ_LEN)
            )
        else:
            # For disaggregated, add to both prefill and decode workers
            deployment_spec.add_arg_to_service(
                prefill_worker, arg_name, str(MAX_SEQ_LEN)
            )
            deployment_spec.add_arg_to_service(
                decode_worker, arg_name, str(MAX_SEQ_LEN)
            )

        # Create overflow failure
        overflow_failure = TokenOverflowFailure(
            time=30,  # Start after 30 seconds
            max_seq_len=MAX_SEQ_LEN,
            overflow_multiplier=OVERFLOW_MULTIPLIER,
        )

        # Create mixed load configuration for overflow + recovery testing
        overflow_tokens = int(MAX_SEQ_LEN * OVERFLOW_MULTIPLIER)
        normal_tokens = 512  # Well within MAX_SEQ_LEN

        # Total requests = overflow + normal
        total_requests = OVERFLOW_REQUESTS + NORMAL_REQUESTS

        # Mixed load that tests both rejection and recovery
        mixed_load = Load(
            clients=3,
            requests_per_client=total_requests,
            input_token_length=normal_tokens,
            output_token_length=50,
            # Mixed token test configuration
            mixed_token_test=True,
            overflow_token_length=overflow_tokens,
            overflow_request_count=OVERFLOW_REQUESTS,
            normal_request_count=NORMAL_REQUESTS,
        )

        scenarios[overflow_scenario_name] = Scenario(
            deployment=deployment_spec,
            load=mixed_load,
            failures=[overflow_failure],
            model=scenario_model,
            backend=backend,
        )


def add_rolling_upgrade_scenarios():
    for backend in ["vllm", "sglang", "trtllm"]:
        for worker_mode in ["agg", "disagg"]:
            yaml_files = {
                "agg": f"examples/backends/{backend}/deploy/agg.yaml",
                "disagg": f"examples/backends/{backend}/deploy/disagg.yaml",
            }
            deployment_info = _create_deployment_info(backend, yaml_files[worker_mode])
            deployment_spec: DeploymentSpec = deployment_info["spec"]

            service_names: list[str] = []

            # setting replicas to 2 so we have availability of 1 replica at a time
            if worker_mode == "agg" and backend == "trtllm":
                service_names.append(WORKER_MAP[backend]["decode_agg"])
            else:
                service_names.append(WORKER_MAP[backend]["decode"])

            if worker_mode == "disagg":
                service_names.append(WORKER_MAP[backend]["prefill"])

            for service_name in service_names:
                deployment_spec.set_service_replicas(service_name, 2)

            load = Load(
                clients=10,
                input_token_length=100,
                output_token_length=100,
                max_retries=1,
                client_type="aiperf",
                max_request_rate=1.0,
                success_threshold=100.0,
                continuous_load=True,
            )

            scenario_name = f"{backend}-{worker_mode}-rolling-upgrade"
            model = "Qwen/Qwen3-0.6B"

            failure = RollingUpgradeFailure(
                time=30,
                service_names=service_names,
            )
            scenarios[scenario_name] = Scenario(
                deployment=deployment_info["spec"],
                load=load,
                failures=[failure],
                model=model,
                backend=backend,
            )


# Add the token overflow scenarios
add_token_overflow_scenarios()

# Add the rolling upgrade scenarios
add_rolling_upgrade_scenarios()
