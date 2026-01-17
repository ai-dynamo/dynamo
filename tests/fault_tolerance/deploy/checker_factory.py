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

"""Factory functions for generating checker lists for scenarios.

This module provides factory functions that determine which checkers
to run based on:
1. Explicit checkers in scenario (highest priority)
2. Pattern matching on scenario name
3. Deployment configuration (redundancy level)
"""

import logging
from typing import List, Optional

from tests.fault_tolerance.deploy.base_checker import BaseChecker
from tests.fault_tolerance.deploy.checkers import (
    BaselineResultsChecker,
    HighAvailabilityResultsChecker,
    NoFailureChecker,
    PodDeletionChecker,
    ProcessTerminationChecker,
    SingleWorkerResultsChecker,
)
from tests.fault_tolerance.deploy.scenarios import Scenario

logger = logging.getLogger(__name__)


def get_checkers_for_scenario(test_name: str, scenario: Scenario) -> List[BaseChecker]:
    """Get appropriate list of checkers for a scenario.

    This factory function determines which checkers to use based on:
    1. Explicit checkers in scenario object (highest priority)
    2. Pattern matching on test name
    3. Deployment redundancy (DP > 1)

    Args:
        test_name: Full test name (e.g., "test_fault_scenario[vllm-agg-tp-1-dp-1-decode_worker_pod]")
        scenario: Scenario object

    Returns:
        List of BaseChecker instances to run
    """
    # 1. Explicit checkers take priority
    if scenario.checkers is not None:
        logger.info(f"Using explicit checkers for {test_name}: {scenario.checkers}")
        return scenario.checkers

    # 2. Pattern-based checker selection
    logger.info(f"Using pattern-based checker selection for {test_name}")

    checkers: List[BaseChecker] = []

    # Stage 1: Scenario verification
    scenario_checker = get_scenario_checker(test_name, scenario)
    if scenario_checker:
        checkers.append(scenario_checker)

    # Stage 2: Results verification
    results_checker = get_results_checker(test_name, scenario)
    if results_checker:
        checkers.append(results_checker)

    logger.info(f"Selected checkers: {[c.name for c in checkers]}")
    return checkers


def get_scenario_checker(test_name: str, scenario: Scenario) -> Optional[BaseChecker]:
    """Get appropriate scenario checker (Stage 1).

    Args:
        test_name: Full test name
        scenario: Scenario object

    Returns:
        Scenario checker instance or None
    """
    # No failures scenario
    if test_name.endswith("-none]"):
        return NoFailureChecker()

    # Pod deletion scenarios
    if "_pod]" in test_name:
        return PodDeletionChecker()

    # Process termination scenarios (not pod deletions)
    if any(
        x in test_name
        for x in [
            "decode_worker]",
            "prefill_worker]",
            "frontend]",
            "scheduler]",
            "detokenizer]",
            "engine_core]",
        ]
    ):
        return ProcessTerminationChecker()

    # Default: no specific scenario checker
    logger.info(f"No specific scenario checker for {test_name}")
    return None


def get_results_checker(test_name: str, scenario: Scenario) -> BaseChecker:
    """Get appropriate results checker (Stage 2).

    Determines checker based on deployment redundancy (DP).

    Args:
        test_name: Full test name
        scenario: Scenario object

    Returns:
        Results checker instance
    """
    # No failures baseline
    if test_name.endswith("-none]"):
        return BaselineResultsChecker()

    # Determine if deployment has redundancy (DP > 1)
    has_redundancy = False

    # Determine worker service name based on backend and deployment type
    if scenario.backend == "vllm":
        worker_service_name = "VllmDecodeWorker"
    elif scenario.backend == "sglang":
        worker_service_name = "decode"
    elif scenario.backend == "trtllm":
        # TensorRT-LLM uses different names for agg vs disagg
        # Check test name to determine deployment type
        if "disagg" in test_name:
            worker_service_name = "TRTLLMDecodeWorker"
        else:
            # Agg deployment uses TRTLLMWorker
            worker_service_name = "TRTLLMWorker"
    else:
        logger.warning(
            f"Unsupported backend: {scenario.backend}, using default checker"
        )
        return SingleWorkerResultsChecker()

    try:
        worker_spec = scenario.deployment[worker_service_name]
        if worker_spec and hasattr(worker_spec, "replicas"):
            has_redundancy = worker_spec.replicas > 1
    except (KeyError, AttributeError) as e:
        logger.warning(f"Could not determine redundancy: {e}")

    # Select appropriate results checker
    if has_redundancy:
        logger.info("Using HighAvailabilityResultsChecker (DP > 1)")
        return HighAvailabilityResultsChecker()
    else:
        logger.info("Using SingleWorkerResultsChecker (DP = 1)")
        return SingleWorkerResultsChecker()
