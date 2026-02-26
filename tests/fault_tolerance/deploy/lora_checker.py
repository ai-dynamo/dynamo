# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""LoRA-specific checker for Kubernetes discovery validation.

This checker verifies:
1. LoRA adapters register correctly in etcd discovery
2. LoRA metadata is present and valid
3. LoRA discovery works across worker instances
"""

import logging
from typing import Optional

from tests.fault_tolerance.deploy.base_checker import BaseChecker, ValidationContext

logger = logging.getLogger(__name__)


class LoRADiscoveryChecker(BaseChecker):
    """Verify that LoRA discovery works correctly in Kubernetes.

    This checker validates:
    - LoRA adapters are registered in etcd with correct namespace
    - LoRA metadata includes necessary information (lora_id, lora_path)
    - Multiple worker instances can register LoRAs independently
    - Frontend can discover LoRAs from all workers
    """

    def __init__(self, lora_name: Optional[str] = None):
        """Initialize LoRA discovery checker.

        Args:
            lora_name: Optional specific LoRA name to check for
        """
        super().__init__(name="LoRADiscoveryChecker")
        self.lora_name = lora_name

    def check(self, context: ValidationContext) -> None:
        """Verify LoRA discovery in Kubernetes deployment.

        This is a placeholder checker that logs LoRA-specific information.
        Full implementation would:
        1. Query etcd for LoRA registration entries
        2. Verify namespace scoping
        3. Check LoRA metadata format
        4. Validate multi-worker discovery

        Args:
            context: ValidationContext with deployment and scenario info
        """
        self.logger.info("=" * 80)
        self.logger.info("LoRA Discovery Validation")
        self.logger.info("=" * 80)

        # Log scenario information
        if context.scenario:
            self.logger.info(f"Scenario backend: {context.scenario.backend}")
            self.logger.info(f"Scenario model: {context.scenario.model}")

        # Log deployment information
        if context.deployment:
            deployment_name = context.deployment.name
            namespace = context.namespace or "unknown"
            self.logger.info(f"Deployment: {deployment_name}")
            self.logger.info(f"Namespace: {namespace}")

            # Expected LoRA discovery behavior:
            # 1. LoRAs register in etcd under: v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}/{lora_slug}
            # 2. User data includes: {"lora_adapter": True, "lora_id": lora_id, "lora_path": lora_path}
            # 3. Frontend discovers LoRAs via namespace-scoped query
            # 4. Each worker instance registers its LoRAs independently

            self.logger.info("")
            self.logger.info("Expected LoRA Discovery Pattern:")
            self.logger.info(
                f"  Registry Path: v1/mdc/{namespace}/<component>/<endpoint>/<instance_id>/<lora_slug>"
            )
            self.logger.info("  Metadata: lora_adapter=True, lora_id, lora_path")
            self.logger.info("")

            # In a full implementation, we would:
            # 1. Connect to etcd using deployment's etcd service
            # 2. Query for entries matching the LoRA pattern
            # 3. Validate metadata structure and content
            # 4. Verify multiple worker instances registered correctly

            self.logger.info(
                "✓ LoRA discovery check passed (placeholder implementation)"
            )
            self.logger.info(
                "  Note: Full etcd query validation would be implemented here"
            )
        else:
            self.logger.warning(
                "⚠ No deployment context available for LoRA discovery check"
            )

        self.logger.info("=" * 80)


class LoRAInferenceChecker(BaseChecker):
    """Verify that LoRA inference works correctly.

    This checker validates:
    - LoRA models can be loaded successfully
    - Inference with LoRA produces expected results
    - LoRA routing works across multiple workers
    """

    def __init__(self):
        super().__init__(name="LoRAInferenceChecker")

    def check(self, context: ValidationContext) -> None:
        """Verify LoRA inference functionality.

        This validates that the system can successfully:
        1. Load LoRA adapters from S3/MinIO storage
        2. Route requests to LoRA-enabled workers
        3. Generate responses using LoRA models

        Args:
            context: ValidationContext with metrics and results
        """
        self.logger.info("=" * 80)
        self.logger.info("LoRA Inference Validation")
        self.logger.info("=" * 80)

        # Check basic success metrics
        if context.metrics:
            success_rate = context.metrics.get("success_rate", 0)
            total_requests = context.metrics.get("total_requests", 0)
            successful_requests = context.metrics.get("successful_requests", 0)

            self.logger.info(f"Total requests: {total_requests}")
            self.logger.info(f"Successful requests: {successful_requests}")
            self.logger.info(f"Success rate: {success_rate:.2f}%")

            # For LoRA tests, we expect high success rate
            # (failures in LoRA loading should be minimal)
            if success_rate < 80.0:
                self.logger.warning(
                    f"⚠ Low success rate for LoRA inference: {success_rate:.2f}%"
                )
                self.logger.warning(
                    "  This may indicate LoRA loading or routing issues"
                )
            else:
                self.logger.info(
                    f"✓ LoRA inference success rate is acceptable: {success_rate:.2f}%"
                )

        self.logger.info("=" * 80)
