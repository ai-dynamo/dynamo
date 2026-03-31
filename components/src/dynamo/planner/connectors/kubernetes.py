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

import json
import logging
import os
from typing import Optional

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.kubernetes_api import KubernetesAPI
from dynamo.planner.errors import (
    DeploymentModelNameMismatchError,
    DeploymentValidationError,
    EmptyTargetReplicasError,
    ModelNameNotFoundError,
    PlannerError,
    UserProvidedModelNameMismatchError,
)
from dynamo.planner.monitoring.dgd_services import (
    get_service_from_sub_component_type_or_name,
)
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class KubernetesConnector(PlannerConnector):
    def __init__(
        self,
        dynamo_namespace: str,
        model_name: Optional[str] = None,
        k8s_namespace: Optional[str] = None,
        parent_dgd_name: Optional[str] = None,
    ):
        self.kube_api = KubernetesAPI(k8s_namespace)

        self.user_provided_model_name: Optional[str] = None
        if model_name:
            self.user_provided_model_name = (
                model_name.lower()
            )  # normalize model name to lowercase (MDC)

        # Allow overriding parent DGD name for centralized planner
        if parent_dgd_name:
            self.parent_dgd_name = parent_dgd_name
        else:
            graph_deployment_name = os.getenv("DYN_PARENT_DGD_K8S_NAME")
            if not graph_deployment_name:
                raise DeploymentValidationError(
                    ["DYN_PARENT_DGD_K8S_NAME environment variable is not set"]
                )
            self.parent_dgd_name = graph_deployment_name

        # For backwards compatibility
        self.graph_deployment_name = self.parent_dgd_name

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Add a component by increasing its replica count by 1"""

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        service = get_service_from_sub_component_type_or_name(
            deployment, sub_component_type
        )
        self.kube_api.update_graph_replicas(
            self.graph_deployment_name,
            service.name,
            service.number_replicas() + 1,
        )
        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self.graph_deployment_name,
            )

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Remove a component by decreasing its replica count by 1"""

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        service = get_service_from_sub_component_type_or_name(
            deployment, sub_component_type
        )
        if service.number_replicas() > 0:
            self.kube_api.update_graph_replicas(
                self.graph_deployment_name,
                service.name,
                service.number_replicas() - 1,
            )
            if blocking:
                await self.kube_api.wait_for_graph_deployment_ready(
                    self.graph_deployment_name,
                )

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ):
        """
        Verify that the deployment contains services with subComponentType prefill and decode and the model name exists.
        Will fallback to worker service names for backwards compatibility. (TODO: deprecate)

        Raises:
            DynamoGraphDeploymentNotFoundError: If the deployment is not found
            DeploymentValidationError: If the deployment does not contain services with subComponentType prefill and decode
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        errors = []

        if require_prefill:
            try:
                get_service_from_sub_component_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                    component_name=prefill_component_name,
                )
            except PlannerError as e:
                errors.append(str(e))

        if require_decode:
            try:
                get_service_from_sub_component_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                    component_name=decode_component_name,
                )
            except PlannerError as e:
                errors.append(str(e))

        try:
            self.get_model_name(
                deployment,
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
        except PlannerError as e:
            errors.append(str(e))

        # Raise combined error if any issues found
        if errors:
            raise DeploymentValidationError(errors)

    def get_model_name(
        self,
        deployment: Optional[dict] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> str:
        """Get the model name from the deployment"""
        try:
            if deployment is None:
                deployment = self.kube_api.get_graph_deployment(
                    self.graph_deployment_name
                )

            # TODO: dynamo/profiler/utils/config.py already contains DGD config parsing
            # and model name logic, should consolidate
            prefill_model_name = None
            decode_model_name = None
            if require_prefill:
                prefill_service = get_service_from_sub_component_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                )
                prefill_model_name = prefill_service.get_model_name()
            if require_decode:
                decode_service = get_service_from_sub_component_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                )
                decode_model_name = decode_service.get_model_name()

            if prefill_model_name is None and decode_model_name is None:
                raise ModelNameNotFoundError()

            # Check model name between prefill and decode
            if prefill_model_name is None:
                model_name = decode_model_name
            elif decode_model_name is None:
                model_name = prefill_model_name
            elif prefill_model_name != decode_model_name:
                raise DeploymentModelNameMismatchError(
                    prefill_model_name, decode_model_name
                )
            else:
                model_name = prefill_model_name

        except PlannerError as e:
            if self.user_provided_model_name:
                logger.warning(
                    f"Failed to get model name from deployment with error: {e}, using provided model name: {self.user_provided_model_name}"
                )
                model_name = self.user_provided_model_name
            else:
                raise e

        if not model_name:
            raise ModelNameNotFoundError()

        # If user provided a model name and it doesn't match the model name from the deployment, raise an error
        if self.user_provided_model_name:
            if model_name != self.user_provided_model_name:
                raise UserProvidedModelNameMismatchError(
                    model_name, self.user_provided_model_name
                )

        return model_name

    def get_gpu_counts(
        self,
        deployment: Optional[dict] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[int, int]:
        """Get the GPU counts for prefill and decode services from the deployment.

        Args:
            deployment: Optional deployment dict, fetched if not provided
            require_prefill: Whether to require prefill service
            require_decode: Whether to require decode service

        Returns:
            Tuple of (prefill_gpu_count, decode_gpu_count)

        Raises:
            DeploymentValidationError: If GPU counts cannot be determined from DGD
        """
        if deployment is None:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        prefill_gpu_count = 0
        decode_gpu_count = 0
        errors = []

        if require_prefill:
            try:
                prefill_service = get_service_from_sub_component_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                )
                prefill_gpu_count = prefill_service.get_gpu_count()
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get prefill GPU count: {e}")

        if require_decode:
            try:
                decode_service = get_service_from_sub_component_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                )
                decode_gpu_count = decode_service.get_gpu_count()
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get decode GPU count: {e}")

        if errors:
            raise DeploymentValidationError(errors)

        return prefill_gpu_count, decode_gpu_count

    def get_frontend_metrics_url(self, port: int = 8000) -> Optional[str]:
        """Auto-discover the Frontend service's metrics URL from the DGD.

        Iterates spec.services to find the service with componentType "frontend",
        then constructs the in-cluster URL using the operator's naming convention:
        http://{dgd_name}-{service_key_lowercase}:{port}/metrics

        Returns:
            The metrics URL string, or None if no frontend service is found.
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        services = deployment.get("spec", {}).get("services", {})

        for service_key, service_spec in services.items():
            if service_spec.get("componentType", "") == "frontend":
                service_name = f"{self.graph_deployment_name}-{service_key.lower()}"
                url = f"http://{service_name}:{port}/metrics"
                logger.info(f"Auto-discovered frontend metrics URL: {url}")
                return url

        return None

    async def wait_for_deployment_ready(self, include_planner: bool = True):
        """Wait for the deployment to be ready.

        Args:
            include_planner: If False, skip the planner service when checking
                readiness. This lets the planner read MDC from worker pods
                without waiting for itself to be marked ready in the DGD.
        """
        await self.kube_api.wait_for_graph_deployment_ready(
            self.graph_deployment_name,
            include_planner=include_planner,
        )

    def _list_worker_metadata_crs(self) -> list[dict]:
        """List all DynamoWorkerMetadata CRs in the current namespace.

        Returns an empty list only when the CRD is not yet installed (404).
        Other API errors (RBAC, connectivity) are re-raised so callers can
        handle them explicitly.
        """
        from kubernetes.client import ApiException

        try:
            result = self.kube_api.custom_api.list_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.kube_api.current_namespace,
                plural="dynamoworkermetadatas",
            )
            return result.get("items", [])
        except ApiException as e:
            if e.status == 404:
                logger.info("DynamoWorkerMetadata CRD not found, skipping MDC")
                return []
            raise

    def _extract_mdc_entries(
        self,
    ) -> list[dict]:
        """Extract MDC entries belonging to this DGD.

        CRs are named after the worker pod (e.g. ``<dgd>-0-<service>-<hash>``),
        so we filter by the DGD name prefix to avoid picking up entries from
        other deployments sharing the namespace.

        Returns a list of dicts, each containing:
            namespace, component, endpoint, instance_id, card_json
        """
        crs = self._list_worker_metadata_crs()
        dgd_prefix = f"{self.graph_deployment_name}-"

        entries: list[dict] = []
        for cr in crs:
            cr_name = cr.get("metadata", {}).get("name", "")
            if not cr_name.startswith(dgd_prefix):
                continue

            data = cr.get("spec", {}).get("data", {})
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    continue
            model_cards = data.get("model_cards", {})
            for _key, instance in model_cards.items():
                if instance.get("type") != "Model":
                    continue
                entries.append(instance)
        return entries

    def _mdc_entry_is_prefill(self, entry: dict) -> bool:
        """Check if an MDC entry is a prefill worker.

        model_type can be serialized as:
        - An integer bitflag (ModelType::Prefill = 1 << 4 = 16)
        - A dict with a "bits" key (serde bitflags format)
        - A string like "Prefill" or "Chat|Completions"
        """
        card = entry.get("card_json", {})
        model_type = card.get("model_type", 0)
        if isinstance(model_type, str):
            return "prefill" in model_type.lower()
        if isinstance(model_type, dict):
            model_type = model_type.get("bits", 0)
        return bool(model_type & 0x10)

    def _build_worker_info_from_mdc(
        self,
        entry: dict,
        sub_component_type: SubComponentType,
    ) -> WorkerInfo:
        """Build a WorkerInfo from an MDC entry, applying the fallback chain.

        Priority: MDC -> DGD container-arg parsing -> hard-coded defaults.
        """
        defaults = build_worker_info_from_defaults(
            self._backend_hint or "vllm", sub_component_type
        )

        card = entry.get("card_json", {})
        runtime_cfg = card.get("runtime_config", {})

        # --- component / endpoint names from MDC wrapper ---
        mdc_component = entry.get("component")
        mdc_endpoint = entry.get("endpoint")

        component_name = mdc_component or defaults.component_name
        endpoint = mdc_endpoint or defaults.endpoint
        if not mdc_component:
            logger.info(
                f"MDC missing 'component' for {sub_component_type.value}, "
                f"falling back to default: {defaults.component_name}"
            )
        if not mdc_endpoint:
            logger.info(
                f"MDC missing 'endpoint' for {sub_component_type.value}, "
                f"falling back to default: {defaults.endpoint}"
            )

        # --- model name ---
        mdc_model = card.get("display_name")
        model_name = mdc_model
        if not model_name:
            # Fallback: parse from DGD container args
            try:
                deployment = self.kube_api.get_graph_deployment(
                    self.graph_deployment_name
                )
                service = get_service_from_sub_component_type_or_name(
                    deployment, sub_component_type
                )
                model_name = service.get_model_name()
                if model_name:
                    logger.info(
                        f"MDC missing model name for {sub_component_type.value}, "
                        f"fell back to DGD container args: {model_name}"
                    )
            except PlannerError:
                pass
        if not model_name:
            logger.warning(
                f"Could not determine model name for {sub_component_type.value} "
                f"from MDC or DGD container args"
            )

        # --- runtime config fields ---
        total_kv_blocks = runtime_cfg.get("total_kv_blocks")
        max_num_seqs = runtime_cfg.get("max_num_seqs")
        max_num_batched_tokens = runtime_cfg.get("max_num_batched_tokens")
        kv_cache_block_size = card.get("kv_cache_block_size")
        context_length = card.get("context_length")

        if total_kv_blocks is None:
            logger.info(f"MDC missing total_kv_blocks for {sub_component_type.value}")
        if max_num_seqs is None:
            logger.info(f"MDC missing max_num_seqs for {sub_component_type.value}")

        # --- k8s_name: resolve from DGD subComponentType ---
        k8s_name = defaults.k8s_name
        try:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
            service = get_service_from_sub_component_type_or_name(
                deployment, sub_component_type
            )
            k8s_name = service.name
        except PlannerError:
            logger.info(
                f"Could not resolve k8s service name for {sub_component_type.value}, "
                f"using default: {defaults.k8s_name}"
            )

        info = WorkerInfo(
            k8s_name=k8s_name,
            component_name=component_name,
            endpoint=endpoint,
            model_name=model_name,
            total_kv_blocks=total_kv_blocks,
            kv_cache_block_size=kv_cache_block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            context_length=context_length,
        )
        return info

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        """Get WorkerInfo for a sub-component, trying MDC first, then fallbacks.

        Args:
            sub_component_type: PREFILL or DECODE
            backend: Backend framework name (for default fallback)
        """
        self._backend_hint = backend
        entries = self._extract_mdc_entries()

        # Resolve the expected component name so we can scope card selection
        # and avoid picking up LoRA-adapter cards that share the same CR but
        # carry a different component/model identity.
        expected_component: Optional[str] = None
        try:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
            service = get_service_from_sub_component_type_or_name(
                deployment, sub_component_type
            )
            expected_component = service.name
        except PlannerError:
            expected_component = build_worker_info_from_defaults(
                backend, sub_component_type
            ).component_name

        for entry in entries:
            is_prefill = self._mdc_entry_is_prefill(entry)
            if sub_component_type == SubComponentType.PREFILL and not is_prefill:
                continue
            if sub_component_type == SubComponentType.DECODE and is_prefill:
                continue

            entry_component = entry.get("component")
            if (
                entry_component
                and expected_component
                and entry_component != expected_component
            ):
                logger.debug(
                    f"Skipping MDC entry with component={entry_component!r}, "
                    f"expected {expected_component!r} for {sub_component_type.value}"
                )
                continue

            info = self._build_worker_info_from_mdc(entry, sub_component_type)
            logger.info(
                f"Built {sub_component_type.value} WorkerInfo from MDC: "
                f"{info.summary()}"
            )
            return info

        # No MDC entry found -- fall back entirely to defaults + DGD arg parsing
        logger.warning(
            f"No DynamoWorkerMetadata CR found for {sub_component_type.value}. "
            f"Workers may not be registered yet. Falling back to defaults."
        )
        info = build_worker_info_from_defaults(backend, sub_component_type)

        # Try to enrich model_name from DGD container args
        try:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
            service = get_service_from_sub_component_type_or_name(
                deployment, sub_component_type
            )
            info.k8s_name = service.name
            arg_model = service.get_model_name()
            if arg_model:
                info.model_name = arg_model
                logger.info(
                    f"Enriched {sub_component_type.value} WorkerInfo model name "
                    f"from DGD args: {arg_model}"
                )
        except PlannerError as e:
            logger.info(
                f"Could not enrich WorkerInfo from DGD for {sub_component_type.value}: {e}"
            )

        logger.info(
            f"Using fallback WorkerInfo for {sub_component_type.value}: {info.summary()}"
        )
        return info

    def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        """
        Get actual ready worker counts for prefill and decode from DGD status.

        Returns:
            tuple[int, int, bool]: (prefill_count, decode_count, is_stable)
            - is_stable: False if any service is in a rollout (scaling should be skipped)
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        prefill_count = 0
        decode_count = 0
        all_stable = True

        if prefill_component_name:
            service = get_service_from_sub_component_type_or_name(
                deployment,
                SubComponentType.PREFILL,
                component_name=prefill_component_name,
            )
            ready_replicas, is_stable = self.kube_api.get_service_replica_status(
                deployment, service.name
            )
            if not is_stable:
                all_stable = False
            prefill_count = ready_replicas

        if decode_component_name:
            service = get_service_from_sub_component_type_or_name(
                deployment,
                SubComponentType.DECODE,
                component_name=decode_component_name,
            )
            ready_replicas, is_stable = self.kube_api.get_service_replica_status(
                deployment, service.name
            )
            if not is_stable:
                all_stable = False
            decode_count = ready_replicas

        return prefill_count, decode_count, all_stable

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise EmptyTargetReplicasError()

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        if not self.kube_api.is_deployment_ready(deployment):
            logger.warning(
                f"Deployment {self.graph_deployment_name} is not ready, ignoring this scaling"
            )
            return

        for target_replica in target_replicas:
            service = get_service_from_sub_component_type_or_name(
                deployment,
                target_replica.sub_component_type,
                component_name=target_replica.component_name,
            )
            current_replicas = service.number_replicas()
            if current_replicas != target_replica.desired_replicas:
                logger.info(
                    f"Updating {target_replica.sub_component_type.value} component {service.name} to desired replica count {target_replica.desired_replicas}"
                )
                self.kube_api.update_graph_replicas(
                    self.graph_deployment_name,
                    service.name,
                    target_replica.desired_replicas,
                )
            else:
                logger.info(
                    f"{target_replica.sub_component_type.value} component {service.name} already at desired replica count {target_replica.desired_replicas}, skipping"
                )

        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self.graph_deployment_name,
            )


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamo_namespace", type=str, default="dynamo")
    parser.add_argument("--k8s_namespace", type=str, default="default")
    parser.add_argument("--action", type=str, choices=["add", "remove"])
    parser.add_argument(
        "--component",
        type=str,
        choices=[t.value for t in SubComponentType],
        default=SubComponentType.PREFILL.value,
        help="Target sub-component to scale",
    )
    parser.add_argument("--blocking", action="store_true")
    args = parser.parse_args()
    connector = KubernetesConnector(
        args.dynamo_namespace, k8s_namespace=args.k8s_namespace
    )

    if args.action == "add":
        task = connector.add_component(SubComponentType(args.component), args.blocking)
    elif args.action == "remove":
        task = connector.remove_component(
            SubComponentType(args.component), args.blocking
        )
    asyncio.run(task)
