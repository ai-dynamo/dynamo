# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes backend for the GlobalPlanner capacity control loop.

:class:`KubernetesCapacityManager` is the concrete
:class:`~dynamo.global_planner.capacity_manager.CapacityManager` — the only place
in the control loop that touches Kubernetes. A ``participant_id`` is
``"{namespace}/{deployment_name}"``, and the read/write logic is lifted
verbatim from the pre-refactor ``ScaleRequestHandler`` so behavior is unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

from dynamo.global_planner.capacity_manager import (
    CapacityManager,
    PoolSnapshot,
    PoolSpec,
)
from dynamo.planner import KubernetesConnector, TargetReplica
from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI

logger = logging.getLogger(__name__)


class KubernetesCapacityManager(CapacityManager):
    """Observe / scale capacity via ``KubernetesConnector`` per deployment.

    A ``participant_id`` is ``"{namespace}/{deployment_name}"``. One
    ``KubernetesConnector`` is cached per participant.
    """

    def __init__(self, namespace: str):
        self.namespace = namespace
        # participant_id -> KubernetesConnector
        self.connectors: dict[str, KubernetesConnector] = {}

    # ------------------------------------------------------------------ #
    # Discovery / registration                                           #
    # ------------------------------------------------------------------ #

    def _managed_deployment_names(
        self, managed_deployments: Optional[set[str]]
    ) -> Optional[set[str]]:
        """Derive the deployment names this GlobalPlanner manages.

        Returns a set of deployment names in explicit mode, or ``None`` in
        implicit mode. The operator convention is
        ``DYN_NAMESPACE = "{namespace}-{deployment_name}"``, so the deployment
        name is the managed identity with the namespace prefix stripped.
        """
        if managed_deployments is None:
            return None

        prefix = f"{self.namespace}-"
        names = set()
        for deployment in managed_deployments:
            if deployment.startswith(prefix):
                names.add(deployment[len(prefix) :])
            else:
                logger.warning(
                    f"Managed deployment '{deployment}' does not start with "
                    f"expected prefix '{prefix}'; cannot derive deployment name"
                )
        return names

    def discover(self, managed_deployments: Optional[set[str]]) -> list[str]:
        """Pre-populate connectors for deployments managed by this GlobalPlanner.

        Ensures the GPU budget accounts for deployments that already exist at
        startup, even if they haven't sent a scale request yet. In explicit mode
        (``managed_deployments`` set) only matching deployments are discovered; in
        implicit mode (``None``) all deployments in the namespace are discovered.
        """
        managed_deployment_names = self._managed_deployment_names(managed_deployments)
        try:
            kube_api = KubernetesAPI(self.namespace)
            dgds = kube_api.list_graph_deployments()
            discovered: list[str] = []
            for dgd in dgds:
                name = dgd.get("metadata", {}).get("name", "")
                if not name:
                    continue
                # In explicit mode, skip deployments not in the managed set.
                if (
                    managed_deployment_names is not None
                    and name not in managed_deployment_names
                ):
                    continue
                participant_id = f"{self.namespace}/{name}"
                if participant_id not in self.connectors:
                    connector = KubernetesConnector(
                        dynamo_namespace="discovered",
                        k8s_namespace=self.namespace,
                        parent_dgd_name=name,
                        raise_not_ready=True,
                    )
                    self.connectors[participant_id] = connector
                discovered.append(name)
            logger.info(
                f"Discovered {len(discovered)} existing deployments: {discovered}"
            )
            return discovered
        except Exception as e:
            logger.warning(f"Failed to discover existing deployments: {e}")
            return []

    def ensure_participant(
        self,
        participant_id: str,
        caller_name: str,
        namespace: str,
        deployment_name: str,
    ) -> None:
        if participant_id not in self.connectors:
            connector = KubernetesConnector(
                dynamo_namespace=caller_name,
                k8s_namespace=namespace,
                parent_dgd_name=deployment_name,
                raise_not_ready=True,
            )
            self.connectors[participant_id] = connector
            logger.debug(f"Created new connector for {participant_id}")
        else:
            logger.debug(f"Reusing cached connector for {participant_id}")

    def participant_exists(self, participant_id: str) -> bool:
        return participant_id in self.connectors

    # ------------------------------------------------------------------ #
    # Observe                                                            #
    # ------------------------------------------------------------------ #

    def observe(self) -> PoolSnapshot:
        """Read current pool state for every known deployment.

        Snapshots ``self.connectors`` up-front via ``list(...)``: this runs on a
        worker thread (the orchestrator calls it via ``asyncio.to_thread``), and
        a concurrent first-time request for another deployment can insert into
        the dict before it blocks on the scale lock. Without the snapshot, that
        insertion races iteration.
        """
        all_pools: PoolSnapshot = {}
        for key, connector in list(self.connectors.items()):
            try:
                all_pools[key] = self._read_pools(connector)
            except Exception as e:
                logger.warning(f"Failed to read deployment for {key}: {e}")
                all_pools[key] = {}
        return all_pools

    def _read_pools(self, connector: KubernetesConnector) -> dict[str, PoolSpec]:
        """Read the current pool state for one deployment.

        Returns a map from sub_type to PoolSpec. Pools with 0 gpu_per_replica are
        included for completeness but contribute 0 to budget math. GPU count is
        read from ``spec.services[].resources.limits.gpu`` only.
        """
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        pools: dict[str, PoolSpec] = {}
        services = deployment.get("spec", {}).get("services", {})
        for svc_spec in services.values():
            sub_type = svc_spec.get("subComponentType", "")
            if not sub_type:
                continue
            gpu_per_replica = int(
                svc_spec.get("resources", {}).get("limits", {}).get("gpu", 0)
            )
            replicas = svc_spec.get("replicas", 0)
            pools[sub_type] = PoolSpec(
                sub_type=sub_type,
                current_replicas=replicas,
                gpu_per_replica=gpu_per_replica,
            )
        return pools

    # ------------------------------------------------------------------ #
    # Scale                                                              #
    # ------------------------------------------------------------------ #

    async def scale(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
    ) -> None:
        """Apply desired replica targets to one participant.

        Raises ``DynamoGraphDeploymentNotReadyError`` when the participant is not
        in a state that can accept scaling; the orchestrator maps that to a soft
        rejection.
        """
        connector = self.connectors[participant_id]
        await connector.set_component_replicas(targets, blocking=blocking)

    def current_replicas(self, participant_id: str) -> dict[str, int]:
        connector = self.connectors[participant_id]
        current_replicas: dict[str, int] = {}
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        for service_name, service_spec in deployment["spec"]["services"].items():
            sub_type = service_spec.get("subComponentType", "")
            if sub_type:
                current_replicas[sub_type] = service_spec.get("replicas", 0)
        return current_replicas
