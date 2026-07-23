# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes backend for the GlobalPlanner capacity control loop.

:class:`KubernetesCapacityManager` is the concrete
:class:`~dynamo.global_planner.capacity_manager.CapacityManager` — the only place
in the control loop that touches Kubernetes. A ``participant_id`` is
``"{namespace}/{deployment_name}"``.

Pool state is read from the v1beta1 DGD component model (``spec.components[]``):
a component's planner pool is its explicit ``prefill``/``decode`` type, or — for
a generic ``worker`` — the role hinted by its Planner's
``TargetReplica.component_name`` (remembered via :meth:`remember_roles`).
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
from dynamo.planner.monitoring.dgd_services import (
    V1BETA1_GENERIC_WORKER_COMPONENT_TYPE,
    Service,
    get_component_type,
    get_components_by_name,
    get_planner_component_role,
)

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
        # participant_id -> {component_name: planner_role}. Generic ``worker``
        # components need the role supplied by their Planner's TargetReplica;
        # specialized prefill/decode components do not need a hint.
        self._component_roles: dict[str, dict[str, str]] = {}

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

    def remember_roles(self, participant_id: str, targets: list[TargetReplica]) -> None:
        """Remember each target's ``component_name -> planner role`` hint so a
        later ``observe`` can map generic ``worker`` components to prefill/decode.
        """
        role_hints = self._component_roles.setdefault(participant_id, {})
        for target in targets:
            if target.component_name:
                role_hints[target.component_name] = target.sub_component_type.value

    # ------------------------------------------------------------------ #
    # Observe                                                            #
    # ------------------------------------------------------------------ #

    def observe(self, require_complete: bool = False) -> PoolSnapshot:
        """Read current pool state for every known deployment.

        When ``require_complete`` is true, any unreadable deployment fails the
        whole snapshot so budget enforcement cannot under-count cluster usage.

        Snapshots ``self.connectors`` up-front via ``list(...)``: this runs on a
        worker thread (the orchestrator calls it via ``asyncio.to_thread``), and
        a concurrent first-time request for another deployment can insert into
        the dict before it blocks on the scale lock. Without the snapshot, that
        insertion races iteration.
        """
        all_pools: PoolSnapshot = {}
        for key, connector in list(self.connectors.items()):
            try:
                all_pools[key] = self._read_pools(
                    connector, self._component_roles.get(key)
                )
            except Exception as e:
                if require_complete:
                    raise RuntimeError(
                        f"Failed to read deployment for {key}: {e}"
                    ) from e
                logger.warning(f"Failed to read deployment for {key}: {e}")
                all_pools[key] = {}
        return all_pools

    def _component_role(
        self,
        component_name: str,
        component: dict,
        role_hints: dict[str, str],
    ) -> str:
        """Resolve a component's planner pool role.

        An explicit ``prefill``/``decode`` type wins; otherwise a generic or
        untyped worker takes the remembered role hint (if any); else ``""``.
        """
        explicit_role = get_planner_component_role(component)
        if explicit_role:
            return explicit_role
        if get_component_type(component) in (
            "",
            V1BETA1_GENERIC_WORKER_COMPONENT_TYPE,
        ):
            return role_hints.get(component_name, "")
        return ""

    @staticmethod
    def _gpu_per_replica(component: dict, service: Service) -> int:
        """GPUs per replica = main-container GPUs × node count (multinode)."""
        multinode = component.get("multinode")
        node_count = 1 if multinode is None else multinode.get("nodeCount", 2)
        return service.get_gpu_count() * int(node_count)

    @staticmethod
    def _record_pool_component(
        components_by_pool: dict[str, str],
        pool_key: str,
        component_name: str,
        deployment_name: str,
    ) -> None:
        """Guard against two components resolving to the same planner pool."""
        previous_component = components_by_pool.get(pool_key)
        if previous_component is not None:
            raise ValueError(
                f"Deployment {deployment_name!r} components {previous_component!r} "
                f"and {component_name!r} both resolve to planner pool {pool_key!r}"
            )
        components_by_pool[pool_key] = component_name

    def _read_pools(
        self,
        connector: KubernetesConnector,
        role_hints: Optional[dict[str, str]] = None,
    ) -> dict[str, PoolSpec]:
        """Read the current pool state for one deployment (v1beta1 components).

        An unmapped generic worker is keyed by component name so it still
        contributes to the total; once its Planner sends a request, the
        component-name hint replaces that key with its prefill/decode role.
        """
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        pools: dict[str, PoolSpec] = {}
        components_by_pool: dict[str, str] = {}
        role_hints = role_hints or {}
        for component_name, component in get_components_by_name(deployment).items():
            pool_key = self._component_role(component_name, component, role_hints)
            component_type = get_component_type(component)
            service = Service(name=component_name, service=component)
            gpu_per_replica = None
            if not pool_key and component_type in (
                "",
                V1BETA1_GENERIC_WORKER_COMPONENT_TYPE,
            ):
                try:
                    gpu_per_replica = self._gpu_per_replica(component, service)
                except ValueError:
                    if component_type == "":
                        # An untyped non-worker component is not a pool.
                        continue
                    raise
                pool_key = component_name
            if not pool_key:
                continue
            self._record_pool_component(
                components_by_pool,
                pool_key,
                component_name,
                connector.parent_dgd_name,
            )
            pools[pool_key] = PoolSpec(
                sub_type=pool_key,
                component_name=component_name,
                current_replicas=service.number_replicas(),
                gpu_per_replica=(
                    gpu_per_replica
                    if gpu_per_replica is not None
                    else self._gpu_per_replica(component, service)
                ),
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
        role_hints = self._component_roles.get(participant_id, {})
        current_replicas: dict[str, int] = {}
        components_by_pool: dict[str, str] = {}
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        for component_name, component in get_components_by_name(deployment).items():
            sub_type = self._component_role(component_name, component, role_hints)
            if sub_type:
                self._record_pool_component(
                    components_by_pool,
                    sub_type,
                    component_name,
                    connector.parent_dgd_name,
                )
                current_replicas[sub_type] = Service(
                    name=component_name, service=component
                ).number_replicas()
        return current_replicas
