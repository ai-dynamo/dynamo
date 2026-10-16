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
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, TypeVar

from kubernetes import client
from kubernetes.aio.utils.retry import retry_on_conflict

from dynamo.global_planner.capacity_manager import (
    CapacityManager,
    PoolSnapshot,
    PoolSpec,
)
from dynamo.planner import KubernetesConnector, TargetReplica
from dynamo.planner.connectors.base import ReplicaUpdateResult
from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.kubernetes import ReplicaUpdateRetryState
from dynamo.planner.errors import DynamoGraphDeploymentNotFoundError
from dynamo.planner.monitoring.dgd_services import (
    V1BETA1_GENERIC_WORKER_COMPONENT_TYPE,
    Service,
    get_component_type,
    get_components_by_name,
    get_planner_component_role,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class _ReplicaSettlement:
    """Accepted replica transaction that must settle before dependent writes."""

    dgd_uid: str
    targets: dict[str, int]
    authorities: dict[str, tuple[bool, str | None]]


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
        self._observe_all_deployments = False
        self._configured_deployment_names: set[str] | None = set()
        self._inventory_api: KubernetesAPI | None = None
        # Never-observed configured DGDs remain zero-capacity creation probes.
        # Once a UID has been observed, disappearance or replacement fails
        # closed because Kubernetes GC does not prove capacity has drained.
        self._missing_participants: set[str] = set()
        self._last_observed_pools: dict[str, dict[str, PoolSpec]] = {}
        self._observed_uids: dict[str, str] = {}
        self._replica_retry_states: ContextVar[
            dict[str, ReplicaUpdateRetryState] | None
        ] = ContextVar("global_planner_replica_retry_states", default=None)
        self._replica_uids: dict[str, str] = {}
        self._replica_settlements: dict[str, _ReplicaSettlement] = {}

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

        Ensures the GPU budget accounts for deployments that have not sent a scale
        request yet. Explicit mode verifies each configured name with one scoped
        GET; implicit mode discovers every deployment in the namespace with one
        LIST.
        """
        managed_deployment_names = self._managed_deployment_names(managed_deployments)
        self._configured_deployment_names = managed_deployment_names
        self._observe_all_deployments = managed_deployment_names is None
        if managed_deployment_names is None:
            try:
                kube_api = self._get_inventory_api()
                dgds = kube_api.list_graph_deployments()
            except Exception as error:
                logger.warning(f"Failed to discover existing deployments: {error}")
                return []
            deployment_names = []
            for dgd in dgds:
                name = dgd.get("metadata", {}).get("name", "")
                if not name:
                    continue
                participant_id = f"{self.namespace}/{name}"
                try:
                    self._record_observed_identity(participant_id, dgd)
                    self._missing_participants.discard(participant_id)
                except Exception as error:
                    logger.warning(
                        "Failed to verify discovered deployment %s: %s",
                        participant_id,
                        error,
                    )
                deployment_names.append(name)
        else:
            self._inventory_api = None
            deployment_names = []
            for name in sorted(managed_deployment_names):
                participant_id = f"{self.namespace}/{name}"
                connector = self.connectors.get(participant_id)
                if connector is None:
                    connector = self._new_connector("discovered", self.namespace, name)
                    self.connectors[participant_id] = connector
                try:
                    deployment = connector.kube_api.get_graph_deployment(name)
                    self._record_observed_identity(participant_id, deployment)
                except DynamoGraphDeploymentNotFoundError:
                    self._mark_missing_deployment(participant_id, connector)
                    logger.info(
                        "Configured deployment %s/%s does not exist yet",
                        self.namespace,
                        name,
                    )
                    continue
                except Exception as error:
                    # Keep an unknown deployment in the accounting set. A later
                    # complete observation will retry the read and fail closed.
                    logger.warning(
                        "Failed to verify configured deployment %s/%s: %s",
                        self.namespace,
                        name,
                        error,
                    )
                else:
                    self._missing_participants.discard(participant_id)
                deployment_names.append(name)

        registered: list[str] = []
        for name in deployment_names:
            participant_id = f"{self.namespace}/{name}"
            if participant_id not in self.connectors:
                self.connectors[participant_id] = self._new_connector(
                    "discovered", self.namespace, name
                )
            registered.append(name)
        logger.info(f"Registered {len(registered)} managed deployments: {registered}")
        return registered

    @staticmethod
    def _new_connector(
        caller_name: str, namespace: str, deployment_name: str
    ) -> KubernetesConnector:
        return KubernetesConnector(
            dynamo_namespace=caller_name,
            k8s_namespace=namespace,
            parent_dgd_name=deployment_name,
            raise_not_ready=True,
            retry_conflicts=False,
        )

    def _discard_connector(
        self, participant_id: str, connector: KubernetesConnector
    ) -> None:
        """Forget a never-observed connector if it is still the current instance."""
        if self.connectors.get(participant_id) is connector:
            del self.connectors[participant_id]
            self._component_roles.pop(participant_id, None)
            self._missing_participants.discard(participant_id)
            self._last_observed_pools.pop(participant_id, None)
            self._observed_uids.pop(participant_id, None)
            self._replica_uids.pop(participant_id, None)
            self._replica_settlements.pop(participant_id, None)

    def _get_inventory_api(self) -> KubernetesAPI:
        if self._inventory_api is None:
            self._inventory_api = KubernetesAPI(self.namespace)
        return self._inventory_api

    def _record_observed_identity(
        self,
        participant_id: str,
        deployment: dict,
        *,
        allow_replacement: bool = False,
    ) -> None:
        metadata = deployment.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("DGD metadata is malformed")
        uid = metadata.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("DGD metadata.uid is unavailable")
        observed_uid = self._observed_uids.get(participant_id)
        if observed_uid is not None and observed_uid != uid:
            if not allow_replacement:
                raise RuntimeError(
                    f"DGD identity changed from {observed_uid!r} to {uid!r}; "
                    "capacity drain cannot be proven"
                )

            # Unbounded refresh may adopt the replacement after dropping all
            # observations tied to the previous DGD generation.
            self._last_observed_pools.pop(participant_id, None)
            self._replica_uids.pop(participant_id, None)
            self._replica_settlements.pop(participant_id, None)
        self._observed_uids[participant_id] = uid

    def _record_observed_pools(
        self, participant_id: str, pools: dict[str, PoolSpec]
    ) -> dict[str, PoolSpec]:
        self._last_observed_pools[participant_id] = dict(pools)
        return pools

    def _mark_missing_deployment(
        self, participant_id: str, connector: KubernetesConnector
    ) -> bool:
        """Mark a missing DGD and report whether it has never been observed."""
        if self.connectors.get(participant_id) is connector:
            self._missing_participants.add(participant_id)
        return participant_id not in self._observed_uids

    def ensure_participant(
        self,
        participant_id: str,
        caller_name: str,
        namespace: str,
        deployment_name: str,
    ) -> None:
        if participant_id not in self.connectors:
            self.connectors[participant_id] = self._new_connector(
                caller_name, namespace, deployment_name
            )
            logger.debug(f"Created new connector for {participant_id}")
        else:
            logger.debug(f"Reusing cached connector for {participant_id}")

    def participant_exists(self, participant_id: str) -> bool:
        return (
            participant_id in self.connectors
            and participant_id not in self._missing_participants
        )

    def refresh_participant(self, participant_id: str) -> None:
        """Refresh one DGD identity for unbounded orchestration."""
        connector = self.connectors.get(participant_id)
        if connector is None:
            return
        try:
            deployment = connector.kube_api.get_graph_deployment(
                connector.parent_dgd_name
            )
        except DynamoGraphDeploymentNotFoundError:
            never_observed = self._mark_missing_deployment(participant_id, connector)
            if not never_observed:
                raise RuntimeError(
                    f"Observed DGD {participant_id} disappeared; capacity drain "
                    "cannot be proven"
                )
            return
        self._record_observed_identity(
            participant_id, deployment, allow_replacement=True
        )
        if self.connectors.get(participant_id) is connector:
            self._missing_participants.discard(participant_id)

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
        worker thread (the orchestrator calls it via ``run_sync_in_thread``), and
        a concurrent first-time request for another deployment can insert into
        the dict before it blocks on the scale lock. Without the snapshot, that
        insertion races iteration.
        """
        all_pools: PoolSnapshot = {}
        connectors_snapshot = list(self.connectors.items())
        if not self._observe_all_deployments:
            for key, connector in connectors_snapshot:
                try:
                    all_pools[key] = self._read_pools(
                        key,
                        connector,
                        self._component_roles.get(key),
                    )
                    if self.connectors.get(key) is connector:
                        self._missing_participants.discard(key)
                except DynamoGraphDeploymentNotFoundError as error:
                    if self._mark_missing_deployment(key, connector):
                        all_pools[key] = {}
                        continue
                    self._record_observation_failure(
                        all_pools,
                        key,
                        RuntimeError(
                            f"observed DGD disappeared: {error}; capacity drain "
                            "cannot be proven"
                        ),
                        require_complete,
                    )
                except Exception as error:
                    self._record_observation_failure(
                        all_pools, key, error, require_complete
                    )
            return all_pools

        # Implicit mode manages every DGD in the planner namespace, so one
        # namespace LIST is both the intended scope and cheaper than per-DGD GETs.
        try:
            inventory_api = self._get_inventory_api()
            deployments = inventory_api.list_graph_deployments()
            deployments_by_key: dict[str, dict] = {}
            for index, deployment in enumerate(deployments):
                metadata = deployment.get("metadata")
                if not isinstance(metadata, Mapping):
                    raise ValueError(f"DGD list item {index} has malformed metadata")
                name = metadata.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"DGD list item {index} has no usable name")
                key = f"{self.namespace}/{name}"
                if key in deployments_by_key:
                    raise ValueError(f"duplicate DGD {name!r} in list response")
                deployments_by_key[key] = deployment

        except Exception as error:
            if require_complete:
                raise RuntimeError(
                    f"Failed to list deployments in {self.namespace}: {error}"
                ) from error
            logger.warning(
                "Failed to list deployments in %s: %s", self.namespace, error
            )
            return {key: {} for key, _ in connectors_snapshot}

        for key in deployments_by_key:
            if key not in self.connectors:
                _, name = key.split("/", 1)
                self.connectors[key] = self._new_connector(
                    "discovered", self.namespace, name
                )

        for key, connector in connectors_snapshot:
            if key in deployments_by_key:
                continue
            connector_namespace = key.partition("/")[0]
            if connector_namespace == self.namespace:
                if not self._mark_missing_deployment(key, connector):
                    self._record_observation_failure(
                        all_pools,
                        key,
                        RuntimeError(
                            "observed DGD is absent from the namespace inventory; "
                            "capacity drain cannot be proven"
                        ),
                        require_complete,
                    )
                else:
                    self._discard_connector(key, connector)
                continue
            try:
                all_pools[key] = self._read_pools(
                    key,
                    connector,
                    self._component_roles.get(key),
                )
                if self.connectors.get(key) is connector:
                    self._missing_participants.discard(key)
            except DynamoGraphDeploymentNotFoundError as error:
                if self._mark_missing_deployment(key, connector):
                    all_pools[key] = {}
                    continue
                self._record_observation_failure(
                    all_pools,
                    key,
                    RuntimeError(
                        f"observed DGD disappeared: {error}; capacity drain "
                        "cannot be proven"
                    ),
                    require_complete,
                )
            except Exception as error:
                self._record_observation_failure(
                    all_pools, key, error, require_complete
                )

        for key, deployment in deployments_by_key.items():
            connector = self.connectors[key]
            try:
                self._record_observed_identity(key, deployment)
                all_pools[key] = self._record_observed_pools(
                    key,
                    self._pools_from_deployment(
                        connector,
                        deployment,
                        self._component_roles.get(key),
                    ),
                )
                if self.connectors.get(key) is connector:
                    self._missing_participants.discard(key)
            except Exception as error:
                self._record_observation_failure(
                    all_pools, key, error, require_complete
                )
        return all_pools

    @staticmethod
    def _record_observation_failure(
        all_pools: PoolSnapshot,
        participant_id: str,
        error: Exception,
        require_complete: bool,
    ) -> None:
        if require_complete:
            raise RuntimeError(
                f"Failed to read deployment for {participant_id}: {error}"
            ) from error
        logger.warning(f"Failed to read deployment for {participant_id}: {error}")
        all_pools[participant_id] = {}

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
    def _gpu_per_replica(deployment: dict, service: Service) -> int:
        """Return the operator-projected unique GPU cost of one replica."""
        return service.get_gpu_shape(deployment).gpus_per_replica

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
        participant_id: str,
        connector: KubernetesConnector,
        role_hints: Optional[dict[str, str]] = None,
    ) -> dict[str, PoolSpec]:
        """Read and parse the current pool state for one deployment."""
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        self._record_observed_identity(participant_id, deployment)
        pools = self._pools_from_deployment(connector, deployment, role_hints)
        return self._record_observed_pools(participant_id, pools)

    def _pools_from_deployment(
        self,
        connector: KubernetesConnector,
        deployment: dict,
        role_hints: Optional[dict[str, str]] = None,
    ) -> dict[str, PoolSpec]:
        """Build one participant's pools from a listed DGD object."""
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
                    gpu_per_replica = self._gpu_per_replica(deployment, service)
                except ValueError:
                    if component_type == "":
                        # An untyped non-worker component is not a pool.
                        continue
                    raise
                if gpu_per_replica == 0:
                    if component_type == "":
                        # Explicit zero marks an observed non-GPU component.
                        continue
                    raise ValueError(
                        f"Planner pool component {component_name!r} has an observed zero-GPU shape"
                    )
                pool_key = component_name
            if not pool_key:
                continue
            self._record_pool_component(
                components_by_pool,
                pool_key,
                component_name,
                connector.parent_dgd_name,
            )
            if gpu_per_replica is None:
                gpu_per_replica = self._gpu_per_replica(deployment, service)
            if gpu_per_replica == 0:
                raise ValueError(
                    f"Planner pool component {component_name!r} has an observed zero-GPU shape"
                )
            pools[pool_key] = PoolSpec(
                sub_type=pool_key,
                component_name=component_name,
                current_replicas=service.number_replicas(),
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
        """Apply desired targets while preserving the CapacityManager contract."""
        await self.scale_with_result(participant_id, targets, blocking)

    async def scale_with_result(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
    ) -> ReplicaUpdateResult:
        """Apply desired replica targets to one participant.

        Raises ``DynamoGraphDeploymentNotReadyError`` when the participant is not
        in a state that can accept scaling; the orchestrator maps that to a soft
        rejection.
        """
        connector = self.connectors[participant_id]
        self._replica_settlements.pop(participant_id, None)
        observed_uid = self._observed_uids.get(participant_id)
        retry_states = self._replica_retry_states.get()
        if retry_states is None:
            retry_state = ReplicaUpdateRetryState(dgd_uid=observed_uid)
        else:
            retry_state = retry_states.setdefault(
                participant_id, ReplicaUpdateRetryState(dgd_uid=observed_uid)
            )
        result = await connector.set_component_replicas_with_result(
            targets,
            blocking=blocking,
            retry_state=retry_state,
        )
        if retry_state.dgd_uid is not None:
            self._replica_uids[participant_id] = retry_state.dgd_uid
        if result.accepted:
            if retry_state.dgd_uid is None:
                raise RuntimeError("accepted replica update has no DGD identity")
            settled_targets: dict[str, int] = {}
            settled_authorities: dict[str, tuple[bool, str | None]] = {}
            pools = self._last_observed_pools.get(participant_id)
            for target in targets:
                selector = (
                    target.sub_component_type.value,
                    target.component_name,
                )
                authority = retry_state.authorities.get(selector)
                if authority is None:
                    raise RuntimeError(
                        f"accepted replica update for {selector!r} has no authority"
                    )
                resolved_component, is_adapter, authority_uid = authority
                previous_target = settled_targets.get(resolved_component)
                if (
                    previous_target is not None
                    and previous_target != target.desired_replicas
                ):
                    raise RuntimeError(
                        f"component {resolved_component!r} has conflicting targets"
                    )
                settled_targets[resolved_component] = target.desired_replicas
                settled_authorities[resolved_component] = (
                    is_adapter,
                    authority_uid,
                )
                if pools is not None:
                    pool_key = next(
                        (
                            key
                            for key, candidate in pools.items()
                            if candidate.component_name == resolved_component
                        ),
                        target.sub_component_type.value,
                    )
                    pool = pools.get(pool_key)
                    if pool is None:
                        continue
                    pools[pool_key] = replace(
                        pool, current_replicas=target.desired_replicas
                    )
            self._replica_settlements[participant_id] = _ReplicaSettlement(
                dgd_uid=retry_state.dgd_uid,
                targets=settled_targets,
                authorities=settled_authorities,
            )
        return result

    def supports_split_readiness(self) -> bool:
        return True

    async def wait_for_ready(self, participant_id: str) -> None:
        """Wait for one participant without repeating its accepted write."""
        connector = self.connectors[participant_id]
        settlement = self._replica_settlements.get(participant_id)
        if settlement is None:
            raise RuntimeError(
                f"participant {participant_id!r} has no accepted replica settlement"
            )
        await connector.wait_for_replica_targets(
            expected_dgd_uid=settlement.dgd_uid,
            expected_targets=settlement.targets,
            expected_authorities=settlement.authorities,
        )
        if self._replica_settlements.get(participant_id) is settlement:
            self._replica_settlements.pop(participant_id, None)

    async def run_with_conflict_retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Re-run the complete budget transaction after a Kubernetes conflict."""

        token = self._replica_retry_states.set({})
        try:
            return await retry_on_conflict(operation)
        finally:
            self._replica_retry_states.reset(token)

    def is_conflict(self, error: Exception) -> bool:
        return isinstance(error, client.ApiException) and error.status == 409

    def current_replicas(self, participant_id: str) -> dict[str, int]:
        connector = self.connectors[participant_id]
        role_hints = dict(self._component_roles.get(participant_id, {}))
        return self._current_replicas(connector, role_hints)

    def capture_current_replicas(
        self, participant_id: str
    ) -> Callable[[], dict[str, int]]:
        connector = self.connectors[participant_id]
        role_hints = dict(self._component_roles.get(participant_id, {}))
        expected_uid = self._replica_uids.get(participant_id)
        return partial(self._current_replicas, connector, role_hints, expected_uid)

    def _current_replicas(
        self,
        connector: KubernetesConnector,
        role_hints: dict[str, str],
        expected_uid: str | None = None,
    ) -> dict[str, int]:
        current_replicas: dict[str, int] = {}
        components_by_pool: dict[str, str] = {}
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        if expected_uid is not None:
            current_uid = (deployment.get("metadata", {}) or {}).get("uid")
            if current_uid != expected_uid:
                raise RuntimeError("DGD identity changed before replica readback")
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
