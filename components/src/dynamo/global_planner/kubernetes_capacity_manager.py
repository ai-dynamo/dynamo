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

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, TypeVar
from uuid import uuid4

from kubernetes import client
from kubernetes.aio.utils.retry import retry_on_conflict

from dynamo.common.async_utils import run_async_to_completion, run_sync_in_thread
from dynamo.global_planner.capacity_manager import (
    CapacityManager,
    PoolSnapshot,
    PoolSpec,
)
from dynamo.planner import KubernetesConnector, TargetReplica
from dynamo.planner.connectors.base import ReplicaUpdateResult
from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.kubernetes import (
    ReplicaUpdateAttempt,
    ReplicaUpdateFailure,
    ReplicaUpdateRetryState,
)
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

DEPLOYMENT_STRATEGY_ANNOTATION = "nvidia.com/deployment-strategy"
DEPLOYMENT_MAX_SURGE_ANNOTATION = "nvidia.com/deployment-rolling-update-max-surge"


@dataclass(frozen=True)
class _ReplicaSettlement:
    """Accepted replica transaction that must settle before dependent writes."""

    dgd_uid: str
    targets: dict[str, int]
    authorities: dict[str, tuple[bool, str | None]]
    uncertain: tuple[ReplicaUpdateAttempt, ...] = ()
    completion_error: ReplicaUpdateFailure | None = None
    protected: bool = False


class _SnapshotConflict(client.ApiException):
    """A stable Kubernetes identity changed during a bracketed observation."""

    def __init__(self, resource: str, name: str) -> None:
        super().__init__(
            status=409,
            reason=f"{resource} {name!r} changed during Pod inventory",
        )


class KubernetesCapacityManager(CapacityManager):
    """Observe / scale capacity via ``KubernetesConnector`` per deployment.

    A ``participant_id`` is ``"{namespace}/{deployment_name}"``. One
    ``KubernetesConnector`` is cached per participant.
    """

    def __init__(self, namespace: str, writer_id: str | None = None):
        self.namespace = namespace
        self._writer_id = writer_id or uuid4().hex
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

            self._clear_uid_bound_state(participant_id)
        self._observed_uids[participant_id] = uid

    def _clear_uid_bound_state(self, participant_id: str) -> None:
        """Drop observations and settlements tied to one DGD UID."""
        self._last_observed_pools.pop(participant_id, None)
        self._replica_uids.pop(participant_id, None)
        self._replica_settlements.pop(participant_id, None)

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

    @staticmethod
    def _named_inventory(objects: list[dict], resource: str) -> dict[str, dict]:
        """Index one Kubernetes LIST response by object name."""
        by_name: dict[str, dict] = {}
        for index, obj in enumerate(objects):
            metadata = obj.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{resource} list item {index} has malformed metadata")
            name = metadata.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{resource} list item {index} has no usable name")
            if name in by_name:
                raise ValueError(f"duplicate {resource} {name!r} in list response")
            by_name[name] = obj
        return by_name

    @staticmethod
    def _require_inventory_unchanged(
        before: Mapping[str, dict],
        after: Mapping[str, dict],
        names: set[str],
        resource: str,
    ) -> None:
        """Require identical object identities and versions around a Pod LIST."""
        for name in sorted(names):
            old = before.get(name)
            current = after.get(name)
            if old is None and current is None:
                continue
            if old is None or current is None:
                raise _SnapshotConflict(resource, name)
            old_metadata = old.get("metadata") or {}
            current_metadata = current.get("metadata") or {}
            if not isinstance(old_metadata, Mapping) or not isinstance(
                current_metadata, Mapping
            ):
                raise ValueError(f"{resource} {name!r} metadata is malformed")
            old_uid = old_metadata.get("uid")
            current_uid = current_metadata.get("uid")
            if not isinstance(old_uid, str) or not old_uid:
                raise ValueError(f"{resource} {name!r} metadata.uid is unavailable")
            if not isinstance(current_uid, str) or not current_uid:
                raise ValueError(f"{resource} {name!r} metadata.uid is unavailable")
            if old_uid != current_uid:
                raise _SnapshotConflict(resource, name)
            old_version = old_metadata.get("resourceVersion")
            current_version = current_metadata.get("resourceVersion")
            if not isinstance(old_version, str) or not old_version:
                raise ValueError(
                    f"{resource} {name!r} metadata.resourceVersion is unavailable"
                )
            if not isinstance(current_version, str) or not current_version:
                raise ValueError(
                    f"{resource} {name!r} metadata.resourceVersion is unavailable"
                )
            if old_version != current_version:
                raise _SnapshotConflict(resource, name)

    @staticmethod
    def _scaling_adapter_names(
        api: KubernetesAPI,
        deployments: Mapping[str, dict],
        deployment_names: set[str],
    ) -> set[str]:
        names: set[str] = set()
        for deployment_name in deployment_names:
            deployment = deployments.get(deployment_name)
            if deployment is None:
                continue
            for component_name, component in get_components_by_name(deployment).items():
                if "scalingAdapter" in component:
                    names.add(api.scaling_adapter_name(deployment_name, component_name))
        return names

    @staticmethod
    def _read_scaling_adapters(
        api: KubernetesAPI,
        deployment: dict,
    ) -> dict[str, dict]:
        """Read the DGDSAs declared by one DGD, indexed by adapter name."""
        metadata = deployment.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("DGD metadata is malformed")
        deployment_name = metadata.get("name")
        if not isinstance(deployment_name, str) or not deployment_name:
            raise ValueError("DGD metadata.name is unavailable")

        adapters: dict[str, dict] = {}
        for component_name, component in get_components_by_name(deployment).items():
            if "scalingAdapter" not in component:
                continue
            adapter_name = api.scaling_adapter_name(deployment_name, component_name)
            adapters[adapter_name] = api.get_service_scaling_adapter(
                deployment_name,
                component_name,
            )
        return adapters

    def _complete_namespace_inventory(
        self,
        api: KubernetesAPI,
    ) -> tuple[dict[str, dict], dict[str, dict], dict[str, list]]:
        """Collect a stable controller/Pod snapshot for one namespace."""
        deployments_before = self._named_inventory(api.list_graph_deployments(), "DGD")
        dgd_names = set(deployments_before)
        adapter_names_before = self._scaling_adapter_names(
            api, deployments_before, dgd_names
        )
        adapters_before = (
            self._named_inventory(api.list_scaling_adapters(), "DGDSA")
            if adapter_names_before
            else {}
        )

        pods_by_graph = api.partition_pods_by_graph(api.list_graph_pods())

        deployments_after = self._named_inventory(
            api.list_graph_deployment_metadata(),
            "DGD",
        )
        stable_dgd_names = set(deployments_before) | set(deployments_after)
        self._require_inventory_unchanged(
            deployments_before,
            deployments_after,
            stable_dgd_names,
            "DGD",
        )
        adapters_after = (
            self._named_inventory(api.list_scaling_adapter_metadata(), "DGDSA")
            if adapter_names_before
            else {}
        )
        self._require_inventory_unchanged(
            adapters_before,
            adapters_after,
            adapter_names_before,
            "DGDSA",
        )

        unknown_graphs = set(pods_by_graph) - set(deployments_after)
        if unknown_graphs:
            raise RuntimeError(
                "nonterminal Pods reference DGDs absent from the namespace "
                f"inventory: {sorted(unknown_graphs)}"
            )
        return deployments_before, adapters_before, pods_by_graph

    def _observe_explicit_complete(
        self,
        connectors_snapshot: list[tuple[str, KubernetesConnector]],
    ) -> PoolSnapshot:
        """Bracket one namespace Pod LIST with scoped reads of configured DGDs."""
        all_pools: PoolSnapshot = {}
        local_connectors = {
            key: connector
            for key, connector in connectors_snapshot
            if connector.kube_api.current_namespace == self.namespace
        }

        if not local_connectors:
            for key, connector in connectors_snapshot:
                try:
                    all_pools[key] = self._read_targeted_complete(
                        key,
                        connector,
                        self._component_roles.get(key),
                    )
                except Exception as error:
                    self._record_observation_failure(all_pools, key, error, True)
            return all_pools

        before: dict[str, tuple[dict, dict[str, dict], dict] | None] = {}
        for key, connector in local_connectors.items():
            try:
                deployment = connector.kube_api.get_graph_deployment(
                    connector.parent_dgd_name
                )
                adapters = self._read_scaling_adapters(
                    connector.kube_api,
                    deployment,
                )
                before[key] = (
                    deployment,
                    adapters,
                    connector.kube_api.fetch_authoritative_replica_targets(
                        deployment,
                        scaling_adapters=adapters,
                    ),
                )
            except DynamoGraphDeploymentNotFoundError:
                before[key] = None
            except Exception as error:
                self._record_observation_failure(all_pools, key, error, True)

        try:
            inventory_api = next(iter(local_connectors.values())).kube_api
            pods_by_graph = inventory_api.partition_pods_by_graph(
                inventory_api.list_graph_pods(
                    {
                        connector.parent_dgd_name
                        for connector in local_connectors.values()
                    }
                )
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to list Pods in {self.namespace}: {error}"
            ) from error

        for key, connector in connectors_snapshot:
            if key not in local_connectors:
                try:
                    all_pools[key] = self._read_targeted_complete(
                        key,
                        connector,
                        self._component_roles.get(key),
                    )
                except Exception as error:
                    self._record_observation_failure(all_pools, key, error, True)
                continue

            initial = before[key]
            if initial is None:
                try:
                    connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
                except DynamoGraphDeploymentNotFoundError as error:
                    if not self._mark_missing_deployment(key, connector):
                        raise RuntimeError(
                            f"observed DGD disappeared: {error}; capacity drain "
                            "cannot be proven"
                        ) from error
                    self._require_missing_deployment_drained(
                        connector,
                        pods_by_graph.get(connector.parent_dgd_name, []),
                    )
                    all_pools[key] = {}
                    continue
                except Exception as error:
                    self._record_observation_failure(all_pools, key, error, True)
                raise _SnapshotConflict("DGD", connector.parent_dgd_name)

            try:
                deployment_before, adapters_before, authoritative_before = initial
                try:
                    deployment = connector.kube_api.get_graph_deployment(
                        connector.parent_dgd_name
                    )
                except DynamoGraphDeploymentNotFoundError as error:
                    raise _SnapshotConflict("DGD", connector.parent_dgd_name) from error
                self._require_inventory_unchanged(
                    {connector.parent_dgd_name: deployment_before},
                    {connector.parent_dgd_name: deployment},
                    {connector.parent_dgd_name},
                    "DGD",
                )
                adapters = self._read_scaling_adapters(
                    connector.kube_api,
                    deployment,
                )
                adapter_names = set(adapters_before) | set(adapters)
                self._require_inventory_unchanged(
                    adapters_before,
                    adapters,
                    adapter_names,
                    "DGDSA",
                )
                authoritative = connector.kube_api.fetch_authoritative_replica_targets(
                    deployment,
                    scaling_adapters=adapters,
                )
                before_targets = self._authoritative_replica_targets(
                    authoritative_before
                )
                current_targets = self._authoritative_replica_targets(authoritative)
                if before_targets != current_targets:
                    raise RuntimeError(
                        "replica authorities changed during Pod inventory: "
                        f"before={before_targets}, after={current_targets}"
                    )
                pools = self._pools_from_deployment(
                    connector,
                    deployment,
                    self._component_roles.get(key),
                    require_settled=True,
                    pods=pods_by_graph.get(connector.parent_dgd_name, []),
                    authoritative_deployment=authoritative,
                )
                self._record_observed_identity(key, deployment)
                all_pools[key] = self._record_observed_pools(key, pools)
                if self.connectors.get(key) is connector:
                    self._missing_participants.discard(key)
            except Exception as error:
                self._record_observation_failure(all_pools, key, error, True)
        return all_pools

    def _read_targeted_complete(
        self,
        participant_id: str,
        connector: KubernetesConnector,
        role_hints: Optional[dict[str, str]],
    ) -> dict[str, PoolSpec]:
        """Read one stable DGD/Pod snapshot, including the missing-DGD case."""
        try:
            pools = self._read_pools(
                participant_id,
                connector,
                role_hints,
                require_settled=True,
            )
        except DynamoGraphDeploymentNotFoundError as error:
            pods = connector.kube_api.list_nonterminal_pods_for_graph(
                connector.parent_dgd_name
            )
            try:
                connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
            except DynamoGraphDeploymentNotFoundError as missing_error:
                if not self._mark_missing_deployment(participant_id, connector):
                    raise RuntimeError(
                        f"observed DGD disappeared: {missing_error}; capacity drain "
                        "cannot be proven"
                    ) from missing_error
                self._require_missing_deployment_drained(connector, pods)
                return {}
            raise _SnapshotConflict("DGD", connector.parent_dgd_name) from error

        if self.connectors.get(participant_id) is connector:
            self._missing_participants.discard(participant_id)
        return pools

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
        if require_complete and self._replica_settlements:
            participants = ", ".join(sorted(self._replica_settlements))
            raise RuntimeError(
                "replica writes are still awaiting authoritative settlement for "
                f"{participants}"
            )
        all_pools: PoolSnapshot = {}
        connectors_snapshot = list(self.connectors.items())
        if not self._observe_all_deployments:
            if require_complete:
                return self._observe_explicit_complete(connectors_snapshot)
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
            if require_complete:
                (
                    deployments_by_name,
                    scaling_adapters_by_name,
                    pods_by_graph,
                ) = self._complete_namespace_inventory(inventory_api)
            else:
                deployments_by_name = self._named_inventory(
                    inventory_api.list_graph_deployments(), "DGD"
                )
                adapter_names = self._scaling_adapter_names(
                    inventory_api,
                    deployments_by_name,
                    set(deployments_by_name),
                )
                scaling_adapters_by_name = (
                    self._named_inventory(
                        inventory_api.list_scaling_adapters(), "DGDSA"
                    )
                    if adapter_names
                    else {}
                )
                pods_by_graph = {}
            deployments_by_key = {
                f"{self.namespace}/{name}": deployment
                for name, deployment in deployments_by_name.items()
            }

        except _SnapshotConflict:
            raise
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
                if require_complete:
                    if not self._mark_missing_deployment(key, connector):
                        self._record_observation_failure(
                            all_pools,
                            key,
                            RuntimeError(
                                "observed DGD is absent from the namespace inventory; "
                                "capacity drain cannot be proven"
                            ),
                            True,
                        )
                    self._require_missing_deployment_drained(
                        connector,
                        pods_by_graph.get(connector.parent_dgd_name, []),
                    )
                    self._discard_connector(key, connector)
                elif not self._mark_missing_deployment(key, connector):
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
                if require_complete:
                    all_pools[key] = self._read_targeted_complete(
                        key,
                        connector,
                        self._component_roles.get(key),
                    )
                else:
                    all_pools[key] = self._read_pools(
                        key,
                        connector,
                        self._component_roles.get(key),
                    )
                if self.connectors.get(key) is connector:
                    self._missing_participants.discard(key)
            except DynamoGraphDeploymentNotFoundError as error:
                if self._mark_missing_deployment(key, connector):
                    if require_complete:
                        self._require_missing_deployment_drained(connector)
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
                pools = self._pools_from_deployment(
                    connector,
                    deployment,
                    self._component_roles.get(key),
                    scaling_adapters=scaling_adapters_by_name,
                    require_settled=require_complete,
                    pods=pods_by_graph.get(key.split("/", 1)[1], []),
                )
                self._record_observed_identity(
                    key,
                    deployment,
                )
                all_pools[key] = self._record_observed_pools(key, pools)
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
        if isinstance(error, _SnapshotConflict):
            raise error
        if require_complete:
            raise RuntimeError(
                f"Failed to read deployment for {participant_id}: {error}"
            ) from error
        logger.warning(f"Failed to read deployment for {participant_id}: {error}")
        all_pools[participant_id] = {}

    @staticmethod
    def _require_missing_deployment_drained(
        connector: KubernetesConnector,
        pods: list | None = None,
    ) -> None:
        """Reject a missing DGD while any of its nonterminal Pods remain."""
        if pods is None:
            pods = connector.kube_api.list_nonterminal_pods_for_graph(
                connector.parent_dgd_name
            )
        if pods:
            raise RuntimeError(
                f"DGD {connector.parent_dgd_name!r} is absent but still has "
                f"{len(pods)} nonterminal Pods"
            )

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
        *,
        require_settled: bool = False,
    ) -> dict[str, PoolSpec]:
        """Read and parse the current pool state for one deployment."""
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        authoritative_deployment = None
        pods = None
        if require_settled:
            adapters_before = self._read_scaling_adapters(
                connector.kube_api,
                deployment,
            )
            authoritative_before = (
                connector.kube_api.fetch_authoritative_replica_targets(
                    deployment,
                    scaling_adapters=adapters_before,
                )
            )
            pods = connector.kube_api.list_nonterminal_pods_for_graph(
                connector.parent_dgd_name
            )
            try:
                current = connector.kube_api.get_graph_deployment(
                    connector.parent_dgd_name
                )
            except DynamoGraphDeploymentNotFoundError as error:
                raise _SnapshotConflict("DGD", connector.parent_dgd_name) from error
            self._require_inventory_unchanged(
                {connector.parent_dgd_name: deployment},
                {connector.parent_dgd_name: current},
                {connector.parent_dgd_name},
                "DGD",
            )
            adapters = self._read_scaling_adapters(
                connector.kube_api,
                current,
            )
            adapter_names = set(adapters_before) | set(adapters)
            self._require_inventory_unchanged(
                adapters_before,
                adapters,
                adapter_names,
                "DGDSA",
            )
            authoritative_deployment = (
                connector.kube_api.fetch_authoritative_replica_targets(
                    current,
                    scaling_adapters=adapters,
                )
            )
            before_targets = self._authoritative_replica_targets(authoritative_before)
            current_targets = self._authoritative_replica_targets(
                authoritative_deployment
            )
            if before_targets != current_targets:
                raise RuntimeError(
                    "replica authorities changed during Pod inventory: "
                    f"before={before_targets}, after={current_targets}"
                )
            deployment = current
        pools = self._pools_from_deployment(
            connector,
            deployment,
            role_hints,
            require_settled=require_settled,
            pods=pods,
            authoritative_deployment=authoritative_deployment,
        )
        self._record_observed_identity(
            participant_id,
            deployment,
        )
        return self._record_observed_pools(participant_id, pools)

    @staticmethod
    def _authoritative_replica_targets(deployment: dict) -> dict[str, int]:
        """Return the replica state whose stability brackets a Pod LIST."""
        return {
            component_name: Service(
                name=component_name,
                service=component,
            ).number_replicas()
            for component_name, component in get_components_by_name(deployment).items()
        }

    def _pools_from_deployment(
        self,
        connector: KubernetesConnector,
        deployment: dict,
        role_hints: Optional[dict[str, str]] = None,
        *,
        scaling_adapters: Mapping[str, dict] | None = None,
        require_settled: bool = False,
        pods: list | None = None,
        authoritative_deployment: dict | None = None,
    ) -> dict[str, PoolSpec]:
        """Build pools from a DGD after resolving live replica authorities."""
        declared_deployment = deployment
        if authoritative_deployment is not None:
            deployment = authoritative_deployment
        elif scaling_adapters is None:
            deployment = connector.kube_api.fetch_authoritative_replica_targets(
                deployment
            )
        else:
            deployment = connector.kube_api.fetch_authoritative_replica_targets(
                deployment,
                scaling_adapters=scaling_adapters,
            )
        if require_settled:
            self._require_capacity_snapshot_settled(
                connector,
                declared_deployment,
                deployment,
                pods,
            )
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

    @staticmethod
    def _require_capacity_snapshot_settled(
        connector: KubernetesConnector,
        declared_deployment: dict,
        authoritative_deployment: dict,
        pods: list | None,
    ) -> None:
        """Fail closed while a budget snapshot can still undercount live GPUs."""
        api = connector.kube_api
        declared_components = get_components_by_name(declared_deployment)
        expected_targets: dict[str, int] = {}
        expected_authorities: dict[str, tuple[bool, str | None]] = {}
        for component_name, component in get_components_by_name(
            authoritative_deployment
        ).items():
            if get_component_type(component) == "planner":
                continue
            expected_targets[component_name] = Service(
                name=component_name,
                service=component,
            ).number_replicas()
            expected_authorities[component_name] = (
                "scalingAdapter" in declared_components.get(component_name, {}),
                None,
            )

        targets_stable, unsettled = api.replica_targets_stable(
            authoritative_deployment,
            expected_targets,
        )
        targets_projected, unprojected = api.adapter_targets_projected(
            declared_deployment,
            expected_targets,
            expected_authorities,
        )
        generation_observed = api.is_spec_generation_observed(declared_deployment)
        blocking_rollout, rollout_reason = api.is_rolling_update_blocking_settlement(
            declared_deployment
        )
        rolling = declared_deployment.get("status", {}).get("rollingUpdate") or {}
        if rolling.get("phase") == "Failed":
            raise RuntimeError(
                f"DGD rollout failed while observing capacity: {rolling.get('message', '')}"
            )
        if (
            not targets_stable
            or not targets_projected
            or not generation_observed
            or blocking_rollout
        ):
            raise RuntimeError(
                "replica capacity is still converging: "
                f"unsettled={unsettled}, unprojected={unprojected}, "
                f"observed={generation_observed}, "
                f"rollout={rollout_reason or 'settled'}"
            )

        unsafe_deployments = (
            KubernetesCapacityManager._deployment_components_with_surge(
                declared_deployment
            )
        )
        if unsafe_deployments:
            raise RuntimeError(
                "hard GPU budgets require Deployment components to use Recreate "
                "or maxSurge=0; unsafe components="
                f"{unsafe_deployments}"
            )

        if pods is None:
            dgd_name = declared_deployment.get("metadata", {}).get("name")
            if not isinstance(dgd_name, str) or not dgd_name:
                raise ValueError("DGD metadata.name is unavailable")
            pods = api.list_nonterminal_pods_for_graph(dgd_name)
        pods = api.exclude_checkpoint_capture_pods(pods)
        dgd_uid = declared_deployment.get("metadata", {}).get("uid")
        if not isinstance(dgd_uid, str) or not dgd_uid:
            raise ValueError("DGD metadata.uid is unavailable")
        identity_mismatches = api.pod_graph_uid_mismatches(pods, dgd_uid)
        component_mismatches = api.pod_component_mismatches(
            pods,
            authoritative_deployment,
        )
        if (
            identity_mismatches
            or component_mismatches
            or api.has_terminating_pods(pods)
            or not api.pcsg_pods_within_desired_replicas(
                authoritative_deployment,
                pods,
            )
            or not api.deployment_pods_within_desired_replicas(
                authoritative_deployment,
                pods,
            )
        ):
            raise RuntimeError("replica Pods are still draining")

    @staticmethod
    def _deployment_components_with_surge(deployment: dict) -> list[str]:
        """Return Deployment-backed components that can create unbudgeted Pods."""
        components = get_components_by_name(deployment)
        statuses = (deployment.get("status", {}) or {}).get("components", {}) or {}
        unsafe: list[str] = []
        for component_name, status in statuses.items():
            if (
                not isinstance(status, Mapping)
                or status.get("componentKind") != "Deployment"
                or not isinstance(status.get("gpusPerReplica"), (int, float))
                or status["gpusPerReplica"] <= 0
            ):
                continue
            component = components.get(component_name)
            if not isinstance(component, Mapping):
                unsafe.append(component_name)
                continue
            annotations = (
                (component.get("podTemplate") or {}).get("metadata") or {}
            ).get("annotations") or {}
            if annotations.get(DEPLOYMENT_STRATEGY_ANNOTATION) == "Recreate":
                continue
            if annotations.get(DEPLOYMENT_MAX_SURGE_ANNOTATION) in ("0", "0%"):
                continue
            unsafe.append(component_name)
        return unsafe

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
        write_admitted = False

        def admit_write() -> None:
            nonlocal write_admitted
            write_admitted = True

        async def operation() -> None:
            result = await self._scale_with_result(
                participant_id,
                targets,
                blocking,
                admit_write,
            )
            if result.accepted and (blocking or result.requires_settlement):
                await self.wait_for_ready(participant_id)

        try:
            await self.reconcile_pending_settlements(participant_id)
            await run_async_to_completion(
                operation(),
                complete_on_cancel=lambda: write_admitted,
            )
        except asyncio.CancelledError:
            self._protect_replica_settlement(participant_id)
            raise

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
        write_admitted = False

        def admit_write() -> None:
            nonlocal write_admitted
            write_admitted = True

        try:
            return await run_async_to_completion(
                self._scale_with_result(
                    participant_id,
                    targets,
                    blocking,
                    admit_write,
                ),
                complete_on_cancel=lambda: write_admitted,
            )
        except asyncio.CancelledError:
            self._protect_replica_settlement(participant_id)
            raise

    async def _scale_with_result(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
        admit_write: Callable[[], None],
    ) -> ReplicaUpdateResult:
        connector = self.connectors[participant_id]
        previous_settlement = self._replica_settlements.get(participant_id)
        if previous_settlement is not None:
            raise RuntimeError(
                f"participant {participant_id!r} has a replica write awaiting settlement"
            )
        observed_uid = self._observed_uids.get(participant_id)
        retry_states = self._replica_retry_states.get()
        if retry_states is None:
            retry_state = ReplicaUpdateRetryState(dgd_uid=observed_uid)
        else:
            retry_state = retry_states.setdefault(
                participant_id, ReplicaUpdateRetryState(dgd_uid=observed_uid)
            )
        admit_write()
        try:
            await connector.ensure_replica_writer_fence(self._writer_id)
            result = await connector.set_component_replicas_with_result(
                targets,
                blocking=blocking,
                retry_state=retry_state,
            )
        except ReplicaUpdateFailure as error:
            if retry_state.dgd_uid is None:
                raise RuntimeError(
                    "possibly committed replica update has no DGD identity"
                ) from error
            settled_targets, settled_authorities = self._attempt_settlement_maps(
                error.committed
            )
            self._replica_uids[participant_id] = retry_state.dgd_uid
            self._replica_settlements[participant_id] = _ReplicaSettlement(
                dgd_uid=retry_state.dgd_uid,
                targets=settled_targets,
                authorities=settled_authorities,
                uncertain=error.uncertain,
                completion_error=error,
                protected=True,
            )
            return ReplicaUpdateResult(
                accepted=True,
                applied=True,
                requires_settlement=True,
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

    def _protect_replica_settlement(self, participant_id: str) -> None:
        settlement = self._replica_settlements.get(participant_id)
        if settlement is not None and not settlement.protected:
            self._replica_settlements[participant_id] = replace(
                settlement,
                protected=True,
            )

    @staticmethod
    def _attempt_settlement_maps(
        attempts: tuple[ReplicaUpdateAttempt, ...],
    ) -> tuple[dict[str, int], dict[str, tuple[bool, str | None]]]:
        targets: dict[str, int] = {}
        authorities: dict[str, tuple[bool, str | None]] = {}
        for attempt in attempts:
            previous = targets.get(attempt.component_name)
            if previous is not None and previous != attempt.target.desired_replicas:
                raise RuntimeError(
                    f"component {attempt.component_name!r} has conflicting settlements"
                )
            targets[attempt.component_name] = attempt.target.desired_replicas
            authorities[attempt.component_name] = (
                attempt.is_adapter,
                attempt.authority_uid,
            )
        return targets, authorities

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
        try:
            if settlement.uncertain:
                resolved = await run_sync_in_thread(
                    connector.resolve_uncertain_replica_attempts,
                    settlement.dgd_uid,
                    settlement.uncertain,
                )
                resolved_targets, resolved_authorities = self._attempt_settlement_maps(
                    resolved
                )
                settlement = replace(
                    settlement,
                    targets={**settlement.targets, **resolved_targets},
                    authorities={**settlement.authorities, **resolved_authorities},
                    uncertain=(),
                )
                self._replica_settlements[participant_id] = settlement
            if settlement.targets:
                await connector.wait_for_replica_targets(
                    expected_dgd_uid=settlement.dgd_uid,
                    expected_targets=settlement.targets,
                    expected_authorities=settlement.authorities,
                )
        except (Exception, asyncio.CancelledError):
            current = self._replica_settlements.get(participant_id)
            if current is settlement and not current.protected:
                self._replica_settlements[participant_id] = replace(
                    current, protected=True
                )
            raise
        if self._replica_settlements.get(participant_id) is settlement:
            self._replica_settlements.pop(participant_id, None)
        if settlement.completion_error is not None:
            raise settlement.completion_error

    async def reconcile_pending_settlements(
        self, participant_id: str | None = None
    ) -> None:
        """Resume failed replica settlements before a new budget observation."""
        if participant_id is None:
            participants = tuple(self._replica_settlements)
        elif participant_id in self._replica_settlements:
            participants = (participant_id,)
        else:
            participants = ()
        for pending_participant_id in participants:
            try:
                await self.wait_for_ready(pending_participant_id)
            except ReplicaUpdateFailure:
                # The physical settlement completed and wait_for_ready removed
                # the marker before surfacing the previous request's error.
                if pending_participant_id in self._replica_settlements:
                    raise

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
        deployment = connector.kube_api.fetch_authoritative_replica_targets(deployment)
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
