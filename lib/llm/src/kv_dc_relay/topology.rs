// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Namespace-scoped serving topology projected independently from endpoint-local CKF pools.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::protocols::EndpointId;
use parking_lot::Mutex;
use tokio::sync::watch;

use super::discovery::{DcMembershipView, EndpointMembership};
use super::identity::{CanonicalModelId, DcPoolCatalog, ModelTarget, WorkerRole};
use crate::worker_type::WorkerType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyReadinessState {
    Ready,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyMember {
    pub endpoint: EndpointId,
    pub roles: Vec<WorkerRole>,
    pub pool_id: Option<PoolId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterReadiness {
    pub model: CanonicalModelId,
    pub state: TopologyReadinessState,
    pub missing_roles: Vec<WorkerRole>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyEntry {
    pub namespace: String,
    pub model: CanonicalModelId,
    pub state: TopologyReadinessState,
    pub present_roles: Vec<WorkerRole>,
    pub missing_roles: Vec<WorkerRole>,
    pub members: Vec<TopologyMember>,
    pub duplicate_role_endpoints: Vec<WorkerRole>,
    pub legacy_fallback_active: bool,
    pub adapters: Vec<AdapterReadiness>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TopologySnapshot {
    pub revision: u64,
    pub entries: Vec<TopologyEntry>,
}

/// Normalized equivalent of one Dynamo WorkerSet for readiness evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyUnit {
    pub(crate) worker_type: Option<WorkerType>,
    pub(crate) live_count: usize,
    pub(crate) needs: Vec<Vec<WorkerType>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TopologyEvaluation {
    pub(crate) ready: bool,
    pub(crate) present_roles: Vec<WorkerRole>,
    pub(crate) missing_roles: Vec<WorkerRole>,
    pub(crate) has_legacy: bool,
}

/// Mirrors `Model::evaluate_namespace` without depending on private WorkerSet state.
pub(crate) fn evaluate_topology(units: &[TopologyUnit]) -> TopologyEvaluation {
    let mut present = HashSet::new();
    let mut missing = HashSet::new();
    let mut has_legacy = false;
    let mut has_live_worker = false;

    for unit in units {
        has_live_worker |= unit.live_count != 0;
        match unit.worker_type {
            Some(worker_type) if unit.live_count != 0 => {
                present.insert(worker_type);
            }
            Some(_) => {}
            None => has_legacy = true,
        }
    }

    if has_legacy {
        return TopologyEvaluation {
            ready: has_live_worker,
            present_roles: sorted_worker_roles(present),
            missing_roles: Vec::new(),
            has_legacy,
        };
    }

    for unit in units {
        let Some(worker_type) = unit.worker_type else {
            continue;
        };
        if !present.contains(&worker_type) {
            missing.insert(worker_type);
        }
        if unit.live_count == 0 || unit.needs.is_empty() {
            continue;
        }
        let satisfied = unit
            .needs
            .iter()
            .any(|alternative| alternative.iter().all(|needed| present.contains(needed)));
        if !satisfied {
            for needed in unit.needs.iter().flatten() {
                if !present.contains(needed) {
                    missing.insert(*needed);
                }
            }
        }
    }

    TopologyEvaluation {
        ready: has_live_worker && missing.is_empty(),
        present_roles: sorted_worker_roles(present),
        missing_roles: sorted_worker_roles(missing),
        has_legacy,
    }
}

#[derive(Default)]
struct AdapterAggregate {
    has_live_member: bool,
    live_roles: HashSet<WorkerRole>,
}

#[derive(Default)]
struct TopologyAggregate {
    units: Vec<TopologyUnit>,
    members: Vec<TopologyMember>,
    adapters: BTreeMap<CanonicalModelId, AdapterAggregate>,
    availability_authoritative: bool,
    initialized: bool,
}

impl TopologyAggregate {
    fn observe_authority(&mut self, authoritative: bool) {
        if !self.initialized {
            self.availability_authoritative = true;
            self.initialized = true;
        }
        self.availability_authoritative &= authoritative;
    }
}

#[derive(Default)]
struct TopologyProjectionInputs {
    membership: DcMembershipView,
    availability: HashMap<EndpointId, Option<HashSet<WorkerId>>>,
    availability_owners: HashMap<EndpointId, u64>,
    pools: HashMap<EndpointId, PoolId>,
    revision: u64,
}

/// Host-owned publication state. PoolRegistry contributes only its read-only catalog projection.
pub(crate) struct TopologyPublisher {
    state: Mutex<TopologyProjectionInputs>,
    sender: watch::Sender<Arc<TopologySnapshot>>,
}

impl TopologyPublisher {
    pub(crate) fn new(membership: DcMembershipView, catalog: &DcPoolCatalog) -> Self {
        let pools = pool_links(catalog);
        let entries = derive_topology(&membership, &HashMap::new(), &pools);
        let revision = 1;
        let (sender, _) = watch::channel(Arc::new(TopologySnapshot { revision, entries }));
        Self {
            state: Mutex::new(TopologyProjectionInputs {
                membership,
                pools,
                revision,
                ..TopologyProjectionInputs::default()
            }),
            sender,
        }
    }

    pub(crate) fn watch(&self) -> watch::Receiver<Arc<TopologySnapshot>> {
        self.sender.subscribe()
    }

    pub(crate) fn snapshot(&self) -> Arc<TopologySnapshot> {
        self.sender.borrow().clone()
    }

    pub(crate) fn replace_membership(&self, membership: DcMembershipView) {
        let mut state = self.state.lock();
        if state.membership == membership {
            return;
        }
        state
            .availability
            .retain(|endpoint, _| membership.endpoints.contains_key(endpoint));
        state
            .availability_owners
            .retain(|endpoint, _| membership.endpoints.contains_key(endpoint));
        state.membership = membership;
        publish_if_changed(&mut state, &self.sender);
    }

    /// Claims readiness publication for one endpoint-slot incarnation.
    ///
    /// A later slot claim fences delayed writes from the retired slot. Replacing an owner also
    /// clears its last availability snapshot until the new slot publishes an authoritative one.
    pub(crate) fn claim_availability(&self, endpoint: EndpointId, slot_incarnation: u64) {
        let mut state = self.state.lock();
        if !state.membership.endpoints.contains_key(&endpoint) {
            return;
        }
        if state
            .availability_owners
            .insert(endpoint.clone(), slot_incarnation)
            == Some(slot_incarnation)
        {
            return;
        }
        if state.availability.remove(&endpoint).is_some() {
            publish_if_changed(&mut state, &self.sender);
        }
    }

    pub(crate) fn replace_catalog(&self, catalog: &DcPoolCatalog) {
        let pools = pool_links(catalog);
        let mut state = self.state.lock();
        if state.pools == pools {
            return;
        }
        state.pools = pools;
        publish_if_changed(&mut state, &self.sender);
    }

    /// `None` means the endpoint's instance watchers have not produced an authoritative
    /// snapshot. `Some(empty)` is authoritative unavailability.
    pub(crate) fn replace_availability(
        &self,
        endpoint: EndpointId,
        slot_incarnation: u64,
        live_workers: Option<HashSet<WorkerId>>,
    ) {
        let mut state = self.state.lock();
        if state.availability_owners.get(&endpoint) != Some(&slot_incarnation) {
            return;
        }
        if state.availability.get(&endpoint) == Some(&live_workers) {
            return;
        }
        state.availability.insert(endpoint, live_workers);
        publish_if_changed(&mut state, &self.sender);
    }
}

fn publish_if_changed(
    state: &mut TopologyProjectionInputs,
    sender: &watch::Sender<Arc<TopologySnapshot>>,
) {
    let entries = derive_topology(&state.membership, &state.availability, &state.pools);
    if sender.borrow().entries == entries {
        return;
    }
    state.revision = state.revision.saturating_add(1);
    sender.send_replace(Arc::new(TopologySnapshot {
        revision: state.revision,
        entries,
    }));
}

fn pool_links(catalog: &DcPoolCatalog) -> HashMap<EndpointId, PoolId> {
    let mut links = HashMap::with_capacity(catalog.pools().len());
    let mut duplicates = HashSet::new();
    for descriptor in catalog.pools() {
        let endpoint = descriptor.serving_endpoint();
        if links.contains_key(endpoint) {
            duplicates.insert(endpoint.clone());
        } else {
            links.insert(endpoint.clone(), descriptor.pool_id());
        }
    }
    for endpoint in duplicates {
        links.remove(&endpoint);
        tracing::error!(
            %endpoint,
            "multiple Relay pools claim one serving endpoint; omitting its topology pool link"
        );
    }
    links
}

fn derive_topology(
    membership: &DcMembershipView,
    availability: &HashMap<EndpointId, Option<HashSet<WorkerId>>>,
    pools: &HashMap<EndpointId, PoolId>,
) -> Vec<TopologyEntry> {
    let mut groups = BTreeMap::<(String, CanonicalModelId), TopologyAggregate>::new();
    let mut live_endpoint_types =
        HashMap::<(String, CanonicalModelId), HashMap<WorkerType, usize>>::new();

    for (endpoint, endpoint_membership) in membership.endpoints.iter() {
        if !endpoint_membership.conflicts.is_empty() {
            continue;
        }
        let base_models = endpoint_membership
            .registrations
            .iter()
            .filter_map(|registration| match registration.target() {
                ModelTarget::Base { base_model } => Some(base_model.clone()),
                ModelTarget::Lora { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        let endpoint_availability = availability.get(endpoint).and_then(Option::as_ref);

        for base_model in base_models {
            let group = groups
                .entry((endpoint_membership.namespace.clone(), base_model.clone()))
                .or_default();
            group.observe_authority(endpoint_availability.is_some());
            group.members.push(TopologyMember {
                endpoint: endpoint.clone(),
                roles: endpoint_membership.roles.clone(),
                pool_id: pools.get(endpoint).copied(),
            });
            group
                .units
                .extend(topology_units(endpoint_membership, endpoint_availability));
            let counts = live_endpoint_types
                .entry((endpoint_membership.namespace.clone(), base_model.clone()))
                .or_default();
            for worker_type in live_worker_types(endpoint_membership, endpoint_availability) {
                *counts.entry(worker_type).or_default() += 1;
            }
            collect_adapter_membership(
                group,
                endpoint_membership,
                &base_model,
                endpoint_availability,
            );
        }
    }

    groups
        .into_iter()
        .map(|((namespace, model), mut group)| {
            let duplicate_role_endpoints = duplicate_role_endpoints(
                live_endpoint_types
                    .remove(&(namespace.clone(), model.clone()))
                    .unwrap_or_default(),
            );
            group.members.sort_unstable_by(|left, right| {
                left.endpoint.to_string().cmp(&right.endpoint.to_string())
            });
            let evaluation = evaluate_topology(&group.units);
            // Parity with `Model::evaluate_namespace`: an ambiguous topology (more than one
            // live endpoint serving a non-Aggregated role) is not ready in the core either.
            let state = if !group.availability_authoritative {
                TopologyReadinessState::Unknown
            } else if evaluation.ready
                && (evaluation.has_legacy || duplicate_role_endpoints.is_empty())
            {
                TopologyReadinessState::Ready
            } else {
                TopologyReadinessState::Unavailable
            };
            let adapters = derive_adapters(&group.adapters, &group.members, state, &evaluation);
            TopologyEntry {
                namespace,
                model,
                state,
                present_roles: if state == TopologyReadinessState::Unknown {
                    Vec::new()
                } else {
                    evaluation.present_roles
                },
                missing_roles: if state == TopologyReadinessState::Unknown {
                    Vec::new()
                } else {
                    evaluation.missing_roles
                },
                members: group.members,
                duplicate_role_endpoints,
                legacy_fallback_active: evaluation.has_legacy,
                adapters,
            }
        })
        .collect()
}

fn topology_units(
    membership: &EndpointMembership,
    live_workers: Option<&HashSet<WorkerId>>,
) -> Vec<TopologyUnit> {
    let mut groups = HashMap::<(Option<WorkerType>, Vec<Vec<WorkerType>>), usize>::new();
    for (&worker_id, topology) in &membership.worker_topology {
        let live = live_workers.is_some_and(|workers| workers.contains(&worker_id));
        let live_count = groups
            .entry((topology.worker_type, topology.needs.clone()))
            .or_default();
        *live_count += usize::from(live);
    }
    groups
        .into_iter()
        .map(|((worker_type, needs), live_count)| TopologyUnit {
            worker_type,
            live_count,
            needs,
        })
        .collect()
}

fn collect_adapter_membership(
    group: &mut TopologyAggregate,
    membership: &EndpointMembership,
    base_model: &CanonicalModelId,
    live_workers: Option<&HashSet<WorkerId>>,
) {
    for (adapter, adapter_membership) in &membership.adapters {
        if &adapter_membership.base_model != base_model {
            continue;
        }
        let aggregate = group.adapters.entry(adapter.clone()).or_default();
        for worker_id in adapter_membership.workers.keys() {
            if !live_workers.is_some_and(|workers| workers.contains(worker_id)) {
                continue;
            }
            aggregate.has_live_member = true;
            if let Some(role) = membership
                .worker_topology
                .get(worker_id)
                .and_then(|topology| topology.worker_type)
                .map(|worker_type| WorkerRole::from_worker_type(Some(worker_type)))
                && role != WorkerRole::Encode
            {
                aggregate.live_roles.insert(role);
            }
        }
    }
}

fn derive_adapters(
    adapters: &BTreeMap<CanonicalModelId, AdapterAggregate>,
    members: &[TopologyMember],
    base_state: TopologyReadinessState,
    evaluation: &TopologyEvaluation,
) -> Vec<AdapterReadiness> {
    let required_roles = members
        .iter()
        .flat_map(|member| member.roles.iter().copied())
        .filter(|role| {
            matches!(
                role,
                WorkerRole::Prefill | WorkerRole::Decode | WorkerRole::Aggregated
            )
        })
        .collect::<BTreeSet<_>>();

    adapters
        .iter()
        .map(|(model, aggregate)| {
            let mut missing = required_roles
                .iter()
                .copied()
                .filter(|role| !aggregate.live_roles.contains(role))
                .collect::<BTreeSet<_>>();
            missing.extend(evaluation.missing_roles.iter().copied());
            let state = match base_state {
                TopologyReadinessState::Unknown => TopologyReadinessState::Unknown,
                TopologyReadinessState::Unavailable => TopologyReadinessState::Unavailable,
                TopologyReadinessState::Ready if evaluation.has_legacy => {
                    if aggregate.has_live_member {
                        TopologyReadinessState::Ready
                    } else {
                        TopologyReadinessState::Unavailable
                    }
                }
                TopologyReadinessState::Ready
                    if missing.is_empty() && aggregate.has_live_member =>
                {
                    TopologyReadinessState::Ready
                }
                TopologyReadinessState::Ready => TopologyReadinessState::Unavailable,
            };
            AdapterReadiness {
                model: model.clone(),
                state,
                missing_roles: if state == TopologyReadinessState::Unknown {
                    Vec::new()
                } else {
                    missing.into_iter().collect()
                },
            }
        })
        .collect()
}

/// Non-Aggregated roles served by more than one live endpoint. The core treats this
/// topology as ambiguous and not ready; the role list explains the gate to consumers.
fn duplicate_role_endpoints(live_endpoint_counts: HashMap<WorkerType, usize>) -> Vec<WorkerRole> {
    let mut roles = live_endpoint_counts
        .into_iter()
        .filter(|&(worker_type, endpoints)| worker_type != WorkerType::Aggregated && endpoints > 1)
        .map(|(worker_type, _)| WorkerRole::from_worker_type(Some(worker_type)))
        .collect::<Vec<_>>();
    roles.sort_unstable();
    roles
}

/// Worker types with at least one live worker on this endpoint, deduplicated.
fn live_worker_types(
    membership: &EndpointMembership,
    live_workers: Option<&HashSet<WorkerId>>,
) -> HashSet<WorkerType> {
    membership
        .worker_topology
        .iter()
        .filter(|(worker_id, _)| live_workers.is_some_and(|workers| workers.contains(worker_id)))
        .filter_map(|(_, topology)| topology.worker_type)
        .collect()
}

fn sorted_worker_roles(roles: HashSet<WorkerType>) -> Vec<WorkerRole> {
    let mut roles = roles
        .into_iter()
        .map(|worker_type| WorkerRole::from_worker_type(Some(worker_type)))
        .collect::<Vec<_>>();
    roles.sort_unstable();
    roles
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dynamo_kv_router::identity::DcId;
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState, ProducerIdentity};

    use super::super::discovery::{
        AdapterMembership, AdapterWorkerMembership, DomainWorkerTopology, EndpointMembership,
    };
    use super::super::identity::{
        CanonicalModelRegistration, DcPoolDescriptor, DcRelayIdentity, ModelTarget,
    };
    use super::super::resolution::resolve_indexer_domain;
    use super::*;
    use crate::model_card::ModelDeploymentCard;

    fn unit(
        worker_type: Option<WorkerType>,
        live_count: usize,
        needs: Vec<Vec<WorkerType>>,
    ) -> TopologyUnit {
        TopologyUnit {
            worker_type,
            live_count,
            needs,
        }
    }

    fn model(value: &str) -> CanonicalModelId {
        CanonicalModelId::new(value).unwrap()
    }

    fn endpoint(
        endpoint: &str,
        canonical_model: &str,
        worker_id: WorkerId,
        worker_type: Option<WorkerType>,
        needs: Vec<Vec<WorkerType>>,
    ) -> EndpointMembership {
        let endpoint = EndpointId::from(endpoint);
        let mut card = ModelDeploymentCard::with_name_only(canonical_model);
        card.source_path = Some(canonical_model.to_string());
        card.kv_cache_block_size = 64;
        let domain = resolve_indexer_domain(&card, &endpoint).unwrap();
        EndpointMembership {
            endpoint: endpoint.clone(),
            generation: 1,
            domain: Some(domain),
            namespace: endpoint.namespace,
            registrations: vec![CanonicalModelRegistration::new(
                model(canonical_model),
                Vec::new(),
            )],
            models: vec![canonical_model.to_string()],
            aliases: Vec::new(),
            roles: vec![WorkerRole::from_worker_type(worker_type)],
            runtime_configs: HashMap::new(),
            worker_topology: HashMap::from([(
                worker_id,
                DomainWorkerTopology { worker_type, needs },
            )]),
            adapters: HashMap::new(),
            conflicts: Vec::new(),
        }
    }

    fn with_adapter(
        mut membership: EndpointMembership,
        base_model: &str,
        adapter: &str,
        worker_id: WorkerId,
    ) -> EndpointMembership {
        let base_model = model(base_model);
        let adapter = model(adapter);
        membership
            .registrations
            .push(CanonicalModelRegistration::with_target(
                adapter.clone(),
                ModelTarget::Lora {
                    base_model: base_model.clone(),
                    adapter: adapter.clone(),
                },
                Vec::new(),
            ));
        membership.adapters.insert(
            adapter,
            AdapterMembership {
                base_model,
                workers: HashMap::from([(
                    worker_id,
                    AdapterWorkerMembership {
                        max_gpu_lora_count: Some(8),
                    },
                )]),
            },
        );
        membership
    }

    fn view(memberships: Vec<EndpointMembership>) -> DcMembershipView {
        DcMembershipView {
            endpoints: Arc::new(
                memberships
                    .into_iter()
                    .map(|membership| (membership.endpoint.clone(), membership))
                    .collect(),
            ),
        }
    }

    fn catalog(view: &DcMembershipView, layout_generation: u64) -> DcPoolCatalog {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture CKF")
            .format();
        let mut memberships = view.endpoints.values().collect::<Vec<_>>();
        memberships.sort_unstable_by_key(|membership| membership.endpoint.to_string());
        let pools = memberships
            .into_iter()
            .map(|membership| {
                let domain = membership.domain.as_ref().expect("materializable fixture");
                let pool_id = PoolId::new(domain.id, DcId::new(3));
                DcPoolDescriptor::new(
                    ProducerIdentity::new(pool_id, 11, layout_generation, format),
                    membership.endpoint.clone(),
                    Arc::from(membership.registrations.clone()),
                    domain.query_semantics,
                    Arc::from(membership.roles.clone()),
                )
            })
            .collect();
        DcPoolCatalog::new(DcRelayIdentity::new(7, 11), layout_generation, pools)
    }

    fn publisher(view: &DcMembershipView) -> TopologyPublisher {
        let publisher = TopologyPublisher::new(view.clone(), &catalog(view, 1));
        for (endpoint, membership) in view.endpoints.iter() {
            publisher.claim_availability(endpoint.clone(), membership.generation);
        }
        publisher
    }

    #[test]
    fn duplicate_endpoint_catalog_links_fail_closed() {
        let membership = endpoint(
            "production.backend.generate",
            "llama",
            1,
            Some(WorkerType::Aggregated),
            Vec::new(),
        );
        let endpoint = membership.endpoint.clone();
        let domain = membership.domain.as_ref().unwrap();
        let format = DcCkfState::new(CkfConfig::new(32)).unwrap().format();
        let descriptor = |dc_id, layout_generation| {
            let pool_id = PoolId::new(domain.id, DcId::new(dc_id));
            DcPoolDescriptor::new(
                ProducerIdentity::new(pool_id, 11, layout_generation, format),
                endpoint.clone(),
                Arc::from(membership.registrations.clone()),
                domain.query_semantics,
                Arc::from(membership.roles.clone()),
            )
        };
        let catalog = DcPoolCatalog::new(
            DcRelayIdentity::new(7, 11),
            2,
            vec![descriptor(3, 1), descriptor(4, 2)],
        );

        assert!(!pool_links(&catalog).contains_key(&endpoint));
    }

    fn publish_live(publisher: &TopologyPublisher, view: &DcMembershipView) {
        for (endpoint, membership) in view.endpoints.iter() {
            publisher.claim_availability(endpoint.clone(), membership.generation);
            publisher.replace_availability(
                endpoint.clone(),
                membership.generation,
                Some(membership.worker_topology.keys().copied().collect()),
            );
        }
    }

    fn entry<'a>(snapshot: &'a TopologySnapshot, canonical_model: &str) -> &'a TopologyEntry {
        snapshot
            .entries
            .iter()
            .find(|entry| entry.model.as_str() == canonical_model)
            .expect("topology entry")
    }

    #[test]
    fn evaluator_matches_aggregated_pd_epd_and_dead_worker_semantics() {
        let aggregated = evaluate_topology(&[unit(Some(WorkerType::Aggregated), 1, vec![])]);
        assert!(aggregated.ready);
        assert_eq!(aggregated.present_roles, [WorkerRole::Aggregated]);

        let pd = evaluate_topology(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 1, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(pd.ready);

        let epd = evaluate_topology(&[
            unit(
                Some(WorkerType::Encode),
                1,
                vec![
                    vec![WorkerType::Prefill, WorkerType::Decode],
                    vec![WorkerType::Aggregated],
                ],
            ),
            unit(Some(WorkerType::Prefill), 1, vec![]),
            unit(Some(WorkerType::Decode), 1, vec![]),
        ]);
        assert!(epd.ready);

        let missing_decode = evaluate_topology(&[
            unit(Some(WorkerType::Prefill), 1, vec![vec![WorkerType::Decode]]),
            unit(Some(WorkerType::Decode), 0, vec![vec![WorkerType::Prefill]]),
        ]);
        assert!(!missing_decode.ready);
        assert_eq!(missing_decode.missing_roles, [WorkerRole::Decode]);
    }

    #[test]
    fn evaluator_matches_legacy_fallback_for_legacy_only_and_mixed_inputs() {
        for units in [
            vec![unit(None, 1, vec![])],
            vec![
                unit(None, 1, vec![]),
                unit(Some(WorkerType::Decode), 0, vec![vec![WorkerType::Prefill]]),
            ],
        ] {
            let evaluation = evaluate_topology(&units);
            assert!(evaluation.ready);
            assert!(evaluation.has_legacy);
            assert!(evaluation.missing_roles.is_empty());
        }
        let unavailable = evaluate_topology(&[unit(None, 0, vec![])]);
        assert!(!unavailable.ready);
        assert!(unavailable.has_legacy);
    }

    #[test]
    fn evaluator_ignores_needs_of_dead_workers_but_reports_their_declared_role() {
        let evaluation = evaluate_topology(&[
            unit(Some(WorkerType::Aggregated), 1, vec![]),
            unit(
                Some(WorkerType::Encode),
                0,
                vec![vec![WorkerType::Prefill, WorkerType::Decode]],
            ),
        ]);
        assert!(!evaluation.ready);
        assert_eq!(evaluation.missing_roles, [WorkerRole::Encode]);
        assert!(!evaluation.missing_roles.contains(&WorkerRole::Prefill));
        assert!(!evaluation.missing_roles.contains(&WorkerRole::Decode));
    }

    #[test]
    fn empty_topology_is_not_ready() {
        let evaluation = evaluate_topology(&[]);
        assert!(!evaluation.ready);
        assert!(!evaluation.has_legacy);
        assert!(evaluation.present_roles.is_empty());
        assert!(evaluation.missing_roles.is_empty());
    }

    #[test]
    fn aggregated_models_in_one_namespace_have_independent_ready_entries() {
        let view = view(vec![
            endpoint(
                "production.llama.generate",
                "meta/llama-3-70b",
                1,
                Some(WorkerType::Aggregated),
                Vec::new(),
            ),
            endpoint(
                "production.mistral.generate",
                "mistralai/mixtral-8x7b",
                2,
                Some(WorkerType::Aggregated),
                Vec::new(),
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();

        assert_eq!(snapshot.entries.len(), 2);
        assert!(snapshot.entries.iter().all(|entry| {
            entry.namespace == "production"
                && entry.state == TopologyReadinessState::Ready
                && entry.present_roles == [WorkerRole::Aggregated]
                && entry.duplicate_role_endpoints.is_empty()
        }));
    }

    #[test]
    fn pd_joins_endpoint_local_pools_into_one_ready_namespace_topology() {
        let view = view(vec![
            endpoint(
                "production.prefill.generate",
                "meta/llama-3-70b",
                1,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.backend.generate",
                "meta/llama-3-70b",
                2,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "meta/llama-3-70b");

        assert_eq!(topology.state, TopologyReadinessState::Ready);
        assert_eq!(
            topology.present_roles,
            [WorkerRole::Prefill, WorkerRole::Decode]
        );
        assert_eq!(topology.members.len(), 2);
        assert!(
            topology
                .members
                .iter()
                .all(|member| member.pool_id.is_some())
        );
        assert!(topology.duplicate_role_endpoints.is_empty());
    }

    #[test]
    fn epd_keeps_encode_readiness_without_materializing_an_empty_pool() {
        let encode = EndpointId::from("production.encoder.generate");
        let view = view(vec![
            endpoint(
                "production.encoder.generate",
                "vision-language",
                1,
                Some(WorkerType::Encode),
                vec![
                    vec![WorkerType::Prefill, WorkerType::Decode],
                    vec![WorkerType::Aggregated],
                ],
            ),
            endpoint(
                "production.prefill.generate",
                "vision-language",
                2,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.backend.generate",
                "vision-language",
                3,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let complete_catalog = catalog(&view, 1);
        let catalog_without_encode = DcPoolCatalog::new(
            complete_catalog.identity(),
            complete_catalog.revision(),
            complete_catalog
                .pools()
                .iter()
                .filter(|descriptor| descriptor.serving_endpoint() != &encode)
                .cloned()
                .collect(),
        );
        let publisher = TopologyPublisher::new(view.clone(), &catalog_without_encode);
        publish_live(&publisher, &view);
        let ready = publisher.snapshot();
        let topology = entry(&ready, "vision-language");
        assert_eq!(topology.state, TopologyReadinessState::Ready);
        let encode_member = topology
            .members
            .iter()
            .find(|member| member.endpoint == encode)
            .expect("encode member");
        assert_eq!(encode_member.roles, [WorkerRole::Encode]);
        assert_eq!(encode_member.pool_id, None);

        publisher.replace_catalog(&complete_catalog);
        let materialized = publisher.snapshot();
        let encode_member = entry(&materialized, "vision-language")
            .members
            .iter()
            .find(|member| member.endpoint == encode)
            .expect("encode member");
        assert!(encode_member.pool_id.is_some());

        publisher.replace_availability(encode, 1, Some(HashSet::new()));
        let unavailable = publisher.snapshot();
        let topology = entry(&unavailable, "vision-language");
        assert_eq!(topology.state, TopologyReadinessState::Unavailable);
        assert_eq!(topology.missing_roles, [WorkerRole::Encode]);
    }

    #[test]
    fn duplicate_pd_endpoints_report_each_duplicated_role() {
        let view = view(vec![
            endpoint(
                "production.prefill-a.generate",
                "llama",
                1,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.prefill-b.generate",
                "llama",
                2,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.decode-a.generate",
                "llama",
                3,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
            endpoint(
                "production.decode-b.generate",
                "llama",
                4,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "llama");

        // Parity with the core evaluation: an ambiguous role topology is not ready,
        // and the duplicated roles explain the gate.
        assert_eq!(topology.state, TopologyReadinessState::Unavailable);
        assert_eq!(
            topology.duplicate_role_endpoints,
            [WorkerRole::Prefill, WorkerRole::Decode]
        );
    }

    #[test]
    fn duplicate_role_endpoints_reports_only_the_duplicated_role() {
        let view = view(vec![
            endpoint(
                "production.prefill-a.generate",
                "llama",
                1,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.prefill-b.generate",
                "llama",
                2,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.decode.generate",
                "llama",
                3,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();

        assert_eq!(
            entry(&snapshot, "llama").duplicate_role_endpoints,
            [WorkerRole::Prefill]
        );
    }

    #[test]
    fn fenced_membership_does_not_contribute_to_topology() {
        let healthy = endpoint(
            "production.backend.generate",
            "llama",
            1,
            Some(WorkerType::Aggregated),
            Vec::new(),
        );
        let mut fenced = endpoint(
            "production.rogue.generate",
            "llama",
            2,
            Some(WorkerType::Prefill),
            Vec::new(),
        );
        fenced
            .conflicts
            .push(super::super::discovery::MaterializationConflict::Endpoint {
                endpoint: fenced.endpoint.clone(),
                reason: "endpoint resolves to multiple indexer domains".to_string(),
            });
        let view = view(vec![healthy, fenced]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "llama");

        assert_eq!(topology.state, TopologyReadinessState::Ready);
        assert_eq!(topology.members.len(), 1);
        assert_eq!(
            topology.members[0].endpoint,
            EndpointId::from("production.backend.generate")
        );
        assert!(!topology.present_roles.contains(&WorkerRole::Prefill));
    }

    #[test]
    fn legacy_fallback_bypasses_the_ambiguity_gate() {
        let view = view(vec![
            endpoint("production.old-a.generate", "llama", 1, None, Vec::new()),
            endpoint(
                "production.prefill-a.generate",
                "llama",
                2,
                Some(WorkerType::Prefill),
                Vec::new(),
            ),
            endpoint(
                "production.prefill-b.generate",
                "llama",
                3,
                Some(WorkerType::Prefill),
                Vec::new(),
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "llama");

        assert_eq!(topology.state, TopologyReadinessState::Ready);
        assert!(topology.legacy_fallback_active);
        assert_eq!(topology.duplicate_role_endpoints, [WorkerRole::Prefill]);
    }

    #[test]
    fn duplicate_encode_endpoints_are_ambiguous_like_the_core_evaluation() {
        let view = view(vec![
            endpoint(
                "production.encode-a.generate",
                "llama",
                1,
                Some(WorkerType::Encode),
                Vec::new(),
            ),
            endpoint(
                "production.encode-b.generate",
                "llama",
                2,
                Some(WorkerType::Encode),
                Vec::new(),
            ),
            endpoint(
                "production.prefill.generate",
                "llama",
                3,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.decode.generate",
                "llama",
                4,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "llama");

        assert_eq!(topology.state, TopologyReadinessState::Unavailable);
        assert_eq!(topology.duplicate_role_endpoints, [WorkerRole::Encode]);
    }

    #[test]
    fn duplicated_aggregated_endpoints_are_legal_scale_out() {
        let view = view(vec![
            endpoint(
                "production.backend-a.generate",
                "llama",
                1,
                Some(WorkerType::Aggregated),
                Vec::new(),
            ),
            endpoint(
                "production.backend-b.generate",
                "llama",
                2,
                Some(WorkerType::Aggregated),
                Vec::new(),
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();

        let entry = entry(&snapshot, "llama");
        assert_eq!(entry.state, TopologyReadinessState::Ready);
        assert_eq!(entry.members.len(), 2);
        assert!(entry.duplicate_role_endpoints.is_empty());
    }

    #[test]
    fn independent_pd_models_do_not_trigger_cross_model_ambiguity() {
        let view = view(vec![
            endpoint(
                "production.llama-prefill.generate",
                "llama",
                1,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.llama-decode.generate",
                "llama",
                2,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
            endpoint(
                "production.mistral-prefill.generate",
                "mistral",
                3,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.mistral-decode.generate",
                "mistral",
                4,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();

        assert_eq!(snapshot.entries.len(), 2);
        assert!(snapshot.entries.iter().all(|entry| {
            entry.state == TopologyReadinessState::Ready
                && entry.duplicate_role_endpoints.is_empty()
        }));
    }

    #[test]
    fn mixed_and_legacy_only_topologies_follow_core_fallback() {
        let mixed = view(vec![
            endpoint("production.legacy.generate", "llama", 1, None, Vec::new()),
            endpoint(
                "production.decode.generate",
                "llama",
                2,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let mixed_publisher = publisher(&mixed);
        mixed_publisher.replace_availability(
            EndpointId::from("production.legacy.generate"),
            1,
            Some(HashSet::from([1])),
        );
        mixed_publisher.replace_availability(
            EndpointId::from("production.decode.generate"),
            1,
            Some(HashSet::new()),
        );
        let snapshot = mixed_publisher.snapshot();
        let topology = entry(&snapshot, "llama");
        assert_eq!(topology.state, TopologyReadinessState::Ready);
        assert!(topology.legacy_fallback_active);
        assert!(topology.missing_roles.is_empty());

        let legacy = view(vec![endpoint(
            "production.legacy.generate",
            "mistral",
            3,
            None,
            Vec::new(),
        )]);
        let publisher = publisher(&legacy);
        publish_live(&publisher, &legacy);
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "mistral");
        assert_eq!(topology.state, TopologyReadinessState::Ready);
        assert!(topology.legacy_fallback_active);
    }

    #[test]
    fn epd_lora_excludes_encode_from_adapter_membership_but_not_base_readiness() {
        let encode = EndpointId::from("production.encoder.generate");
        let view = view(vec![
            endpoint(
                "production.encoder.generate",
                "vision-language",
                1,
                Some(WorkerType::Encode),
                vec![
                    vec![WorkerType::Prefill, WorkerType::Decode],
                    vec![WorkerType::Aggregated],
                ],
            ),
            with_adapter(
                endpoint(
                    "production.prefill.generate",
                    "vision-language",
                    2,
                    Some(WorkerType::Prefill),
                    vec![vec![WorkerType::Decode]],
                ),
                "vision-language",
                "tenant-a",
                2,
            ),
            with_adapter(
                endpoint(
                    "production.backend.generate",
                    "vision-language",
                    3,
                    Some(WorkerType::Decode),
                    vec![vec![WorkerType::Prefill]],
                ),
                "vision-language",
                "tenant-a",
                3,
            ),
        ]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "vision-language");
        assert_eq!(topology.adapters.len(), 1);
        assert_eq!(topology.adapters[0].state, TopologyReadinessState::Ready);
        assert!(topology.adapters[0].missing_roles.is_empty());

        publisher.replace_availability(encode, 1, Some(HashSet::new()));
        let snapshot = publisher.snapshot();
        let topology = entry(&snapshot, "vision-language");
        assert_eq!(topology.state, TopologyReadinessState::Unavailable);
        assert_eq!(
            topology.adapters[0].state,
            TopologyReadinessState::Unavailable
        );
    }

    #[test]
    fn unknown_is_limited_to_members_without_authoritative_availability() {
        let view = view(vec![
            endpoint(
                "production.prefill.generate",
                "llama",
                1,
                Some(WorkerType::Prefill),
                vec![vec![WorkerType::Decode]],
            ),
            endpoint(
                "production.decode.generate",
                "llama",
                2,
                Some(WorkerType::Decode),
                vec![vec![WorkerType::Prefill]],
            ),
        ]);
        let publisher = publisher(&view);
        let initial = publisher.snapshot();
        assert_eq!(
            entry(&initial, "llama").state,
            TopologyReadinessState::Unknown
        );

        publisher.replace_availability(
            EndpointId::from("production.prefill.generate"),
            1,
            Some(HashSet::from([1])),
        );
        let partial = publisher.snapshot();
        assert_eq!(
            entry(&partial, "llama").state,
            TopologyReadinessState::Unknown
        );

        publisher.replace_availability(
            EndpointId::from("production.decode.generate"),
            1,
            Some(HashSet::from([2])),
        );
        let ready = publisher.snapshot();
        assert_eq!(entry(&ready, "llama").state, TopologyReadinessState::Ready);
        let revision = ready.revision;
        publisher.replace_availability(
            EndpointId::from("production.decode.generate"),
            1,
            Some(HashSet::from([2])),
        );
        assert_eq!(publisher.snapshot().revision, revision);
    }

    #[test]
    fn retired_slot_cannot_overwrite_readded_endpoint_availability() {
        let endpoint_id = EndpointId::from("production.backend.generate");
        let old_view = view(vec![endpoint(
            "production.backend.generate",
            "llama",
            1,
            Some(WorkerType::Aggregated),
            Vec::new(),
        )]);
        let endpoint = endpoint_id;
        let publisher = publisher(&old_view);
        publisher.replace_availability(endpoint.clone(), 1, Some(HashSet::from([1])));
        assert_eq!(
            entry(&publisher.snapshot(), "llama").state,
            TopologyReadinessState::Ready
        );

        publisher.replace_membership(DcMembershipView::default());
        let mut replacement = old_view.endpoints[&endpoint].clone();
        replacement.generation = 2;
        publisher.replace_membership(view(vec![replacement]));
        publisher.claim_availability(endpoint.clone(), 2);
        publisher.replace_availability(endpoint.clone(), 2, Some(HashSet::new()));
        let replacement_snapshot = publisher.snapshot();
        assert_eq!(
            entry(&replacement_snapshot, "llama").state,
            TopologyReadinessState::Unavailable
        );

        publisher.replace_availability(endpoint, 1, Some(HashSet::from([1])));
        let after_zombie_write = publisher.snapshot();
        assert_eq!(after_zombie_write.revision, replacement_snapshot.revision);
        assert_eq!(
            entry(&after_zombie_write, "llama").state,
            TopologyReadinessState::Unavailable
        );
    }

    #[test]
    fn membership_update_keeps_availability_owned_by_the_same_slot() {
        let endpoint_id = EndpointId::from("production.backend.generate");
        let old_view = view(vec![endpoint(
            "production.backend.generate",
            "llama",
            1,
            Some(WorkerType::Aggregated),
            Vec::new(),
        )]);
        let endpoint = endpoint_id;
        let publisher = publisher(&old_view);
        publisher.replace_availability(endpoint.clone(), 1, Some(HashSet::from([1])));
        let before = publisher.snapshot();
        assert_eq!(entry(&before, "llama").state, TopologyReadinessState::Ready);

        let mut updated = old_view.endpoints[&endpoint].clone();
        updated.generation = 2;
        publisher.replace_membership(view(vec![updated]));
        let after = publisher.snapshot();

        assert_eq!(after.revision, before.revision);
        assert_eq!(entry(&after, "llama").state, TopologyReadinessState::Ready);
    }

    #[test]
    fn producer_generation_swap_does_not_churn_stable_topology_links() {
        let view = view(vec![endpoint(
            "production.backend.generate",
            "llama",
            1,
            Some(WorkerType::Aggregated),
            Vec::new(),
        )]);
        let publisher = publisher(&view);
        publish_live(&publisher, &view);
        let before = publisher.snapshot();
        let pool_id = before.entries[0].members[0].pool_id;

        publisher.replace_catalog(&catalog(&view, 2));
        let after = publisher.snapshot();
        assert_eq!(after.revision, before.revision);
        assert_eq!(after.entries[0].members[0].pool_id, pool_id);
    }
}
