// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::identity::IndexerDomainId;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    ModelCardInstanceId,
};
use dynamo_runtime::protocols::EndpointId;
use futures::{Stream, StreamExt, future::try_join_all};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::identity::{CanonicalModelId, CanonicalModelRegistration, ModelAlias, ModelTarget};
use super::resolution::{ResolvedIndexerDomain, resolve_indexer_domain};
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;
use crate::worker_type::WorkerType;

const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);
const KV_EVENT_HASH_FORMAT_VERSION: u16 = 1;

pub(crate) type KvCacheDomainKey = ResolvedIndexerDomain;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvDcRelayDiscoveryConfig {
    pub namespaces: Vec<String>,
    pub endpoint_prefixes: Vec<String>,
    pub watch_all: bool,
}

impl KvDcRelayDiscoveryConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.watch_all || !self.namespaces.is_empty(),
            "KV DC Relay requires at least one discovery namespace or explicit watch_all"
        );
        anyhow::ensure!(
            !self.watch_all || self.namespaces.is_empty(),
            "KV DC Relay watch_all cannot be combined with explicit discovery namespaces"
        );

        let mut unique_namespaces = HashSet::new();
        for namespace in &self.namespaces {
            anyhow::ensure!(
                !namespace.trim().is_empty(),
                "KV DC Relay discovery namespaces must not be empty"
            );
            anyhow::ensure!(
                namespace.trim() == namespace,
                "KV DC Relay discovery namespaces must not contain surrounding whitespace"
            );
            anyhow::ensure!(
                unique_namespaces.insert(namespace),
                "duplicate KV DC Relay discovery namespace: {namespace}"
            );
        }

        let mut unique_prefixes = HashSet::new();
        for prefix in &self.endpoint_prefixes {
            anyhow::ensure!(
                !prefix.trim().is_empty(),
                "KV DC Relay endpoint prefixes must not be empty"
            );
            anyhow::ensure!(
                prefix.trim() == prefix,
                "KV DC Relay endpoint prefixes must not contain surrounding whitespace"
            );
            anyhow::ensure!(
                unique_prefixes.insert(prefix),
                "duplicate KV DC Relay endpoint prefix: {prefix}"
            );
            anyhow::ensure!(
                self.watch_all
                    || self.namespaces.iter().any(|namespace| {
                        prefix == namespace
                            || prefix
                                .strip_prefix(namespace)
                                .is_some_and(|suffix| suffix.starts_with('.'))
                    }),
                "KV DC Relay endpoint prefix {prefix} is outside the configured namespaces"
            );
        }
        Ok(())
    }

    fn queries(&self) -> Vec<DiscoveryQuery> {
        if self.watch_all {
            vec![DiscoveryQuery::AllModels]
        } else {
            self.namespaces
                .iter()
                .map(|namespace| DiscoveryQuery::NamespacedModels {
                    namespace: namespace.clone(),
                })
                .collect()
        }
    }

    fn filter(&self) -> DcDiscoveryFilter {
        DcDiscoveryFilter {
            endpoint_prefixes: self.endpoint_prefixes.clone(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DcDiscoveryFilter {
    endpoint_prefixes: Vec<String>,
}

impl DcDiscoveryFilter {
    fn matches(&self, endpoint: &EndpointId) -> bool {
        if self.endpoint_prefixes.is_empty() {
            return true;
        }
        self.endpoint_prefixes
            .iter()
            .any(|prefix| endpoint_matches_prefix(endpoint, prefix))
    }
}

fn endpoint_matches_prefix(endpoint: &EndpointId, prefix: &str) -> bool {
    let mut parts = prefix.split('.');
    for expected in [
        endpoint.namespace.as_str(),
        endpoint.component.as_str(),
        endpoint.name.as_str(),
    ] {
        match parts.next() {
            None => return true,
            Some(actual) if actual == expected => {}
            Some(_) => return false,
        }
    }
    parts.next().is_none()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DomainSlotId {
    pub(crate) endpoint: EndpointId,
    pub(crate) indexer_domain: IndexerDomainId,
}

impl DomainSlotId {
    fn new(endpoint: EndpointId, indexer_domain: IndexerDomainId) -> Self {
        Self {
            endpoint,
            indexer_domain,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MembershipConflict {
    Endpoint {
        endpoint: EndpointId,
        reason: String,
    },
    Card {
        card: ModelCardInstanceId,
        reason: String,
    },
    Worker {
        worker_id: WorkerId,
        reason: String,
    },
    Binding {
        model: CanonicalModelId,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DomainWorkerTopology {
    pub(crate) worker_type: Option<WorkerType>,
    pub(crate) needs: Vec<Vec<WorkerType>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdapterWorkerMembership {
    pub(crate) max_gpu_lora_count: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdapterMembership {
    pub(crate) base_model: CanonicalModelId,
    pub(crate) workers: HashMap<WorkerId, AdapterWorkerMembership>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DomainMembership {
    pub(crate) slot_id: DomainSlotId,
    pub(crate) generation: u64,
    pub(crate) compatibility_conflict: bool,
    pub(crate) domain: KvCacheDomainKey,
    pub(crate) namespace: String,
    pub(crate) registrations: Vec<CanonicalModelRegistration>,
    pub(crate) models: Vec<String>,
    pub(crate) aliases: Vec<String>,
    pub(crate) roles: Vec<String>,
    pub(crate) runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
    pub(crate) worker_topology: HashMap<WorkerId, DomainWorkerTopology>,
    pub(crate) adapters: HashMap<CanonicalModelId, AdapterMembership>,
    pub(crate) conflicts: Vec<MembershipConflict>,
}

impl DomainMembership {
    pub(crate) fn is_materializable(&self) -> bool {
        !self.compatibility_conflict && !self.registrations.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct DcMembershipView {
    pub(crate) domains: Arc<HashMap<DomainSlotId, DomainMembership>>,
}

#[derive(Debug, Clone)]
struct ProjectedBaseCard<'a> {
    id: &'a ModelCardInstanceId,
    card: &'a ModelDeploymentCard,
    domain: KvCacheDomainKey,
    model: Option<CanonicalModelId>,
    aliases: Vec<ModelAlias>,
}

#[derive(Debug, Clone, PartialEq)]
struct WorkerFacts {
    runtime_config: ModelRuntimeConfig,
    worker_type: Option<WorkerType>,
    needs: Vec<Vec<WorkerType>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BindingIdentity {
    model: CanonicalModelId,
    target: ModelTarget,
}

#[derive(Debug, Clone)]
struct RegistrationClaim {
    binding: BindingIdentity,
    aliases: Vec<ModelAlias>,
}

struct DomainMembershipBuilder {
    domain: KvCacheDomainKey,
    claims: Vec<RegistrationClaim>,
    runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
    worker_topology: HashMap<WorkerId, DomainWorkerTopology>,
    adapters: HashMap<CanonicalModelId, AdapterMembership>,
    conflicts: Vec<MembershipConflict>,
}

impl DomainMembershipBuilder {
    fn new(domain: KvCacheDomainKey) -> Self {
        Self {
            domain,
            claims: Vec::new(),
            runtime_configs: HashMap::new(),
            worker_topology: HashMap::new(),
            adapters: HashMap::new(),
            conflicts: Vec::new(),
        }
    }
}

pub(crate) struct DcMembershipWatch {
    receiver: watch::Receiver<DcMembershipView>,
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

impl DcMembershipWatch {
    pub(crate) async fn start(
        discovery: Arc<dyn Discovery>,
        config: KvDcRelayDiscoveryConfig,
        parent_cancel: CancellationToken,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let queries = config.queries();
        let filter = config.filter();
        let initial = list_queries(&discovery, &queries).await?;
        let mut state = MembershipState::default();
        state.replace_all(initial, &filter);
        let (sender, receiver) = watch::channel(state.view(&filter));
        let cancel = parent_cancel.child_token();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            run_membership_watch(discovery, queries, filter, state, sender, task_cancel).await;
        });
        Ok(Self {
            receiver,
            cancel,
            task,
        })
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<DcMembershipView> {
        self.receiver.clone()
    }

    pub(crate) async fn shutdown(self) {
        self.cancel.cancel();
        if let Err(error) = self.task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV DC Relay model-card watch failed during shutdown");
        }
    }
}

struct MembershipState {
    cards: HashMap<ModelCardInstanceId, ModelDeploymentCard>,
    next_domain_generation: u64,
    previous: Arc<HashMap<DomainSlotId, DomainMembership>>,
    warned_invalid_models: HashSet<ModelCardInstanceId>,
    warned_invalid_aliases: HashSet<(ModelCardInstanceId, String)>,
    warned_orphan_adapters: HashSet<ModelCardInstanceId>,
}

impl Default for MembershipState {
    fn default() -> Self {
        Self {
            cards: HashMap::new(),
            next_domain_generation: 1,
            previous: Arc::default(),
            warned_invalid_models: HashSet::new(),
            warned_invalid_aliases: HashSet::new(),
            warned_orphan_adapters: HashSet::new(),
        }
    }
}

impl MembershipState {
    fn replace_all(&mut self, instances: Vec<DiscoveryInstance>, filter: &DcDiscoveryFilter) {
        let mut next = HashMap::new();
        for instance in instances {
            let Some((id, card)) = decode_card(instance) else {
                continue;
            };
            if filter.matches(&endpoint_id(&id)) {
                next.insert(id, card);
            }
        }
        self.cards = next;
    }

    fn apply(&mut self, event: DiscoveryEvent, filter: &DcDiscoveryFilter) {
        match event {
            DiscoveryEvent::Added(instance) => {
                let Some((id, card)) = decode_card(instance) else {
                    return;
                };
                if filter.matches(&endpoint_id(&id)) {
                    self.cards.insert(id, card);
                }
            }
            DiscoveryEvent::Removed(DiscoveryInstanceId::Model(id)) => {
                self.cards.remove(&id);
            }
            DiscoveryEvent::Removed(_) => {}
        }
    }

    fn view(&mut self, filter: &DcDiscoveryFilter) -> DcMembershipView {
        let mut invalid_models = HashSet::new();
        let mut invalid_aliases = HashSet::new();
        let mut orphan_adapters = HashSet::new();
        let mut incompatible_endpoints = HashSet::new();
        let mut grouped: HashMap<EndpointId, Vec<(&ModelCardInstanceId, &ModelDeploymentCard)>> =
            HashMap::new();
        for (id, card) in &self.cards {
            let endpoint = endpoint_id(id);
            if filter.matches(&endpoint) {
                grouped.entry(endpoint).or_default().push((id, card));
            }
        }

        let mut builders = HashMap::<DomainSlotId, DomainMembershipBuilder>::new();
        for (endpoint, cards) in grouped {
            let mut base_cards = Vec::new();
            let mut adapter_cards = Vec::new();
            for (id, card) in cards {
                if id.model_suffix.is_some() || card.lora.is_some() {
                    adapter_cards.push((id, card));
                    continue;
                }
                let domain = resolve_indexer_domain(card, &endpoint, KV_EVENT_HASH_FORMAT_VERSION);
                let slot_id = DomainSlotId::new(endpoint.clone(), domain.id);
                builders
                    .entry(slot_id)
                    .or_insert_with(|| DomainMembershipBuilder::new(domain.clone()));

                let model = match CanonicalModelId::new(card.name().to_string()) {
                    Ok(model) => Some(model),
                    Err(error) => {
                        invalid_models.insert(id.clone());
                        if !self.warned_invalid_models.contains(id) {
                            tracing::warn!(
                                endpoint = %endpoint,
                                model = card.name(),
                                %error,
                                "Ignoring model card with invalid canonical model identity"
                            );
                        }
                        None
                    }
                };
                let aliases = model
                    .as_ref()
                    .map(|model| {
                        valid_aliases(
                            id,
                            card,
                            model,
                            &endpoint,
                            &mut invalid_aliases,
                            &self.warned_invalid_aliases,
                        )
                    })
                    .unwrap_or_default();
                base_cards.push(ProjectedBaseCard {
                    id,
                    card,
                    domain,
                    model,
                    aliases,
                });
            }

            let endpoint_domains: HashSet<_> = base_cards
                .iter()
                .map(|projection| projection.domain.clone())
                .collect();
            if endpoint_domains.len() > 1 {
                incompatible_endpoints.insert(endpoint.clone());
            }

            let mut worker_domains = HashMap::<WorkerId, HashSet<KvCacheDomainKey>>::new();
            let mut worker_facts = HashMap::<WorkerId, WorkerFacts>::new();
            let mut conflicting_worker_facts = HashSet::new();
            for projection in &base_cards {
                worker_domains
                    .entry(projection.id.instance_id)
                    .or_default()
                    .insert(projection.domain.clone());
                let facts = WorkerFacts {
                    runtime_config: projection.card.runtime_config.clone(),
                    worker_type: projection.card.worker_type,
                    needs: projection.card.needs.clone(),
                };
                if worker_facts
                    .insert(projection.id.instance_id, facts.clone())
                    .is_some_and(|previous| previous != facts)
                {
                    conflicting_worker_facts.insert(projection.id.instance_id);
                }
            }
            let ambiguous_workers: HashSet<_> = worker_domains
                .iter()
                .filter_map(|(&worker_id, domains)| (domains.len() > 1).then_some(worker_id))
                .collect();

            for projection in &base_cards {
                let slot_id = DomainSlotId::new(endpoint.clone(), projection.domain.id);
                let Some(builder) = builders.get_mut(&slot_id) else {
                    tracing::error!(
                        endpoint = %endpoint,
                        worker_id = projection.id.instance_id,
                        "KV DC Relay lost a projected base-model domain while building membership"
                    );
                    continue;
                };
                let worker_id = projection.id.instance_id;
                if ambiguous_workers.contains(&worker_id) {
                    builder.conflicts.push(MembershipConflict::Worker {
                        worker_id,
                        reason: "worker resolves to multiple indexer domains".to_string(),
                    });
                    continue;
                }
                if conflicting_worker_facts.contains(&worker_id) {
                    builder.conflicts.push(MembershipConflict::Worker {
                        worker_id,
                        reason: "worker publishes conflicting runtime, role, or needs facts"
                            .to_string(),
                    });
                    continue;
                }
                let Some(model) = projection.model.clone() else {
                    builder.conflicts.push(MembershipConflict::Card {
                        card: projection.id.clone(),
                        reason: "invalid canonical model identity".to_string(),
                    });
                    continue;
                };
                builder
                    .runtime_configs
                    .insert(worker_id, projection.card.runtime_config.clone());
                builder.worker_topology.insert(
                    worker_id,
                    DomainWorkerTopology {
                        worker_type: projection.card.worker_type,
                        needs: projection.card.needs.clone(),
                    },
                );
                builder.claims.push(RegistrationClaim {
                    binding: BindingIdentity {
                        model: model.clone(),
                        target: ModelTarget::Base { base_model: model },
                    },
                    aliases: projection.aliases.clone(),
                });
            }

            for (id, card) in adapter_cards {
                let worker_id = id.instance_id;
                let worker_bases: Vec<_> = base_cards
                    .iter()
                    .filter(|projection| projection.id.instance_id == worker_id)
                    .collect();
                if worker_bases.is_empty() {
                    orphan_adapters.insert(id.clone());
                    if !self.warned_orphan_adapters.contains(id) {
                        tracing::warn!(
                            endpoint = %endpoint,
                            worker_id,
                            card = %id.to_path(),
                            "Ignoring adapter card without a backing base model card"
                        );
                    }
                    continue;
                }
                let worker_domain_ids: HashSet<_> = worker_bases
                    .iter()
                    .map(|projection| projection.domain.id)
                    .collect();
                if worker_domain_ids.len() != 1
                    || ambiguous_workers.contains(&worker_id)
                    || conflicting_worker_facts.contains(&worker_id)
                {
                    for domain_id in worker_domain_ids {
                        if let Some(builder) =
                            builders.get_mut(&DomainSlotId::new(endpoint.clone(), domain_id))
                        {
                            builder.conflicts.push(MembershipConflict::Card {
                                card: id.clone(),
                                reason: "adapter worker has no unambiguous backing base domain"
                                    .to_string(),
                            });
                        }
                    }
                    continue;
                }
                let Some(&domain_id) = worker_domain_ids.iter().next() else {
                    continue;
                };
                let Some(builder) =
                    builders.get_mut(&DomainSlotId::new(endpoint.clone(), domain_id))
                else {
                    tracing::error!(
                        endpoint = %endpoint,
                        worker_id,
                        card = %id.to_path(),
                        "KV DC Relay lost an adapter's backing domain while building membership"
                    );
                    continue;
                };
                let base_models: HashSet<_> = worker_bases
                    .iter()
                    .filter_map(|projection| projection.model.clone())
                    .collect();
                if base_models.len() != 1 {
                    builder.conflicts.push(MembershipConflict::Card {
                        card: id.clone(),
                        reason: "adapter does not resolve to exactly one backing base model"
                            .to_string(),
                    });
                    continue;
                }
                let Some(base_model) = base_models.into_iter().next() else {
                    continue;
                };
                let adapter_name = card
                    .lora
                    .as_ref()
                    .map(|lora| lora.name.as_str())
                    .or(id.model_suffix.as_deref());
                let Some(adapter_name) = adapter_name else {
                    builder.conflicts.push(MembershipConflict::Card {
                        card: id.clone(),
                        reason: "adapter card has no adapter identity".to_string(),
                    });
                    continue;
                };
                let adapter = match CanonicalModelId::new(adapter_name.to_string()) {
                    Ok(adapter) => adapter,
                    Err(error) => {
                        invalid_models.insert(id.clone());
                        if !self.warned_invalid_models.contains(id) {
                            tracing::warn!(
                                endpoint = %endpoint,
                                model = adapter_name,
                                %error,
                                "Ignoring adapter card with invalid canonical model identity"
                            );
                        }
                        builder.conflicts.push(MembershipConflict::Card {
                            card: id.clone(),
                            reason: "invalid adapter model identity".to_string(),
                        });
                        continue;
                    }
                };
                let aliases = valid_aliases(
                    id,
                    card,
                    &adapter,
                    &endpoint,
                    &mut invalid_aliases,
                    &self.warned_invalid_aliases,
                );
                let target = ModelTarget::Lora {
                    base_model: base_model.clone(),
                    adapter: adapter.clone(),
                };
                builder.claims.push(RegistrationClaim {
                    binding: BindingIdentity {
                        model: adapter.clone(),
                        target,
                    },
                    aliases,
                });
                let capacity = card.lora.as_ref().and_then(|lora| lora.max_gpu_lora_count);
                match builder.adapters.entry(adapter) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(AdapterMembership {
                            base_model,
                            workers: HashMap::from([(
                                worker_id,
                                AdapterWorkerMembership {
                                    max_gpu_lora_count: capacity,
                                },
                            )]),
                        });
                    }
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        if entry.get().base_model != base_model {
                            builder.conflicts.push(MembershipConflict::Card {
                                card: id.clone(),
                                reason: "adapter identity resolves to conflicting base models"
                                    .to_string(),
                            });
                            continue;
                        }
                        let workers = &mut entry.get_mut().workers;
                        let facts = AdapterWorkerMembership {
                            max_gpu_lora_count: capacity,
                        };
                        if workers
                            .insert(worker_id, facts.clone())
                            .is_some_and(|previous| previous != facts)
                        {
                            workers.remove(&worker_id);
                            builder.conflicts.push(MembershipConflict::Worker {
                                worker_id,
                                reason: "worker publishes conflicting adapter capacity".to_string(),
                            });
                        }
                    }
                }
            }
        }

        let mut lookup_owners = HashMap::<String, HashSet<BindingIdentity>>::new();
        for builder in builders.values() {
            for claim in &builder.claims {
                lookup_owners
                    .entry(claim.binding.model.as_str().to_string())
                    .or_default()
                    .insert(claim.binding.clone());
                for alias in &claim.aliases {
                    lookup_owners
                        .entry(alias.as_str().to_string())
                        .or_default()
                        .insert(claim.binding.clone());
                }
            }
        }
        let ambiguous_names: HashSet<_> = lookup_owners
            .into_iter()
            .filter_map(|(name, owners)| (owners.len() > 1).then_some(name))
            .collect();

        let mut domains = HashMap::new();
        for (slot_id, mut builder) in builders {
            let compatibility_conflict = incompatible_endpoints.contains(&slot_id.endpoint);
            if compatibility_conflict {
                builder.conflicts.push(MembershipConflict::Endpoint {
                    endpoint: slot_id.endpoint.clone(),
                    reason: "endpoint resolves to multiple indexer domains".to_string(),
                });
            }
            let mut grouped_claims = HashMap::<BindingIdentity, HashSet<ModelAlias>>::new();
            for claim in builder.claims {
                if ambiguous_names.contains(claim.binding.model.as_str()) {
                    builder.conflicts.push(MembershipConflict::Binding {
                        model: claim.binding.model,
                        reason: "request-facing model name resolves to conflicting targets"
                            .to_string(),
                    });
                    continue;
                }
                let aliases = grouped_claims.entry(claim.binding).or_default();
                aliases.extend(
                    claim
                        .aliases
                        .into_iter()
                        .filter(|alias| !ambiguous_names.contains(alias.as_str())),
                );
            }
            let mut registrations: Vec<_> = grouped_claims
                .into_iter()
                .map(|(binding, aliases)| {
                    CanonicalModelRegistration::with_target(
                        binding.model,
                        binding.target,
                        aliases.into_iter().collect(),
                    )
                })
                .collect();
            registrations.sort_unstable();
            let models = registrations
                .iter()
                .map(|registration| registration.model().as_str().to_string())
                .collect::<HashSet<_>>();
            let aliases = registrations
                .iter()
                .flat_map(|registration| registration.aliases())
                .map(|alias| alias.as_str().to_string())
                .collect::<HashSet<_>>();
            let roles = builder
                .worker_topology
                .values()
                .filter_map(|topology| topology.worker_type)
                .map(|worker_type| worker_type.as_str().to_string())
                .collect::<HashSet<_>>();
            builder
                .conflicts
                .sort_by(|left, right| format!("{left:?}").cmp(&format!("{right:?}")));
            builder.conflicts.dedup();
            let mut candidate = DomainMembership {
                slot_id: slot_id.clone(),
                generation: 0,
                compatibility_conflict,
                domain: builder.domain,
                namespace: slot_id.endpoint.namespace.clone(),
                registrations,
                models: sorted(models),
                aliases: sorted(aliases),
                roles: sorted(roles),
                runtime_configs: builder.runtime_configs,
                worker_topology: builder.worker_topology,
                adapters: builder.adapters,
                conflicts: builder.conflicts,
            };
            candidate.generation = match self.previous.get(&slot_id) {
                Some(previous) if same_membership(previous, &candidate) => previous.generation,
                _ => {
                    let generation = self.next_domain_generation;
                    self.next_domain_generation = generation.saturating_add(1);
                    generation
                }
            };
            domains.insert(slot_id, candidate);
        }

        self.warned_invalid_models = invalid_models;
        self.warned_invalid_aliases = invalid_aliases;
        self.warned_orphan_adapters = orphan_adapters;
        let domains = Arc::new(domains);
        self.previous = domains.clone();
        DcMembershipView { domains }
    }
}

async fn run_membership_watch(
    discovery: Arc<dyn Discovery>,
    queries: Vec<DiscoveryQuery>,
    filter: DcDiscoveryFilter,
    mut state: MembershipState,
    sender: watch::Sender<DcMembershipView>,
    cancel: CancellationToken,
) {
    let mut retry_delay = Duration::from_millis(100);
    let mut watch_failures = 0u64;
    let mut reconcile_failures = 0u64;
    loop {
        let stream_cancel = cancel.child_token();
        let streams = open_query_streams(&discovery, &queries, &stream_cancel).await;
        let mut stream = match streams {
            Ok(stream) => stream,
            Err(error) => {
                stream_cancel.cancel();
                watch_failures = watch_failures.saturating_add(1);
                if watch_failures == 1 {
                    tracing::error!(
                        %error,
                        query_count = queries.len(),
                        "Failed to watch scoped KV DC Relay model-card membership"
                    );
                } else {
                    tracing::debug!(
                        %error, watch_failures, query_count = queries.len(),
                        retry_ms = retry_delay.as_millis(),
                        "Scoped KV DC Relay model-card watch retry failed"
                    );
                }
                if !retry_or_cancel(retry_delay, &cancel).await {
                    return;
                }
                retry_delay = (retry_delay * 2).min(Duration::from_secs(5));
                continue;
            }
        };
        let mut reconcile = tokio::time::interval(RECONCILE_INTERVAL);
        reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                event = stream.next() => match event {
                    Some(Ok(Some(event))) => {
                        watch_failures = 0;
                        retry_delay = Duration::from_millis(100);
                        state.apply(event, &filter);
                        publish_membership_if_changed(&sender, state.view(&filter));
                    }
                    Some(Err(error)) => {
                        watch_failures = watch_failures.saturating_add(1);
                        if watch_failures == 1 {
                            tracing::error!(%error, "Scoped KV DC Relay model-card discovery stream failed; rebinding");
                        } else {
                            tracing::debug!(
                                %error, watch_failures, retry_ms = retry_delay.as_millis(),
                                "Scoped KV DC Relay model-card discovery stream failed again; rebinding"
                            );
                        }
                        break;
                    }
                    Some(Ok(None)) | None => {
                        watch_failures = watch_failures.saturating_add(1);
                        if watch_failures == 1 {
                            tracing::error!("Scoped KV DC Relay model-card discovery stream closed; rebinding");
                        } else {
                            tracing::debug!(
                                watch_failures, retry_ms = retry_delay.as_millis(),
                                "Scoped KV DC Relay model-card discovery stream closed again; rebinding"
                            );
                        }
                        break;
                    }
                },
                _ = reconcile.tick() => match list_queries(&discovery, &queries).await {
                    Ok(instances) => {
                        watch_failures = 0;
                        reconcile_failures = 0;
                        retry_delay = Duration::from_millis(100);
                        state.replace_all(instances, &filter);
                        publish_membership_if_changed(&sender, state.view(&filter));
                    }
                    Err(error) => {
                        reconcile_failures = reconcile_failures.saturating_add(1);
                        if reconcile_failures == 1 {
                            tracing::warn!(%error, "Failed periodic KV DC Relay membership reconciliation");
                        } else {
                            tracing::debug!(
                                %error, reconcile_failures,
                                "Periodic KV DC Relay membership reconciliation failed again"
                            );
                        }
                    }
                },
            }
        }
        stream_cancel.cancel();
        if !retry_or_cancel(retry_delay, &cancel).await {
            return;
        }
        retry_delay = (retry_delay * 2).min(Duration::from_secs(5));
    }
}

fn publish_membership_if_changed(sender: &watch::Sender<DcMembershipView>, next: DcMembershipView) {
    sender.send_if_modified(move |current| {
        if current == &next {
            return false;
        }
        *current = next;
        true
    });
}

async fn list_queries(
    discovery: &Arc<dyn Discovery>,
    queries: &[DiscoveryQuery],
) -> anyhow::Result<Vec<DiscoveryInstance>> {
    let results = try_join_all(queries.iter().cloned().map(|query| discovery.list(query))).await?;
    Ok(results.into_iter().flatten().collect())
}

type RebindingDiscoveryStream =
    Pin<Box<dyn Stream<Item = anyhow::Result<Option<DiscoveryEvent>>> + Send>>;

async fn open_query_streams(
    discovery: &Arc<dyn Discovery>,
    queries: &[DiscoveryQuery],
    cancel: &CancellationToken,
) -> anyhow::Result<futures::stream::SelectAll<RebindingDiscoveryStream>> {
    let opened = try_join_all(
        queries
            .iter()
            .cloned()
            .map(|query| discovery.list_and_watch(query, Some(cancel.clone()))),
    )
    .await?;
    let mut streams = futures::stream::SelectAll::new();
    for stream in opened {
        let stream = stream
            .map(|event| event.map(Some))
            .chain(futures::stream::once(async { Ok(None) }));
        streams.push(Box::pin(stream) as RebindingDiscoveryStream);
    }
    Ok(streams)
}

async fn retry_or_cancel(delay: Duration, cancel: &CancellationToken) -> bool {
    tokio::select! {
        _ = cancel.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}

fn decode_card(instance: DiscoveryInstance) -> Option<(ModelCardInstanceId, ModelDeploymentCard)> {
    let DiscoveryInstanceId::Model(id) = instance.id() else {
        return None;
    };
    match instance.deserialize_model::<ModelDeploymentCard>() {
        Ok(card) => Some((id, card)),
        Err(error) => {
            tracing::warn!(instance = %id.to_path(), %error, "Ignoring malformed KV DC Relay model card");
            None
        }
    }
}

fn endpoint_id(id: &ModelCardInstanceId) -> EndpointId {
    EndpointId {
        namespace: id.namespace.clone(),
        component: id.component.clone(),
        name: id.endpoint.clone(),
    }
}

fn sorted(values: HashSet<String>) -> Vec<String> {
    let mut values: Vec<_> = values.into_iter().collect();
    values.sort_unstable();
    values
}

fn valid_aliases(
    id: &ModelCardInstanceId,
    card: &ModelDeploymentCard,
    model: &CanonicalModelId,
    endpoint: &EndpointId,
    invalid_aliases: &mut HashSet<(ModelCardInstanceId, String)>,
    warned_invalid_aliases: &HashSet<(ModelCardInstanceId, String)>,
) -> Vec<ModelAlias> {
    let mut aliases = HashSet::new();
    for alias in &card.aliases {
        match ModelAlias::new(alias.clone()) {
            Ok(alias) if alias.as_str() != model.as_str() => {
                aliases.insert(alias);
            }
            Ok(_) => {}
            Err(error) => {
                let key = (id.clone(), alias.clone());
                invalid_aliases.insert(key.clone());
                if !warned_invalid_aliases.contains(&key) {
                    tracing::warn!(
                        endpoint = %endpoint,
                        model = %model,
                        alias,
                        %error,
                        "Ignoring invalid model alias"
                    );
                }
            }
        }
    }
    let mut aliases: Vec<_> = aliases.into_iter().collect();
    aliases.sort_unstable();
    aliases
}

fn same_membership(left: &DomainMembership, right: &DomainMembership) -> bool {
    left.slot_id == right.slot_id
        && left.domain == right.domain
        && left.compatibility_conflict == right.compatibility_conflict
        && left.namespace == right.namespace
        && left.registrations == right.registrations
        && left.models == right.models
        && left.aliases == right.aliases
        && left.roles == right.roles
        && left.runtime_configs == right.runtime_configs
        && left.worker_topology == right.worker_topology
        && left.adapters == right.adapters
        && left.conflicts == right.conflicts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::LoraInfo;
    use crate::worker_type::WorkerType;

    fn card(name: &str, artifact: &str, block_size: u32) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(name);
        card.source_path = Some(artifact.to_string());
        card.kv_cache_block_size = block_size;
        card.worker_type = Some(WorkerType::Aggregated);
        card
    }

    #[test]
    fn discovery_scope_requires_explicit_namespaces_or_watch_all() {
        let config = KvDcRelayDiscoveryConfig::default();
        assert!(config.validate().is_err());

        let config = KvDcRelayDiscoveryConfig {
            watch_all: true,
            ..Default::default()
        };
        assert_eq!(config.queries(), vec![DiscoveryQuery::AllModels]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn discovery_scope_uses_one_server_side_query_per_namespace() {
        let config = KvDcRelayDiscoveryConfig {
            namespaces: vec!["prod-a".into(), "prod-b".into()],
            endpoint_prefixes: vec!["prod-a.backend".into()],
            watch_all: false,
        };
        assert!(config.validate().is_ok());
        assert_eq!(
            config.queries(),
            vec![
                DiscoveryQuery::NamespacedModels {
                    namespace: "prod-a".into()
                },
                DiscoveryQuery::NamespacedModels {
                    namespace: "prod-b".into()
                },
            ]
        );
        let filter = config.filter();
        assert!(filter.matches(&EndpointId::from("prod-a.backend.generate")));
        assert!(!filter.matches(&EndpointId::from("prod-a.backend2.generate")));
        assert!(!filter.matches(&EndpointId::from("prod-b.backend.generate")));
    }

    #[test]
    fn discovery_scope_rejects_prefix_outside_assigned_namespaces() {
        let config = KvDcRelayDiscoveryConfig {
            namespaces: vec!["prod-a".into()],
            endpoint_prefixes: vec!["prod-b.backend".into()],
            watch_all: false,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn discovery_scope_rejects_surrounding_whitespace() {
        let padded_namespace = KvDcRelayDiscoveryConfig {
            namespaces: vec![" prod-a".into()],
            endpoint_prefixes: Vec::new(),
            watch_all: false,
        };
        assert!(padded_namespace.validate().is_err());

        let padded_prefix = KvDcRelayDiscoveryConfig {
            namespaces: vec!["prod-a".into()],
            endpoint_prefixes: vec!["prod-a.backend ".into()],
            watch_all: false,
        };
        assert!(padded_prefix.validate().is_err());
    }

    #[test]
    fn unchanged_membership_does_not_advance_the_watch_version() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                1,
                None,
                card("llama", "meta/llama", 64),
            )),
            &filter,
        );
        let initial = state.view(&filter);
        let (sender, mut receiver) = watch::channel(initial.clone());

        publish_membership_if_changed(&sender, state.view(&filter));
        assert!(!receiver.has_changed().unwrap());

        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                2,
                None,
                card("chat", "meta/llama", 64),
            )),
            &filter,
        );
        publish_membership_if_changed(&sender, state.view(&filter));
        assert!(receiver.has_changed().unwrap());
        assert_ne!(*receiver.borrow_and_update(), initial);
    }

    #[test]
    fn reappearing_domain_advances_its_generation() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let discovery_instance = instance("generate", 1, None, card("llama", "meta/llama", 64));
        let instance_id = DiscoveryInstanceId::Model(ModelCardInstanceId {
            namespace: "prod".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 1,
            model_suffix: None,
        });

        state.apply(DiscoveryEvent::Added(discovery_instance.clone()), &filter);
        let initial = state.view(&filter);
        assert_eq!(initial.domains.len(), 1);
        assert_eq!(initial.domains.values().next().unwrap().generation, 1);

        state.apply(DiscoveryEvent::Removed(instance_id), &filter);
        assert!(state.view(&filter).domains.is_empty());

        state.apply(DiscoveryEvent::Added(discovery_instance), &filter);
        let reappeared = state.view(&filter);
        assert_eq!(reappeared.domains.len(), 1);
        assert_eq!(reappeared.domains.values().next().unwrap().generation, 2);
    }

    fn instance(
        endpoint: &str,
        instance_id: u64,
        model_suffix: Option<&str>,
        card: ModelDeploymentCard,
    ) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: "prod".to_string(),
            component: "backend".to_string(),
            endpoint: endpoint.to_string(),
            instance_id,
            card_json: serde_json::to_value(card).unwrap(),
            model_suffix: model_suffix.map(str::to_string),
        }
    }

    fn domains_for_endpoint<'a>(
        view: &'a DcMembershipView,
        endpoint: &EndpointId,
    ) -> Vec<&'a DomainMembership> {
        view.domains
            .iter()
            .filter(|(slot_id, _)| &slot_id.endpoint == endpoint)
            .map(|(_, membership)| membership)
            .collect()
    }

    #[test]
    fn incompatible_domains_under_one_endpoint_are_fenced_together() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                1,
                None,
                card("llama", "meta/llama", 64),
            )),
            &filter,
        );
        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                2,
                None,
                card("embed", "nvidia/embed", 32),
            )),
            &filter,
        );

        let endpoint = EndpointId::from("prod.backend.generate");
        let view = state.view(&filter);
        let domains = domains_for_endpoint(&view, &endpoint);
        assert_eq!(domains.len(), 2);
        assert!(domains.iter().all(|membership| {
            !membership.is_materializable()
                && membership.compatibility_conflict
                && membership.conflicts.iter().any(|conflict| {
                    matches!(
                        conflict,
                        MembershipConflict::Endpoint { endpoint: conflicted, .. } if conflicted == &endpoint
                    )
                })
        }));
        assert!(
            domains
                .iter()
                .any(|membership| membership.models == ["llama"])
        );
        assert!(
            domains
                .iter()
                .any(|membership| membership.models == ["embed"])
        );
        let llama_generation = domains
            .iter()
            .find(|membership| membership.models == ["llama"])
            .unwrap()
            .generation;

        state.apply(
            DiscoveryEvent::Removed(DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace: "prod".to_string(),
                component: "backend".to_string(),
                endpoint: "generate".to_string(),
                instance_id: 2,
                model_suffix: None,
            })),
            &filter,
        );
        let view = state.view(&filter);
        let domains = domains_for_endpoint(&view, &endpoint);
        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].models, ["llama"]);
        assert!(!domains[0].compatibility_conflict);
        assert!(domains[0].is_materializable());
        assert!(domains[0].generation > llama_generation);
    }

    #[test]
    fn adapter_is_a_loaded_overlay_on_the_backing_base_domain() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                1,
                None,
                card("llama", "meta/llama", 64),
            )),
            &filter,
        );
        let mut adapter = card("tenant-a", "meta/llama", 64);
        adapter.lora = Some(LoraInfo {
            name: "tenant-a".to_string(),
            max_gpu_lora_count: Some(4),
        });
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, Some("tenant-a"), adapter)),
            &filter,
        );

        let endpoint = EndpointId::from("prod.backend.generate");
        let view = state.view(&filter);
        let domains = domains_for_endpoint(&view, &endpoint);
        assert_eq!(domains.len(), 1);
        let membership = domains[0];
        assert_eq!(membership.runtime_configs.len(), 1);
        assert_eq!(membership.models, ["llama", "tenant-a"]);
        let adapter = CanonicalModelId::new("tenant-a").unwrap();
        let overlay = &membership.adapters[&adapter];
        assert_eq!(overlay.base_model.as_str(), "llama");
        assert_eq!(overlay.workers[&1].max_gpu_lora_count, Some(4));
        let registration = membership
            .registrations
            .iter()
            .find(|registration| registration.model() == &adapter)
            .unwrap();
        assert_eq!(
            registration.target(),
            &ModelTarget::Lora {
                base_model: CanonicalModelId::new("llama").unwrap(),
                adapter,
            }
        );
    }

    #[test]
    fn base_model_aliases_remain_attached_to_their_canonical_registration() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut model = card("llama", "meta/llama", 64);
        model.aliases = vec!["chat".to_string(), "instruct".to_string()];
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, None, model)),
            &filter,
        );

        let view = state.view(&filter);
        let endpoint = EndpointId::from("prod.backend.generate");
        let domains = domains_for_endpoint(&view, &endpoint);
        assert_eq!(domains.len(), 1);
        let membership = domains[0];
        assert_eq!(membership.registrations.len(), 1);
        let registration = &membership.registrations[0];
        assert_eq!(registration.model().as_str(), "llama");
        assert_eq!(
            registration
                .aliases()
                .iter()
                .map(ModelAlias::as_str)
                .collect::<Vec<_>>(),
            vec!["chat", "instruct"]
        );
    }

    #[test]
    fn alias_that_is_also_a_different_canonical_model_is_not_published() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut llama = card("llama", "meta/llama", 64);
        llama.aliases = vec!["chat".to_string()];
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, None, llama)),
            &filter,
        );
        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                2,
                None,
                card("chat", "meta/llama", 64),
            )),
            &filter,
        );

        let view = state.view(&filter);
        let endpoint = EndpointId::from("prod.backend.generate");
        let domains = domains_for_endpoint(&view, &endpoint);
        assert_eq!(domains.len(), 1);
        let membership = domains[0];
        assert_eq!(membership.models, ["llama"]);
        assert!(membership.aliases.is_empty());
        assert!(membership.conflicts.iter().any(|conflict| {
            matches!(
                conflict,
                MembershipConflict::Binding { model, .. } if model.as_str() == "chat"
            )
        }));
    }
}
