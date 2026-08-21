// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

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

use super::identity::{
    CanonicalModelId, CanonicalModelRegistration, ModelAlias, ModelTarget, WorkerRole,
};
use super::resolution::{ResolvedIndexerDomain, resolve_indexer_domain};
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;
use crate::model_type::ModelType;
use crate::worker_type::WorkerType;

const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

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

/// Which projection a discovery inconsistency invalidates.
///
/// Pool-materialization conflicts fence CKF publication but do not erase independently valid
/// serving bindings from the topology. Serving-binding conflicts suppress only the binding that
/// cannot be named or resolved; other bindings on the endpoint remain usable. Serving-topology
/// conflicts fence both projections because the endpoint's workers cannot be assigned safely to
/// request-facing models without adding more identity to the membership schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MaterializationConflictScope {
    PoolMaterialization,
    ServingBinding,
    ServingTopology,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationConflictSubject {
    Endpoint(EndpointId),
    Card(ModelCardInstanceId),
    Worker(WorkerId),
    Binding(CanonicalModelId),
}

/// A structural inconsistency found while projecting one endpoint.
///
/// Invalid aliases and orphan adapter cards are soft discovery errors: they are omitted and
/// logged, but do not create a conflict for an otherwise valid base endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationConflict {
    pub(crate) scope: MaterializationConflictScope,
    pub(crate) subject: MaterializationConflictSubject,
    pub(crate) reason: String,
}

impl MaterializationConflict {
    pub(crate) fn pool(subject: MaterializationConflictSubject, reason: impl Into<String>) -> Self {
        Self {
            scope: MaterializationConflictScope::PoolMaterialization,
            subject,
            reason: reason.into(),
        }
    }

    fn serving(subject: MaterializationConflictSubject, reason: impl Into<String>) -> Self {
        Self {
            scope: MaterializationConflictScope::ServingBinding,
            subject,
            reason: reason.into(),
        }
    }

    pub(crate) fn serving_topology(
        subject: MaterializationConflictSubject,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            scope: MaterializationConflictScope::ServingTopology,
            subject,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DomainWorkerTopology {
    pub(crate) worker_type: Option<WorkerType>,
    /// Part of the request plane's WorkerSet identity: two same-role worker sets
    /// serving different surfaces (for example Chat vs Completions) are distinct
    /// routes there and must stay distinct readiness units here.
    pub(crate) model_type: ModelType,
    pub(crate) needs: Vec<Vec<WorkerType>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdapterWorkerMembership {
    pub(crate) max_gpu_lora_count: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdapterMembership {
    pub(crate) base_model: CanonicalModelId,
    /// Runtime LoRA identity used to salt requests and cache hashes. This is independent from the
    /// request-facing model name used as the key in `EndpointMembership::adapters`.
    pub(crate) adapter: CanonicalModelId,
    pub(crate) workers: HashMap<WorkerId, AdapterWorkerMembership>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EndpointMembership {
    pub(crate) endpoint: EndpointId,
    pub(crate) generation: u64,
    pub(crate) domain: Option<KvCacheDomainKey>,
    pub(crate) namespace: String,
    pub(crate) registrations: Vec<CanonicalModelRegistration>,
    pub(crate) models: Vec<String>,
    pub(crate) aliases: Vec<String>,
    pub(crate) roles: Vec<WorkerRole>,
    pub(crate) runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
    pub(crate) worker_topology: HashMap<WorkerId, DomainWorkerTopology>,
    pub(crate) adapters: HashMap<CanonicalModelId, AdapterMembership>,
    pub(crate) conflicts: Vec<MaterializationConflict>,
}

impl EndpointMembership {
    pub(crate) fn has_pool_materialization_conflict(&self) -> bool {
        self.conflicts.iter().any(|conflict| {
            matches!(
                conflict.scope,
                MaterializationConflictScope::PoolMaterialization
                    | MaterializationConflictScope::ServingTopology
            )
        })
    }

    pub(crate) fn has_serving_topology_conflict(&self) -> bool {
        self.conflicts
            .iter()
            .any(|conflict| conflict.scope == MaterializationConflictScope::ServingTopology)
    }

    pub(crate) fn is_materializable(&self) -> bool {
        self.domain.is_some()
            && !self.registrations.is_empty()
            && !self.has_pool_materialization_conflict()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct DcMembershipView {
    pub(crate) endpoints: Arc<HashMap<EndpointId, EndpointMembership>>,
}

#[derive(Debug, Clone)]
struct ProjectedBaseCard<'a> {
    id: &'a ModelCardInstanceId,
    card: &'a ModelDeploymentCard,
    domain: Option<KvCacheDomainKey>,
    model: Option<CanonicalModelId>,
    aliases: Vec<ModelAlias>,
}

#[derive(Debug)]
struct StoredModelCard {
    card: ModelDeploymentCard,
    serialized: serde_json::Value,
}

impl PartialEq for StoredModelCard {
    fn eq(&self, other: &Self) -> bool {
        self.serialized == other.serialized
    }
}

impl StoredModelCard {
    fn update_taints(&mut self, taints: Vec<String>) -> bool {
        let normalized = taints.iter().cloned().collect();
        if self.card.runtime_config.taints == normalized {
            return false;
        }
        self.card.runtime_config.taints = normalized;

        if let Some(runtime_config) = self
            .serialized
            .get_mut("runtime_config")
            .and_then(serde_json::Value::as_object_mut)
        {
            runtime_config.insert(
                "taints".to_string(),
                serde_json::Value::Array(
                    taints.into_iter().map(serde_json::Value::String).collect(),
                ),
            );
        } else {
            self.serialized = serde_json::to_value(&self.card)
                .expect("a deserialized model deployment card must remain serializable");
        }
        true
    }
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

struct EndpointMembershipBuilder {
    domain: Option<KvCacheDomainKey>,
    claims: Vec<RegistrationClaim>,
    runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
    worker_topology: HashMap<WorkerId, DomainWorkerTopology>,
    adapters: HashMap<CanonicalModelId, AdapterMembership>,
    conflicts: Vec<MaterializationConflict>,
}

impl EndpointMembershipBuilder {
    fn new(domain: Option<KvCacheDomainKey>) -> Self {
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

struct ProjectionDiagnostics<'a> {
    invalid_models: HashSet<ModelCardInstanceId>,
    invalid_query_semantics: HashSet<ModelCardInstanceId>,
    invalid_aliases: HashSet<(ModelCardInstanceId, String)>,
    orphan_adapters: HashSet<ModelCardInstanceId>,
    warned_invalid_models: &'a HashSet<ModelCardInstanceId>,
    warned_invalid_query_semantics: &'a HashSet<ModelCardInstanceId>,
    warned_invalid_aliases: &'a HashSet<(ModelCardInstanceId, String)>,
    warned_orphan_adapters: &'a HashSet<ModelCardInstanceId>,
}

impl<'a> ProjectionDiagnostics<'a> {
    fn new(
        warned_invalid_models: &'a HashSet<ModelCardInstanceId>,
        warned_invalid_query_semantics: &'a HashSet<ModelCardInstanceId>,
        warned_invalid_aliases: &'a HashSet<(ModelCardInstanceId, String)>,
        warned_orphan_adapters: &'a HashSet<ModelCardInstanceId>,
    ) -> Self {
        Self {
            invalid_models: HashSet::new(),
            invalid_query_semantics: HashSet::new(),
            invalid_aliases: HashSet::new(),
            orphan_adapters: HashSet::new(),
            warned_invalid_models,
            warned_invalid_query_semantics,
            warned_invalid_aliases,
            warned_orphan_adapters,
        }
    }

    fn valid_aliases(
        &mut self,
        id: &ModelCardInstanceId,
        card: &ModelDeploymentCard,
        model: &CanonicalModelId,
        endpoint: &EndpointId,
    ) -> Vec<ModelAlias> {
        valid_aliases(
            id,
            card,
            model,
            endpoint,
            &mut self.invalid_aliases,
            self.warned_invalid_aliases,
        )
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
    cards: HashMap<ModelCardInstanceId, StoredModelCard>,
    next_membership_generation: u64,
    previous: Arc<HashMap<EndpointId, EndpointMembership>>,
    warned_invalid_models: HashSet<ModelCardInstanceId>,
    warned_invalid_query_semantics: HashSet<ModelCardInstanceId>,
    warned_invalid_aliases: HashSet<(ModelCardInstanceId, String)>,
    warned_orphan_adapters: HashSet<ModelCardInstanceId>,
    #[cfg(test)]
    projection_count: usize,
}

impl Default for MembershipState {
    fn default() -> Self {
        Self {
            cards: HashMap::new(),
            next_membership_generation: 1,
            previous: Arc::default(),
            warned_invalid_models: HashSet::new(),
            warned_invalid_query_semantics: HashSet::new(),
            warned_invalid_aliases: HashSet::new(),
            warned_orphan_adapters: HashSet::new(),
            #[cfg(test)]
            projection_count: 0,
        }
    }
}

impl MembershipState {
    fn replace_all(
        &mut self,
        instances: Vec<DiscoveryInstance>,
        filter: &DcDiscoveryFilter,
    ) -> bool {
        let mut next = HashMap::new();
        for instance in instances {
            let Some((id, card)) = decode_card(instance) else {
                continue;
            };
            if filter.matches(&endpoint_id(&id)) {
                next.insert(id, card);
            }
        }
        if self.cards == next {
            return false;
        }
        self.cards = next;
        true
    }

    fn apply(&mut self, event: DiscoveryEvent, filter: &DcDiscoveryFilter) -> bool {
        match event {
            DiscoveryEvent::Added(instance) => {
                let Some((id, card)) = decode_card(instance) else {
                    return false;
                };
                if filter.matches(&endpoint_id(&id)) {
                    if self.cards.get(&id) == Some(&card) {
                        return false;
                    }
                    self.cards.insert(id, card);
                    return true;
                }
                false
            }
            DiscoveryEvent::ModelTaintsUpdated(update) => {
                if !filter.matches(&endpoint_id(&update.id)) {
                    return false;
                }
                let Some(card) = self.cards.get_mut(&update.id) else {
                    tracing::warn!(
                        instance_id = update.id.instance_id,
                        "Ignoring taint update for an unknown cross-DC model card"
                    );
                    return false;
                };
                card.update_taints(update.taints)
            }
            DiscoveryEvent::Removed(DiscoveryInstanceId::Model(id)) => {
                self.cards.remove(&id).is_some()
            }
            DiscoveryEvent::Removed(_) => false,
        }
    }

    fn view(&mut self, filter: &DcDiscoveryFilter) -> DcMembershipView {
        #[cfg(test)]
        {
            self.projection_count = self.projection_count.saturating_add(1);
        }
        let mut diagnostics = ProjectionDiagnostics::new(
            &self.warned_invalid_models,
            &self.warned_invalid_query_semantics,
            &self.warned_invalid_aliases,
            &self.warned_orphan_adapters,
        );
        let mut grouped: HashMap<EndpointId, Vec<(&ModelCardInstanceId, &ModelDeploymentCard)>> =
            HashMap::new();
        for (id, stored) in &self.cards {
            let endpoint = endpoint_id(id);
            if filter.matches(&endpoint) {
                grouped
                    .entry(endpoint)
                    .or_default()
                    .push((id, &stored.card));
            }
        }

        let mut builders = HashMap::<EndpointId, EndpointMembershipBuilder>::new();
        for (endpoint, cards) in grouped {
            let mut base_cards = Vec::new();
            let mut adapter_cards = Vec::new();
            let mut query_semantics_conflicts = Vec::new();
            let mut serving_shape_conflicts = Vec::new();
            for (id, card) in cards {
                match (id.model_suffix.is_some(), card.lora.is_some()) {
                    (true, true) => {
                        adapter_cards.push((id, card));
                        continue;
                    }
                    (false, false) => {}
                    _ => {
                        diagnostics.invalid_models.insert(id.clone());
                        if !diagnostics.warned_invalid_models.contains(id) {
                            tracing::warn!(
                                endpoint = %endpoint,
                                model = card.name(),
                                card = %id.to_path(),
                                "Ignoring model card whose LoRA discovery identity and metadata disagree"
                            );
                        }
                        serving_shape_conflicts.push(MaterializationConflict::serving(
                            MaterializationConflictSubject::Card(id.clone()),
                            "LoRA discovery identity and card metadata disagree",
                        ));
                        continue;
                    }
                }
                let domain = match resolve_indexer_domain(card, &endpoint) {
                    Ok(domain) => Some(domain),
                    Err(error) => {
                        diagnostics.invalid_query_semantics.insert(id.clone());
                        if !diagnostics.warned_invalid_query_semantics.contains(id) {
                            tracing::warn!(
                                endpoint = %endpoint,
                                model = card.name(),
                                %error,
                                "Model card has invalid KV query semantics"
                            );
                        }
                        query_semantics_conflicts.push(MaterializationConflict::pool(
                            MaterializationConflictSubject::Card(id.clone()),
                            error.to_string(),
                        ));
                        None
                    }
                };
                let model = match CanonicalModelId::new(card.name().to_string()) {
                    Ok(model) => Some(model),
                    Err(error) => {
                        diagnostics.invalid_models.insert(id.clone());
                        if !diagnostics.warned_invalid_models.contains(id) {
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
                    .map(|model| diagnostics.valid_aliases(id, card, model, &endpoint))
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
                .filter_map(|projection| projection.domain.clone())
                .collect();
            let domain_count = endpoint_domains.len();
            let all_domains_known = base_cards
                .iter()
                .all(|projection| projection.domain.is_some());
            let domain = (all_domains_known && domain_count == 1)
                .then(|| endpoint_domains.into_iter().next())
                .flatten();
            let mut builder = EndpointMembershipBuilder::new(domain);
            builder.conflicts.extend(query_semantics_conflicts);
            builder.conflicts.extend(serving_shape_conflicts);
            if domain_count > 1 {
                builder.conflicts.push(MaterializationConflict::pool(
                    MaterializationConflictSubject::Endpoint(endpoint.clone()),
                    "endpoint resolves to multiple indexer domains",
                ));
            }

            let base_models: HashSet<_> = base_cards
                .iter()
                .filter_map(|projection| projection.model.clone())
                .collect();
            if base_models.len() > 1 {
                builder
                    .conflicts
                    .push(MaterializationConflict::serving_topology(
                        MaterializationConflictSubject::Endpoint(endpoint.clone()),
                        "endpoint resolves to multiple canonical base models",
                    ));
            }

            let mut base_candidates = HashMap::<WorkerId, Vec<&ProjectedBaseCard<'_>>>::new();
            for projection in &base_cards {
                base_candidates
                    .entry(projection.id.instance_id)
                    .or_default()
                    .push(projection);
            }

            let mut base_by_worker = HashMap::with_capacity(base_candidates.len());
            for (worker_id, projections) in base_candidates {
                let [projection] = projections.as_slice() else {
                    builder
                        .conflicts
                        .push(MaterializationConflict::serving_topology(
                            MaterializationConflictSubject::Worker(worker_id),
                            "worker publishes multiple base model cards",
                        ));
                    continue;
                };
                base_by_worker.insert(worker_id, *projection);
                let Some(model) = projection.model.clone() else {
                    builder.conflicts.push(MaterializationConflict::serving(
                        MaterializationConflictSubject::Card(projection.id.clone()),
                        "invalid canonical model identity",
                    ));
                    continue;
                };
                builder
                    .runtime_configs
                    .insert(worker_id, projection.card.runtime_config.clone());
                builder.worker_topology.insert(
                    worker_id,
                    DomainWorkerTopology {
                        worker_type: projection.card.worker_type,
                        model_type: projection.card.model_type,
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
                let Some(base) = base_by_worker.get(&worker_id).copied() else {
                    diagnostics.orphan_adapters.insert(id.clone());
                    if !diagnostics.warned_orphan_adapters.contains(id) {
                        tracing::warn!(
                            endpoint = %endpoint,
                            worker_id,
                            card = %id.to_path(),
                            "Ignoring adapter card without a backing base model card"
                        );
                    }
                    continue;
                };
                let Some(base_model) = base.model.clone() else {
                    builder.conflicts.push(MaterializationConflict::serving(
                        MaterializationConflictSubject::Card(id.clone()),
                        "adapter's backing base model identity is invalid",
                    ));
                    continue;
                };
                let Some(lora) = card.lora.as_ref() else {
                    builder.conflicts.push(MaterializationConflict::serving(
                        MaterializationConflictSubject::Card(id.clone()),
                        "adapter card has no runtime LoRA metadata",
                    ));
                    continue;
                };
                let served_model = match CanonicalModelId::new(card.name().to_string()) {
                    Ok(model) => model,
                    Err(error) => {
                        diagnostics.invalid_models.insert(id.clone());
                        if !diagnostics.warned_invalid_models.contains(id) {
                            tracing::warn!(
                                endpoint = %endpoint,
                                model = card.name(),
                                %error,
                                "Ignoring adapter card with invalid request-facing model identity"
                            );
                        }
                        builder.conflicts.push(MaterializationConflict::serving(
                            MaterializationConflictSubject::Card(id.clone()),
                            "invalid adapter request-facing model identity",
                        ));
                        continue;
                    }
                };
                let runtime_adapter = match CanonicalModelId::new(lora.name.clone()) {
                    Ok(adapter) => adapter,
                    Err(error) => {
                        diagnostics.invalid_models.insert(id.clone());
                        if !diagnostics.warned_invalid_models.contains(id) {
                            tracing::warn!(
                                endpoint = %endpoint,
                                model = card.name(),
                                adapter = lora.name,
                                %error,
                                "Ignoring adapter card with invalid runtime LoRA identity"
                            );
                        }
                        builder.conflicts.push(MaterializationConflict::serving(
                            MaterializationConflictSubject::Card(id.clone()),
                            "invalid runtime LoRA identity",
                        ));
                        continue;
                    }
                };
                let aliases = diagnostics.valid_aliases(id, card, &served_model, &endpoint);
                let target = ModelTarget::Lora {
                    base_model: base_model.clone(),
                    adapter: runtime_adapter.clone(),
                };
                builder.claims.push(RegistrationClaim {
                    binding: BindingIdentity {
                        model: served_model.clone(),
                        target,
                    },
                    aliases,
                });
                let capacity = lora.max_gpu_lora_count;
                match builder.adapters.entry(served_model) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(AdapterMembership {
                            base_model,
                            adapter: runtime_adapter,
                            workers: HashMap::from([(
                                worker_id,
                                AdapterWorkerMembership {
                                    max_gpu_lora_count: capacity,
                                },
                            )]),
                        });
                    }
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        if entry.get().base_model != base_model
                            || entry.get().adapter != runtime_adapter
                        {
                            builder.conflicts.push(MaterializationConflict::serving(
                                MaterializationConflictSubject::Card(id.clone()),
                                "adapter request-facing model resolves to conflicting runtime targets",
                            ));
                            continue;
                        }
                        let workers = &mut entry.get_mut().workers;
                        let facts = AdapterWorkerMembership {
                            max_gpu_lora_count: capacity,
                        };
                        if workers
                            .insert(worker_id, facts)
                            .is_some_and(|previous| previous != facts)
                        {
                            workers.remove(&worker_id);
                            builder.conflicts.push(MaterializationConflict::pool(
                                MaterializationConflictSubject::Worker(worker_id),
                                "worker publishes conflicting adapter capacity",
                            ));
                        }
                    }
                }
            }
            builders.insert(endpoint, builder);
        }

        // The local ModelManager can resolve a collision by its local first-wins order. A Relay
        // federates independently owned endpoints and has no safe canonical owner to choose, so
        // any request-facing name with multiple targets is omitted from every endpoint.
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

        let mut endpoints = HashMap::new();
        for (endpoint, mut builder) in builders {
            let mut grouped_claims = HashMap::<BindingIdentity, HashSet<ModelAlias>>::new();
            for claim in builder.claims {
                if ambiguous_names.contains(claim.binding.model.as_str()) {
                    builder.conflicts.push(MaterializationConflict::serving(
                        MaterializationConflictSubject::Binding(claim.binding.model),
                        "request-facing model name resolves to conflicting targets",
                    ));
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
            builder.adapters.retain(|served_model, adapter| {
                registrations.iter().any(|registration| {
                    registration.model() == served_model
                        && registration.target()
                            == &ModelTarget::Lora {
                                base_model: adapter.base_model.clone(),
                                adapter: adapter.adapter.clone(),
                            }
                })
            });
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
                .map(|topology| WorkerRole::from_worker_type(topology.worker_type))
                .collect::<HashSet<_>>();
            builder
                .conflicts
                .sort_by(|left, right| format!("{left:?}").cmp(&format!("{right:?}")));
            builder.conflicts.dedup();
            let mut candidate = EndpointMembership {
                endpoint: endpoint.clone(),
                generation: self
                    .previous
                    .get(&endpoint)
                    .map_or(0, |previous| previous.generation),
                domain: builder.domain,
                namespace: endpoint.namespace.clone(),
                registrations,
                models: sorted(models),
                aliases: sorted(aliases),
                roles: sorted_values(roles),
                runtime_configs: builder.runtime_configs,
                worker_topology: builder.worker_topology,
                adapters: builder.adapters,
                conflicts: builder.conflicts,
            };
            let changed = self
                .previous
                .get(&endpoint)
                .is_none_or(|previous| !same_membership(previous, &candidate));
            if changed {
                candidate.generation = self.next_membership_generation;
                self.next_membership_generation = self.next_membership_generation.saturating_add(1);
            }
            endpoints.insert(endpoint, candidate);
        }

        let ProjectionDiagnostics {
            invalid_models,
            invalid_query_semantics,
            invalid_aliases,
            orphan_adapters,
            ..
        } = diagnostics;
        self.warned_invalid_models = invalid_models;
        self.warned_invalid_query_semantics = invalid_query_semantics;
        self.warned_invalid_aliases = invalid_aliases;
        self.warned_orphan_adapters = orphan_adapters;
        let endpoints = Arc::new(endpoints);
        self.previous = endpoints.clone();
        DcMembershipView { endpoints }
    }
}

#[cfg(test)]
pub(super) fn project_instances_for_test(instances: Vec<DiscoveryInstance>) -> DcMembershipView {
    let filter = DcDiscoveryFilter::default();
    let mut state = MembershipState::default();
    state.replace_all(instances, &filter);
    state.view(&filter)
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
                        if state.apply(event, &filter) {
                            publish_membership_if_changed(&sender, state.view(&filter));
                        }
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
                        if state.replace_all(instances, &filter) {
                            publish_membership_if_changed(&sender, state.view(&filter));
                        }
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

fn decode_card(instance: DiscoveryInstance) -> Option<(ModelCardInstanceId, StoredModelCard)> {
    let DiscoveryInstanceId::Model(id) = instance.id() else {
        return None;
    };
    let DiscoveryInstance::Model { card_json, .. } = &instance else {
        return None;
    };
    match instance.deserialize_model::<ModelDeploymentCard>() {
        Ok(mut card) => {
            crate::discovery::readiness::normalize_legacy_prefill_topology(&mut card);
            Some((
                id,
                StoredModelCard {
                    card,
                    serialized: card_json.clone(),
                },
            ))
        }
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

fn sorted_values<T: Ord>(values: HashSet<T>) -> Vec<T> {
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

fn same_membership(left: &EndpointMembership, right: &EndpointMembership) -> bool {
    left.endpoint == right.endpoint
        && left.domain == right.domain
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
    use dynamo_runtime::discovery::ModelTaintsUpdate;

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
        let first = instance("generate", 1, None, card("llama", "meta/llama", 64));
        assert!(state.apply(DiscoveryEvent::Added(first.clone()), &filter));
        let initial = state.view(&filter);
        let (sender, mut receiver) = watch::channel(initial.clone());
        let projection_count = state.projection_count;

        let changed = state.apply(DiscoveryEvent::Added(first.clone()), &filter);
        if changed {
            publish_membership_if_changed(&sender, state.view(&filter));
        }
        assert!(!changed);
        assert_eq!(state.projection_count, projection_count);
        assert!(!receiver.has_changed().unwrap());

        let changed = state.replace_all(vec![first], &filter);
        if changed {
            publish_membership_if_changed(&sender, state.view(&filter));
        }
        assert!(!changed);
        assert_eq!(state.projection_count, projection_count);
        assert!(!receiver.has_changed().unwrap());

        let changed = state.apply(
            DiscoveryEvent::Removed(DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace: "prod".to_string(),
                component: "backend".to_string(),
                endpoint: "generate".to_string(),
                instance_id: 999,
                model_suffix: None,
            })),
            &filter,
        );
        if changed {
            publish_membership_if_changed(&sender, state.view(&filter));
        }
        assert!(!changed);
        assert_eq!(state.projection_count, projection_count);
        assert!(!receiver.has_changed().unwrap());

        let changed = state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                2,
                None,
                card("llama", "meta/llama", 64),
            )),
            &filter,
        );
        if changed {
            publish_membership_if_changed(&sender, state.view(&filter));
        }
        assert!(changed);
        assert_eq!(state.projection_count, projection_count + 1);
        assert!(receiver.has_changed().unwrap());
        assert_ne!(*receiver.borrow_and_update(), initial);
    }

    #[test]
    fn reappearing_endpoint_advances_its_generation() {
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
        let endpoint = EndpointId::from("prod.backend.generate");

        state.apply(DiscoveryEvent::Added(discovery_instance.clone()), &filter);
        let initial = state.view(&filter);
        assert_eq!(membership_for_endpoint(&initial, &endpoint).generation, 1);

        state.apply(DiscoveryEvent::Removed(instance_id), &filter);
        assert!(state.view(&filter).endpoints.is_empty());

        state.apply(DiscoveryEvent::Added(discovery_instance), &filter);
        let reappeared = state.view(&filter);
        assert_eq!(
            membership_for_endpoint(&reappeared, &endpoint).generation,
            2
        );
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

    fn membership_for_endpoint<'a>(
        view: &'a DcMembershipView,
        endpoint: &EndpointId,
    ) -> &'a EndpointMembership {
        &view.endpoints[endpoint]
    }

    fn is_pool_endpoint_conflict(
        conflict: &MaterializationConflict,
        endpoint: &EndpointId,
        reason: &str,
    ) -> bool {
        conflict.scope == MaterializationConflictScope::PoolMaterialization
            && matches!(
                &conflict.subject,
                MaterializationConflictSubject::Endpoint(conflicted) if conflicted == endpoint
            )
            && conflict.reason.contains(reason)
    }

    #[test]
    fn zero_block_size_is_not_materialized() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        state.replace_all(
            vec![instance(
                "generate",
                1,
                None,
                card("llama", "meta/llama", 0),
            )],
            &filter,
        );

        let view = state.view(&filter);
        let membership = membership_for_endpoint(&view, &EndpointId::from("prod.backend.generate"));
        assert!(!membership.is_materializable());
        assert!(membership.domain.is_none());
        assert!(membership.conflicts.iter().any(|conflict| {
            conflict.scope == MaterializationConflictScope::PoolMaterialization
                && matches!(&conflict.subject, MaterializationConflictSubject::Card(_))
        }));
        assert_eq!(membership.models, ["llama"]);
        assert_eq!(membership.registrations.len(), 1);
        assert!(membership.worker_topology.contains_key(&1));
    }

    #[test]
    fn encode_role_is_preserved_for_surface_less_and_front_door_cards() {
        use crate::model_type::ModelType;

        let mut surface_less = card("vision-language", "vision-language", 16);
        surface_less.worker_type = Some(WorkerType::Encode);
        surface_less.model_type = ModelType::empty();
        let mut front_door = surface_less.clone();
        front_door.model_type = ModelType::Chat;

        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        state.replace_all(
            vec![
                instance("encoder-internal", 1, None, surface_less),
                instance("encoder-frontdoor", 2, None, front_door),
            ],
            &filter,
        );
        let view = state.view(&filter);

        for endpoint in [
            EndpointId::from("prod.backend.encoder-internal"),
            EndpointId::from("prod.backend.encoder-frontdoor"),
        ] {
            let membership = membership_for_endpoint(&view, &endpoint);
            assert_eq!(membership.roles, [WorkerRole::Encode]);
            assert_eq!(
                membership
                    .worker_topology
                    .values()
                    .next()
                    .unwrap()
                    .worker_type,
                Some(WorkerType::Encode)
            );
        }
    }

    #[test]
    fn standard_and_eagle_workers_under_one_endpoint_are_fenced_together() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let standard = card("llama", "meta/llama", 64);
        let mut eagle = standard.clone();
        eagle.runtime_config.enable_eagle = true;
        state.replace_all(
            vec![
                instance("generate", 1, None, standard),
                instance("generate", 2, None, eagle),
            ],
            &filter,
        );

        let endpoint = EndpointId::from("prod.backend.generate");
        let view = state.view(&filter);
        assert_eq!(view.endpoints.len(), 1);
        let membership = membership_for_endpoint(&view, &endpoint);
        assert!(membership.domain.is_none());
        assert!(!membership.is_materializable());
        assert!(membership.conflicts.iter().any(|conflict| {
            is_pool_endpoint_conflict(conflict, &endpoint, "multiple indexer domains")
        }));
    }

    #[test]
    fn prefill_and_decode_on_distinct_endpoints_materialize_independently() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut prefill = card("llama", "meta/llama", 64);
        prefill.worker_type = Some(WorkerType::Prefill);
        let mut decode = prefill.clone();
        decode.worker_type = Some(WorkerType::Decode);
        state.replace_all(
            vec![
                instance("prefill", 1, None, prefill),
                instance("generate", 2, None, decode),
            ],
            &filter,
        );

        let view = state.view(&filter);
        assert_eq!(view.endpoints.len(), 2);
        let prefill = membership_for_endpoint(&view, &EndpointId::from("prod.backend.prefill"));
        let decode = membership_for_endpoint(&view, &EndpointId::from("prod.backend.generate"));
        assert!(prefill.is_materializable());
        assert!(decode.is_materializable());
        assert_eq!(prefill.roles, [WorkerRole::Prefill]);
        assert_eq!(decode.roles, [WorkerRole::Decode]);
        assert_ne!(
            prefill.domain.as_ref().unwrap().id,
            decode.domain.as_ref().unwrap().id
        );
    }

    #[test]
    fn scoped_taint_update_reprojects_known_card_without_synthesizing_unknown_worker() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut base = card("llama", "meta/llama", 64);
        base.runtime_config.taints = HashSet::from(["old".to_string()]);
        let base_instance = instance("generate", 1, None, base);
        let DiscoveryInstanceId::Model(id) = base_instance.id() else {
            unreachable!()
        };
        assert!(state.apply(DiscoveryEvent::Added(base_instance), &filter));

        assert!(state.apply(
            DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                id: id.clone(),
                taints: vec!["blue".to_string(), "gpu".to_string()],
            }),
            &filter,
        ));
        let endpoint = EndpointId::from("prod.backend.generate");
        let view = state.view(&filter);
        let membership = membership_for_endpoint(&view, &endpoint);
        assert_eq!(
            membership.runtime_configs.get(&1).unwrap().taints,
            HashSet::from(["blue".to_string(), "gpu".to_string()])
        );

        assert!(!state.apply(
            DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                id: id.clone(),
                taints: vec!["blue".to_string(), "gpu".to_string()],
            }),
            &filter,
        ));
        assert!(!state.apply(
            DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
                id: ModelCardInstanceId {
                    instance_id: 99,
                    ..id
                },
                taints: vec!["unknown".to_string()],
            }),
            &filter,
        ));
        assert_eq!(state.cards.len(), 1);
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
        assert_eq!(view.endpoints.len(), 1);
        let membership = membership_for_endpoint(&view, &endpoint);
        assert!(membership.domain.is_none());
        assert!(!membership.is_materializable());
        assert_eq!(membership.models, ["embed", "llama"]);
        assert!(membership.conflicts.iter().any(|conflict| {
            is_pool_endpoint_conflict(conflict, &endpoint, "multiple indexer domains")
        }));
        let conflicted_generation = membership.generation;

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
        let membership = membership_for_endpoint(&view, &endpoint);
        assert_eq!(membership.models, ["llama"]);
        assert!(membership.domain.is_some());
        assert!(membership.conflicts.is_empty());
        assert!(membership.is_materializable());
        assert!(membership.generation > conflicted_generation);
    }

    #[test]
    fn different_base_models_under_one_endpoint_are_a_hard_conflict() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        state.replace_all(
            vec![
                instance("generate", 1, None, card("llama", "meta/llama", 64)),
                instance("generate", 2, None, card("chat", "meta/llama", 64)),
            ],
            &filter,
        );

        let endpoint = EndpointId::from("prod.backend.generate");
        let view = state.view(&filter);
        let membership = membership_for_endpoint(&view, &endpoint);
        assert!(membership.domain.is_some());
        assert!(!membership.is_materializable());
        assert!(membership.conflicts.iter().any(|conflict| {
            conflict.scope == MaterializationConflictScope::ServingTopology
                && matches!(
                    &conflict.subject,
                    MaterializationConflictSubject::Endpoint(conflicted)
                        if conflicted == &endpoint
                )
                && conflict.reason.contains("canonical base models")
        }));
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
        let membership = membership_for_endpoint(&view, &endpoint);
        assert!(membership.is_materializable());
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
    fn adapter_served_model_is_distinct_from_runtime_lora_identity() {
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
        let mut adapter = card("tenant-chat", "meta/llama", 64);
        adapter.lora = Some(LoraInfo {
            name: "org/runtime-adapter-v2".to_string(),
            max_gpu_lora_count: Some(4),
        });
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, Some("adapter-v2"), adapter)),
            &filter,
        );

        let view = state.view(&filter);
        let membership = membership_for_endpoint(&view, &EndpointId::from("prod.backend.generate"));
        let served_model = CanonicalModelId::new("tenant-chat").unwrap();
        let runtime_adapter = CanonicalModelId::new("org/runtime-adapter-v2").unwrap();
        let overlay = &membership.adapters[&served_model];
        assert_eq!(overlay.adapter, runtime_adapter);
        let registration = membership
            .registrations
            .iter()
            .find(|registration| registration.model() == &served_model)
            .expect("request-facing adapter registration");
        assert_eq!(
            registration.target(),
            &ModelTarget::Lora {
                base_model: CanonicalModelId::new("llama").unwrap(),
                adapter: runtime_adapter,
            }
        );
    }

    #[test]
    fn malformed_adapter_shape_suppresses_only_that_binding() {
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
        let mut malformed = card("tenant-chat", "meta/llama", 64);
        malformed.lora = Some(LoraInfo {
            name: "org/runtime-adapter-v2".to_string(),
            max_gpu_lora_count: Some(4),
        });
        // The card says LoRA, but its discovery identity has no model suffix.
        state.apply(
            DiscoveryEvent::Added(instance("generate", 2, None, malformed)),
            &filter,
        );

        let view = state.view(&filter);
        let membership = membership_for_endpoint(&view, &EndpointId::from("prod.backend.generate"));
        assert!(membership.is_materializable());
        assert_eq!(membership.models, ["llama"]);
        assert!(membership.adapters.is_empty());
        assert!(membership.conflicts.iter().any(|conflict| {
            conflict.scope == MaterializationConflictScope::ServingBinding
                && matches!(
                    &conflict.subject,
                    MaterializationConflictSubject::Card(card) if card.instance_id == 2
                )
                && conflict.reason.contains("LoRA discovery identity")
        }));
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
        let membership = membership_for_endpoint(&view, &endpoint);
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
    fn federated_alias_collision_is_omitted_from_every_endpoint() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut llama = card("llama", "meta/llama", 64);
        llama.aliases = vec!["shared-chat".to_string()];
        let mut mistral = card("mistral", "mistral/model", 64);
        mistral.aliases = vec!["shared-chat".to_string()];
        state.replace_all(
            vec![
                instance("llama", 1, None, llama),
                instance("mistral", 2, None, mistral),
            ],
            &filter,
        );

        let view = state.view(&filter);
        for membership in view.endpoints.values() {
            assert!(membership.is_materializable());
            assert!(membership.aliases.is_empty());
            assert_eq!(membership.registrations.len(), 1);
            assert!(membership.registrations[0].aliases().is_empty());
        }
    }
}
