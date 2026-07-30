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

use super::identity::{CanonicalModelId, CanonicalModelRegistration, ModelAlias, ModelTarget};
use super::resolution::{ResolvedIndexerDomain, resolve_indexer_domain};
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;

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

/// A structural inconsistency that makes the endpoint unsafe to materialize.
///
/// Invalid aliases and orphan adapter cards are soft discovery errors: they are omitted and
/// logged, but do not create a conflict for an otherwise valid base endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationConflict {
    Endpoint {
        endpoint: EndpointId,
        reason: String,
    },
    Card {
        card: ModelCardInstanceId,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EndpointMembership {
    pub(crate) endpoint: EndpointId,
    pub(crate) generation: u64,
    pub(crate) domain: Option<KvCacheDomainKey>,
    pub(crate) registrations: Vec<CanonicalModelRegistration>,
    pub(crate) models: Vec<String>,
    pub(crate) aliases: Vec<String>,
    pub(crate) roles: Vec<String>,
    pub(crate) runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
    pub(crate) conflicts: Vec<MaterializationConflict>,
}

impl EndpointMembership {
    pub(crate) fn is_materializable(&self) -> bool {
        self.domain.is_some() && !self.registrations.is_empty() && self.conflicts.is_empty()
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
    domain: KvCacheDomainKey,
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
    roles: HashSet<String>,
    conflicts: Vec<MaterializationConflict>,
}

struct ProjectionDiagnostics<'a> {
    invalid_models: HashSet<ModelCardInstanceId>,
    invalid_aliases: HashSet<(ModelCardInstanceId, String)>,
    orphan_adapters: HashSet<ModelCardInstanceId>,
    warned_invalid_models: &'a HashSet<ModelCardInstanceId>,
    warned_invalid_aliases: &'a HashSet<(ModelCardInstanceId, String)>,
    warned_orphan_adapters: &'a HashSet<ModelCardInstanceId>,
}

impl<'a> ProjectionDiagnostics<'a> {
    fn new(
        warned_invalid_models: &'a HashSet<ModelCardInstanceId>,
        warned_invalid_aliases: &'a HashSet<(ModelCardInstanceId, String)>,
        warned_orphan_adapters: &'a HashSet<ModelCardInstanceId>,
    ) -> Self {
        Self {
            invalid_models: HashSet::new(),
            invalid_aliases: HashSet::new(),
            orphan_adapters: HashSet::new(),
            warned_invalid_models,
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
            for (id, card) in cards {
                if id.model_suffix.is_some() || card.lora.is_some() {
                    adapter_cards.push((id, card));
                    continue;
                }
                let domain = resolve_indexer_domain(card, &endpoint, KV_EVENT_HASH_FORMAT_VERSION);
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
                .map(|projection| projection.domain.clone())
                .collect();
            let domain_count = endpoint_domains.len();
            let domain = (domain_count == 1)
                .then(|| endpoint_domains.into_iter().next())
                .flatten();
            let mut builder = EndpointMembershipBuilder {
                domain,
                claims: Vec::new(),
                runtime_configs: HashMap::new(),
                roles: HashSet::new(),
                conflicts: Vec::new(),
            };
            if domain_count > 1 {
                builder.conflicts.push(MaterializationConflict::Endpoint {
                    endpoint: endpoint.clone(),
                    reason: "endpoint resolves to multiple indexer domains".to_string(),
                });
            }

            let base_models: HashSet<_> = base_cards
                .iter()
                .filter_map(|projection| projection.model.clone())
                .collect();
            if base_models.len() > 1 {
                builder.conflicts.push(MaterializationConflict::Endpoint {
                    endpoint: endpoint.clone(),
                    reason: "endpoint resolves to multiple canonical base models".to_string(),
                });
            }

            let mut base_by_worker = HashMap::with_capacity(base_cards.len());
            for projection in &base_cards {
                let worker_id = projection.id.instance_id;
                debug_assert!(base_by_worker.insert(worker_id, projection).is_none());
                let Some(model) = projection.model.clone() else {
                    builder.conflicts.push(MaterializationConflict::Card {
                        card: projection.id.clone(),
                        reason: "invalid canonical model identity".to_string(),
                    });
                    continue;
                };
                builder
                    .runtime_configs
                    .insert(worker_id, projection.card.runtime_config.clone());
                if let Some(worker_type) = projection.card.worker_type {
                    builder.roles.insert(worker_type.as_str().to_string());
                }
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
                    builder.conflicts.push(MaterializationConflict::Card {
                        card: id.clone(),
                        reason: "adapter's backing base model identity is invalid".to_string(),
                    });
                    continue;
                };
                let adapter_name = card
                    .lora
                    .as_ref()
                    .map(|lora| lora.name.as_str())
                    .or(id.model_suffix.as_deref());
                let Some(adapter_name) = adapter_name else {
                    builder.conflicts.push(MaterializationConflict::Card {
                        card: id.clone(),
                        reason: "adapter card has no adapter identity".to_string(),
                    });
                    continue;
                };
                let adapter = match CanonicalModelId::new(adapter_name.to_string()) {
                    Ok(adapter) => adapter,
                    Err(error) => {
                        diagnostics.invalid_models.insert(id.clone());
                        if !diagnostics.warned_invalid_models.contains(id) {
                            tracing::warn!(
                                endpoint = %endpoint,
                                model = adapter_name,
                                %error,
                                "Ignoring adapter card with invalid canonical model identity"
                            );
                        }
                        builder.conflicts.push(MaterializationConflict::Card {
                            card: id.clone(),
                            reason: "invalid adapter model identity".to_string(),
                        });
                        continue;
                    }
                };
                let aliases = diagnostics.valid_aliases(id, card, &adapter, &endpoint);
                let target = ModelTarget::Lora {
                    base_model,
                    adapter: adapter.clone(),
                };
                builder.claims.push(RegistrationClaim {
                    binding: BindingIdentity {
                        model: adapter,
                        target,
                    },
                    aliases,
                });
            }
            builders.insert(endpoint, builder);
        }

        let mut endpoints = HashMap::new();
        for (endpoint, mut builder) in builders {
            let mut lookup_owners = HashMap::<String, HashSet<BindingIdentity>>::new();
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
            for name in lookup_owners
                .into_iter()
                .filter_map(|(name, owners)| (owners.len() > 1).then_some(name))
            {
                builder.conflicts.push(MaterializationConflict::Endpoint {
                    endpoint: endpoint.clone(),
                    reason: format!(
                        "request-facing name {name} resolves to conflicting targets within the endpoint pool"
                    ),
                });
            }

            let mut grouped_claims = HashMap::<BindingIdentity, HashSet<ModelAlias>>::new();
            for claim in builder.claims {
                let aliases = grouped_claims.entry(claim.binding).or_default();
                aliases.extend(claim.aliases);
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
                registrations,
                models: sorted(models),
                aliases: sorted(aliases),
                roles: sorted(builder.roles),
                runtime_configs: builder.runtime_configs,
                conflicts: builder.conflicts,
            };
            if self.previous.get(&endpoint) != Some(&candidate) {
                candidate.generation = self.next_membership_generation;
                self.next_membership_generation = self.next_membership_generation.saturating_add(1);
            }
            endpoints.insert(endpoint, candidate);
        }

        let ProjectionDiagnostics {
            invalid_models,
            invalid_aliases,
            orphan_adapters,
            ..
        } = diagnostics;
        self.warned_invalid_models = invalid_models;
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
        Ok(card) => Some((
            id,
            StoredModelCard {
                card,
                serialized: card_json.clone(),
            },
        )),
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

        state.apply(DiscoveryEvent::Added(discovery_instance.clone()), &filter);
        let initial = state.view(&filter);
        assert_eq!(initial.endpoints.len(), 1);
        assert_eq!(initial.endpoints.values().next().unwrap().generation, 1);

        state.apply(DiscoveryEvent::Removed(instance_id), &filter);
        assert!(state.view(&filter).endpoints.is_empty());

        state.apply(DiscoveryEvent::Added(discovery_instance), &filter);
        let reappeared = state.view(&filter);
        assert_eq!(reappeared.endpoints.len(), 1);
        assert_eq!(reappeared.endpoints.values().next().unwrap().generation, 2);
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
            matches!(
                conflict,
                MaterializationConflict::Endpoint { endpoint: conflicted, .. } if conflicted == &endpoint
            )
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
                card("chat", "meta/llama", 64),
            )),
            &filter,
        );

        let endpoint = EndpointId::from("prod.backend.generate");
        let view = state.view(&filter);
        let membership = membership_for_endpoint(&view, &endpoint);
        assert!(membership.domain.is_some());
        assert!(!membership.is_materializable());
        assert!(membership.conflicts.iter().any(|conflict| {
            matches!(
                conflict,
                MaterializationConflict::Endpoint { endpoint: conflicted, reason }
                    if conflicted == &endpoint && reason.contains("canonical base models")
            )
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
    fn adapter_metadata_does_not_override_the_base_pool_domain() {
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
        let mut adapter = card("tenant-a", "unrelated/adapter", 1);
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
        assert!(membership.conflicts.is_empty());
        assert_eq!(
            membership
                .domain
                .as_ref()
                .unwrap()
                .diagnostic_model_artifact,
            "meta/llama"
        );
        assert_eq!(membership.models, ["llama", "tenant-a"]);
    }

    #[test]
    fn orphan_adapter_is_soft_and_does_not_block_a_valid_base_endpoint() {
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
            DiscoveryEvent::Added(instance("generate", 2, Some("tenant-a"), adapter)),
            &filter,
        );

        let endpoint = EndpointId::from("prod.backend.generate");
        let view = state.view(&filter);
        let membership = membership_for_endpoint(&view, &endpoint);
        assert!(membership.is_materializable());
        assert!(membership.conflicts.is_empty());
        assert_eq!(membership.models, ["llama"]);
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
    fn request_names_are_scoped_to_each_endpoint_pool() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut fast = card("llama", "meta/llama", 64);
        fast.aliases = vec!["chat".to_string()];
        let mut slow = card("mistral", "mistral/model", 64);
        slow.aliases = vec!["chat".to_string()];
        state.apply(
            DiscoveryEvent::Added(instance("fast", 1, None, fast)),
            &filter,
        );
        state.apply(
            DiscoveryEvent::Added(instance("slow", 2, None, slow)),
            &filter,
        );

        let view = state.view(&filter);
        assert_eq!(view.endpoints.len(), 2);
        assert!(view.endpoints.values().all(|membership| {
            membership.is_materializable() && membership.aliases == ["chat"]
        }));
    }

    #[test]
    fn request_names_must_be_unambiguous_within_an_endpoint_pool() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut base = card("llama", "meta/llama", 64);
        base.aliases = vec!["tenant-a".to_string()];
        let mut adapter = card("tenant-a", "unrelated/adapter", 1);
        adapter.lora = Some(LoraInfo {
            name: "tenant-a".to_string(),
            max_gpu_lora_count: Some(4),
        });
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, None, base)),
            &filter,
        );
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, Some("tenant-a"), adapter)),
            &filter,
        );

        let view = state.view(&filter);
        let membership = &view.endpoints[&EndpointId::from("prod.backend.generate")];
        assert!(!membership.is_materializable());
        assert!(membership.conflicts.iter().any(|conflict| {
            matches!(
                conflict,
                MaterializationConflict::Endpoint { reason, .. }
                    if reason.contains("tenant-a")
            )
        }));
    }

    #[test]
    fn invalid_alias_is_soft_and_does_not_block_the_endpoint() {
        let filter = DcDiscoveryFilter::default();
        let mut state = MembershipState::default();
        let mut llama = card("llama", "meta/llama", 64);
        llama.aliases = vec!["".to_string()];
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, None, llama)),
            &filter,
        );

        let view = state.view(&filter);
        let endpoint = EndpointId::from("prod.backend.generate");
        let membership = membership_for_endpoint(&view, &endpoint);
        assert!(membership.is_materializable());
        assert_eq!(membership.models, ["llama"]);
        assert!(membership.aliases.is_empty());
        assert!(membership.conflicts.is_empty());
    }
}
