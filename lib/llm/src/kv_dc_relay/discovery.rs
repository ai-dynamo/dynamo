// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
    ModelCardInstanceId,
};
use dynamo_runtime::protocols::EndpointId;
use futures::StreamExt;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::resolution::{ResolvedIndexerDomain, resolve_indexer_domain};
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;

const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);
const KV_EVENT_HASH_FORMAT_VERSION: u16 = 1;

pub(crate) type KvCacheDomainKey = ResolvedIndexerDomain;

#[derive(Debug, Clone, Default)]
pub(crate) struct DcDiscoveryFilter {
    pub(crate) namespace: Option<String>,
    pub(crate) endpoint_prefix: Option<String>,
}

impl DcDiscoveryFilter {
    fn matches(&self, endpoint: &EndpointId) -> bool {
        if self
            .namespace
            .as_ref()
            .is_some_and(|namespace| endpoint.namespace != *namespace)
        {
            return false;
        }
        self.endpoint_prefix.as_ref().is_none_or(|prefix| {
            format!(
                "{}.{}.{}",
                endpoint.namespace, endpoint.component, endpoint.name
            )
            .starts_with(prefix)
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EndpointMembership {
    pub(crate) endpoint: EndpointId,
    pub(crate) generation: u64,
    pub(crate) domain: Option<KvCacheDomainKey>,
    pub(crate) compatibility_conflict: bool,
    pub(crate) models: Vec<String>,
    pub(crate) aliases: Vec<String>,
    pub(crate) roles: Vec<String>,
    pub(crate) runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct DcMembershipView {
    pub(crate) endpoints: HashMap<EndpointId, EndpointMembership>,
}

pub(crate) struct DcMembershipWatch {
    receiver: watch::Receiver<DcMembershipView>,
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

impl DcMembershipWatch {
    pub(crate) async fn start(
        discovery: Arc<dyn Discovery>,
        filter: DcDiscoveryFilter,
        parent_cancel: CancellationToken,
    ) -> anyhow::Result<Self> {
        let initial = discovery.list(DiscoveryQuery::AllModels).await?;
        let mut state = MembershipState::default();
        state.replace_all(initial, &filter);
        let (sender, receiver) = watch::channel(state.view(&filter));
        let cancel = parent_cancel.child_token();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            run_membership_watch(discovery, filter, state, sender, task_cancel).await;
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

#[derive(Default)]
struct MembershipState {
    cards: HashMap<ModelCardInstanceId, ModelDeploymentCard>,
    endpoint_generations: HashMap<EndpointId, u64>,
    previous: HashMap<EndpointId, EndpointMembership>,
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
        let mut grouped: HashMap<EndpointId, Vec<(&ModelCardInstanceId, &ModelDeploymentCard)>> =
            HashMap::new();
        for (id, card) in &self.cards {
            let endpoint = endpoint_id(id);
            if filter.matches(&endpoint) {
                grouped.entry(endpoint).or_default().push((id, card));
            }
        }

        let mut endpoints = HashMap::new();
        for (endpoint, cards) in grouped {
            let mut domains = HashSet::new();
            let mut models = HashSet::new();
            let mut aliases = HashSet::new();
            let mut roles = HashSet::new();
            let mut runtime_configs = HashMap::new();

            for (id, card) in cards {
                models.insert(card.name().to_string());
                aliases.extend(card.aliases.iter().cloned());
                if let Some(role) = card.worker_type {
                    roles.insert(format!("{role:?}").to_lowercase());
                }
                if id.model_suffix.is_some() || card.lora.is_some() {
                    continue;
                }
                domains.insert(resolve_indexer_domain(
                    card,
                    &endpoint,
                    KV_EVENT_HASH_FORMAT_VERSION,
                ));
                runtime_configs.insert(id.instance_id, card.runtime_config.clone());
            }

            let compatibility_conflict = domains.len() > 1;
            let domain = (domains.len() == 1)
                .then(|| domains.into_iter().next())
                .flatten();
            let mut candidate = EndpointMembership {
                endpoint: endpoint.clone(),
                generation: 0,
                domain,
                compatibility_conflict,
                models: sorted(models),
                aliases: sorted(aliases),
                roles: sorted(roles),
                runtime_configs,
            };
            let changed = self
                .previous
                .get(&endpoint)
                .is_none_or(|previous| !same_membership(previous, &candidate));
            let generation = self
                .endpoint_generations
                .entry(endpoint.clone())
                .or_default();
            if changed {
                *generation = generation.saturating_add(1);
            }
            candidate.generation = *generation;
            endpoints.insert(endpoint, candidate);
        }

        self.endpoint_generations
            .retain(|endpoint, _| endpoints.contains_key(endpoint));
        self.previous = endpoints.clone();
        DcMembershipView { endpoints }
    }
}

async fn run_membership_watch(
    discovery: Arc<dyn Discovery>,
    filter: DcDiscoveryFilter,
    mut state: MembershipState,
    sender: watch::Sender<DcMembershipView>,
    cancel: CancellationToken,
) {
    let mut retry_delay = Duration::from_millis(100);
    loop {
        let stream_cancel = cancel.child_token();
        let stream = discovery
            .list_and_watch(DiscoveryQuery::AllModels, Some(stream_cancel.clone()))
            .await;
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                tracing::error!(%error, "Failed to watch DC-wide model-card membership");
                if !retry_or_cancel(retry_delay, &cancel).await {
                    return;
                }
                retry_delay = (retry_delay * 2).min(Duration::from_secs(5));
                continue;
            }
        };
        retry_delay = Duration::from_millis(100);
        let mut reconcile = tokio::time::interval(RECONCILE_INTERVAL);
        reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                event = stream.next() => match event {
                    Some(Ok(event)) => {
                        state.apply(event, &filter);
                        sender.send_replace(state.view(&filter));
                    }
                    Some(Err(error)) => {
                        tracing::error!(%error, "DC-wide model-card discovery stream failed; rebinding");
                        break;
                    }
                    None => break,
                },
                _ = reconcile.tick() => match discovery.list(DiscoveryQuery::AllModels).await {
                    Ok(instances) => {
                        state.replace_all(instances, &filter);
                        sender.send_replace(state.view(&filter));
                    }
                    Err(error) => tracing::warn!(%error, "Failed periodic KV DC Relay membership reconciliation"),
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

fn same_membership(left: &EndpointMembership, right: &EndpointMembership) -> bool {
    left.endpoint == right.endpoint
        && left.domain == right.domain
        && left.compatibility_conflict == right.compatibility_conflict
        && left.models == right.models
        && left.aliases == right.aliases
        && left.roles == right.roles
        && left.runtime_configs == right.runtime_configs
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

    #[test]
    fn exact_endpoint_membership_fences_conflicting_base_cards_but_ignores_lora_domains() {
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
                "embeddings",
                2,
                None,
                card("embed", "nvidia/embed", 32),
            )),
            &filter,
        );
        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                4,
                None,
                card("llama-public", "meta/llama", 64),
            )),
            &filter,
        );

        let view = state.view(&filter);
        assert_eq!(view.endpoints.len(), 2);
        let generate = &view.endpoints[&EndpointId::from("prod.backend.generate")];
        assert_eq!(
            generate.domain.as_ref().unwrap().diagnostic_model_artifact,
            "meta/llama"
        );
        assert!(!generate.compatibility_conflict);
        assert_eq!(generate.runtime_configs.len(), 2);
        assert_eq!(generate.models, vec!["llama", "llama-public"]);

        state.apply(
            DiscoveryEvent::Added(instance(
                "generate",
                3,
                None,
                card("other", "other/artifact", 64),
            )),
            &filter,
        );
        let view = state.view(&filter);
        let generate = &view.endpoints[&EndpointId::from("prod.backend.generate")];
        assert!(generate.compatibility_conflict);
        assert!(generate.domain.is_none());

        state.apply(
            DiscoveryEvent::Removed(DiscoveryInstanceId::Model(ModelCardInstanceId {
                namespace: "prod".to_string(),
                component: "backend".to_string(),
                endpoint: "generate".to_string(),
                instance_id: 3,
                model_suffix: None,
            })),
            &filter,
        );
        let mut adapter = card("llama-adapter", "unrelated/adapter", 1);
        adapter.lora = Some(LoraInfo {
            name: "tenant-a".to_string(),
            max_gpu_lora_count: None,
        });
        state.apply(
            DiscoveryEvent::Added(instance("generate", 1, Some("tenant-a"), adapter)),
            &filter,
        );

        let view = state.view(&filter);
        let generate = &view.endpoints[&EndpointId::from("prod.backend.generate")];
        assert!(!generate.compatibility_conflict);
        assert_eq!(
            generate.domain.as_ref().unwrap().diagnostic_model_artifact,
            "meta/llama"
        );
        assert_eq!(generate.runtime_configs.len(), 2);
        assert!(generate.models.iter().any(|model| model == "llama-adapter"));
    }
}
