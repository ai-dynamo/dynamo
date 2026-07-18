// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_kv_router::protocols::{WorkerId, WorkerWithDpRank};
use dynamo_runtime::{component::Instance, protocols::EndpointId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::local_model::runtime_config::ModelRuntimeConfig;

pub type PublisherId = u64;

/// Canonical identity of the logical KV source whose incarnations are reconciled.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvSourceKey {
    pub kv_state_endpoint: EndpointId,
    pub worker: WorkerWithDpRank,
}

impl KvSourceKey {
    pub fn new(kv_state_endpoint: EndpointId, worker: WorkerWithDpRank) -> Self {
        Self {
            kv_state_endpoint,
            worker,
        }
    }
}

/// Exact identity of one KV source incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvSourceId {
    pub key: KvSourceKey,
    pub publisher_id: PublisherId,
}

/// LLM-specific descriptor decoded from a generic event-source advertisement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvEventSource {
    pub kv_state_endpoint: EndpointId,
    pub worker: WorkerWithDpRank,
    pub publisher_id: PublisherId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_target: Option<Instance>,
}

impl KvEventSource {
    pub fn source_id(&self) -> KvSourceId {
        KvSourceId {
            key: self.source_key(),
            publisher_id: self.publisher_id,
        }
    }
}

/// Source input accepted by the pure membership reconciler.
///
/// Runtime discovery can implement this trait on an LLM adapter after decoding its opaque
/// metadata, while tests and in-process callers can use [`KvEventSource`] directly.
pub trait KvSourceAdvertisement: Clone + Eq {
    fn kv_state_endpoint(&self) -> &EndpointId;
    fn worker(&self) -> WorkerWithDpRank;
    fn publisher_id(&self) -> PublisherId;
    fn recovery_target(&self) -> Option<&Instance>;

    fn source_key(&self) -> KvSourceKey {
        KvSourceKey::new(self.kv_state_endpoint().clone(), self.worker())
    }
}

impl KvSourceAdvertisement for KvEventSource {
    fn kv_state_endpoint(&self) -> &EndpointId {
        &self.kv_state_endpoint
    }

    fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    fn publisher_id(&self) -> PublisherId {
        self.publisher_id
    }

    fn recovery_target(&self) -> Option<&Instance> {
        self.recovery_target.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvSourceAmbiguity {
    /// More than one random publisher incarnation advertises the same canonical source.
    Incarnations { publisher_ids: Vec<PublisherId> },
    /// One publisher ID advertised more than one immutable source descriptor.
    ConflictingDescriptor { publisher_id: PublisherId },
    /// Active base cards disagree about the effective KV-state endpoint.
    EndpointMapping { endpoints: Vec<EndpointId> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvSourceStatus<S = KvEventSource> {
    Missing,
    ActiveRecoverable(S),
    ActiveLiveOnly(S),
    Ambiguous(KvSourceAmbiguity),
}

impl<S> KvSourceStatus<S> {
    pub fn active_source(&self) -> Option<&S> {
        match self {
            Self::ActiveRecoverable(source) | Self::ActiveLiveOnly(source) => Some(source),
            Self::Missing | Self::Ambiguous(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSourceTransition<S = KvEventSource> {
    pub key: KvSourceKey,
    pub previous: KvSourceStatus<S>,
    pub current: KvSourceStatus<S>,
    /// The lifecycle owner must acknowledge a logical-rank reset before accepting the new state.
    pub requires_cold_reset: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvStateEndpointResolution {
    Resolved(EndpointId),
    Ambiguous { endpoints: Vec<EndpointId> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSourceMembershipView<S = KvEventSource> {
    /// The exact serving endpoint whose workers this view describes.
    pub serving_endpoint: EndpointId,
    pub endpoint_resolution: KvStateEndpointResolution,
    /// Membership remains indexed by logical worker and global DP rank. Publisher incarnation is
    /// deliberately absent from this routing/index identity.
    pub sources: HashMap<WorkerWithDpRank, KvSourceStatus<S>>,
    /// Monotonic cold-reset fence for each logical source in `sources`.
    ///
    /// A consumer must cold-reset a logical rank before accepting a source when this value
    /// differs from the last value it applied. Keeping the cumulative generation in every
    /// snapshot makes reset-relevant transitions observable even when a Tokio watch receiver
    /// coalesces intermediate snapshots.
    pub lifecycle_generations: HashMap<WorkerWithDpRank, u64>,
    /// Whether the serving runtime config expects a worker-local recovery target.
    /// This is expectation/readiness metadata only and never admits a serving worker.
    pub recovery_expected: HashMap<WorkerWithDpRank, bool>,
}

impl<S> KvSourceMembershipView<S> {
    pub fn status(&self, worker: &WorkerWithDpRank) -> Option<&KvSourceStatus<S>> {
        self.sources.get(worker)
    }

    pub fn lifecycle_generation(&self, worker: &WorkerWithDpRank) -> Option<u64> {
        self.lifecycle_generations.get(worker).copied()
    }

    pub fn recovery_expected(&self, worker: &WorkerWithDpRank) -> Option<bool> {
        self.recovery_expected.get(worker).copied()
    }

    pub fn resolved_kv_state_endpoint(&self) -> Option<&EndpointId> {
        match &self.endpoint_resolution {
            KvStateEndpointResolution::Resolved(endpoint) => Some(endpoint),
            KvStateEndpointResolution::Ambiguous { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KvSourceMembershipError {
    #[error(
        "publisher {publisher_id} changed its immutable KV source descriptor for worker {worker_id} rank {dp_rank}"
    )]
    ConflictingIncarnation {
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: u32,
    },
}

/// Reconciles advertised KV sources without coupling KV health to serving membership.
///
/// # Source-incarnation contract
///
/// - Incarnations may exist sequentially and may briefly overlap in discovery.
/// - Only one incarnation may be active for KV state.
/// - `publisher_id` is random, not ordered; its value never implies recency.
/// - Zero advertisements is missing, one is selectable, and more than one is ambiguous.
/// - Ambiguity fails KV handling closed until exactly one incarnation remains.
/// - Ordinary serving continues throughout missing and ambiguous KV states.
/// - Removal is incarnation-specific: removing publisher `100` cannot remove publisher `205`.
/// - Resolving ambiguity requires a cold reset before activating the sole remaining incarnation.
/// - The index stays keyed by logical `(WorkerId, dp_rank)`; incarnation is a lifecycle fence,
///   never a routing identity.
/// - Recovery targets are immutable within an incarnation. Replacing one replaces the entire
///   source incarnation.
/// - Supported multi-node restarts preserve the logical worker/rank key but create a new rank
///   publisher and recovery target. A future backend that resurrects cache state beneath a
///   surviving publisher must instead recreate that publisher or emit an ordered `Cleared`
///   barrier.
#[derive(Debug, Clone)]
pub struct KvSourceMembership<S = KvEventSource> {
    advertisements: HashMap<KvSourceKey, HashMap<PublisherId, S>>,
    conflicting_descriptors: HashSet<KvSourceId>,
}

impl<S> Default for KvSourceMembership<S> {
    fn default() -> Self {
        Self {
            advertisements: HashMap::new(),
            conflicting_descriptors: HashSet::new(),
        }
    }
}

impl<S> KvSourceMembership<S>
where
    S: KvSourceAdvertisement,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        source: S,
    ) -> Result<Option<KvSourceTransition<S>>, KvSourceMembershipError> {
        let key = source.source_key();
        let publisher_id = source.publisher_id();
        let previous = self.status(&key);

        let incarnations = self.advertisements.entry(key.clone()).or_default();
        if let Some(existing) = incarnations.get(&publisher_id) {
            if existing == &source {
                return Ok(None);
            }
            self.conflicting_descriptors.insert(KvSourceId {
                key: key.clone(),
                publisher_id,
            });
            return Err(KvSourceMembershipError::ConflictingIncarnation {
                publisher_id,
                worker_id: key.worker.worker_id,
                dp_rank: key.worker.dp_rank,
            });
        }
        incarnations.insert(publisher_id, source);

        let current = self.status(&key);
        Ok(Some(transition(key, previous, current)))
    }

    pub fn remove(&mut self, source_id: &KvSourceId) -> Option<KvSourceTransition<S>> {
        let previous = self.status(&source_id.key);
        let incarnations = self.advertisements.get_mut(&source_id.key)?;

        // Removal is fenced by the exact random incarnation; a delayed removal cannot erase its
        // replacement or otherwise mutate another publisher's lifecycle.
        incarnations.remove(&source_id.publisher_id)?;
        self.conflicting_descriptors.remove(source_id);
        if incarnations.is_empty() {
            self.advertisements.remove(&source_id.key);
        }

        let current = self.status(&source_id.key);
        Some(transition(source_id.key.clone(), previous, current))
    }

    /// Fail one logical source closed after its publisher violates immutable attribution.
    pub fn invalidate_descriptor(&mut self, source_id: KvSourceId) {
        self.conflicting_descriptors.insert(source_id);
    }

    pub fn status(&self, key: &KvSourceKey) -> KvSourceStatus<S> {
        if let Some(source_id) = self
            .conflicting_descriptors
            .iter()
            .find(|source_id| &source_id.key == key)
        {
            return KvSourceStatus::Ambiguous(KvSourceAmbiguity::ConflictingDescriptor {
                publisher_id: source_id.publisher_id,
            });
        }

        let Some(incarnations) = self.advertisements.get(key) else {
            return KvSourceStatus::Missing;
        };

        if incarnations.len() > 1 {
            let mut publisher_ids: Vec<_> = incarnations.keys().copied().collect();
            // Sorting is only for stable diagnostics/tests; no publisher is selected by value.
            publisher_ids.sort_unstable();
            return KvSourceStatus::Ambiguous(KvSourceAmbiguity::Incarnations { publisher_ids });
        }

        let Some(source) = incarnations.values().next().cloned() else {
            return KvSourceStatus::Missing;
        };
        if source.recovery_target().is_some() {
            KvSourceStatus::ActiveRecoverable(source)
        } else {
            KvSourceStatus::ActiveLiveOnly(source)
        }
    }

    /// Join source advertisements with the current serving/runtime-config snapshot.
    ///
    /// Only workers present in the supplied snapshot appear in the result. This keeps source-only
    /// advertisements from creating schedulable workers and lets serving continue independently.
    pub fn view(
        &self,
        serving_endpoint: &EndpointId,
        runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> KvSourceMembershipView<S> {
        let endpoint_resolution =
            resolve_kv_state_endpoint(serving_endpoint, runtime_configs.values());
        let workers: Vec<_> = runtime_configs
            .iter()
            .flat_map(|(&worker_id, config)| {
                (0..config.data_parallel_size).filter_map(move |offset| {
                    config
                        .data_parallel_start_rank
                        .checked_add(offset)
                        .map(|dp_rank| {
                            (
                                WorkerWithDpRank::new(worker_id, dp_rank),
                                config.enable_local_indexer,
                            )
                        })
                })
            })
            .collect();

        let sources: HashMap<WorkerWithDpRank, KvSourceStatus<S>> = match &endpoint_resolution {
            KvStateEndpointResolution::Resolved(kv_state_endpoint) => workers
                .iter()
                .copied()
                .map(|(worker, _)| {
                    let key = KvSourceKey::new(kv_state_endpoint.clone(), worker);
                    (worker, self.status(&key))
                })
                .collect(),
            KvStateEndpointResolution::Ambiguous { endpoints } => {
                let ambiguity = KvSourceAmbiguity::EndpointMapping {
                    endpoints: endpoints.clone(),
                };
                workers
                    .iter()
                    .copied()
                    .map(|(worker, _)| (worker, KvSourceStatus::Ambiguous(ambiguity.clone())))
                    .collect()
            }
        };
        let recovery_expected = workers
            .into_iter()
            .collect::<HashMap<WorkerWithDpRank, bool>>();

        KvSourceMembershipView {
            serving_endpoint: serving_endpoint.clone(),
            endpoint_resolution,
            lifecycle_generations: sources.keys().map(|worker| (*worker, 0)).collect(),
            recovery_expected,
            sources,
        }
    }
}

/// Resolve the effective KV-state endpoint advertised by active base runtime configs.
///
/// An omitted mapping and an explicit mapping to `serving_endpoint` are equal after fallback.
/// More than one effective endpoint fails only KV membership closed.
pub fn resolve_kv_state_endpoint<'a>(
    serving_endpoint: &EndpointId,
    runtime_configs: impl IntoIterator<Item = &'a ModelRuntimeConfig>,
) -> KvStateEndpointResolution {
    let mut endpoints: Vec<_> = runtime_configs
        .into_iter()
        .map(|config| config.effective_kv_state_endpoint(serving_endpoint))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if endpoints.is_empty() {
        return KvStateEndpointResolution::Resolved(serving_endpoint.clone());
    }
    if endpoints.len() == 1 {
        return KvStateEndpointResolution::Resolved(endpoints.pop().expect("endpoint exists"));
    }

    endpoints.sort_by(|left, right| {
        (&left.namespace, &left.component, &left.name).cmp(&(
            &right.namespace,
            &right.component,
            &right.name,
        ))
    });
    KvStateEndpointResolution::Ambiguous { endpoints }
}

fn transition<S>(
    key: KvSourceKey,
    previous: KvSourceStatus<S>,
    current: KvSourceStatus<S>,
) -> KvSourceTransition<S> {
    let requires_cold_reset = !matches!(
        (&previous, &current),
        (
            KvSourceStatus::Missing,
            KvSourceStatus::ActiveRecoverable(_) | KvSourceStatus::ActiveLiveOnly(_)
        )
    );
    KvSourceTransition {
        key,
        previous,
        current,
        requires_cold_reset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::component::TransportType;

    fn endpoint(name: &str) -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: name.to_string(),
        }
    }

    fn source(
        endpoint: &EndpointId,
        worker_id: WorkerId,
        rank: u32,
        publisher_id: u64,
    ) -> KvEventSource {
        KvEventSource {
            kv_state_endpoint: endpoint.clone(),
            worker: WorkerWithDpRank::new(worker_id, rank),
            publisher_id,
            recovery_target: None,
        }
    }

    fn recoverable_source(
        endpoint: &EndpointId,
        worker_id: WorkerId,
        rank: u32,
        publisher_id: u64,
    ) -> KvEventSource {
        KvEventSource {
            recovery_target: Some(Instance {
                component: "query".to_string(),
                endpoint: "rank".to_string(),
                namespace: "ns".to_string(),
                instance_id: publisher_id,
                transport: TransportType::Tcp("tcp://127.0.0.1:1234".to_string()),
                device_type: None,
            }),
            ..source(endpoint, worker_id, rank, publisher_id)
        }
    }

    #[test]
    fn effective_endpoint_fallback_agrees_with_explicit_serving_mapping() {
        let serving = endpoint("generate");
        let configs = [
            ModelRuntimeConfig::default(),
            ModelRuntimeConfig {
                kv_state_endpoint: Some(serving.clone()),
                ..Default::default()
            },
        ];

        assert_eq!(
            resolve_kv_state_endpoint(&serving, &configs),
            KvStateEndpointResolution::Resolved(serving)
        );
    }

    #[test]
    fn conflicting_effective_endpoints_fail_kv_membership_closed() {
        let serving = endpoint("generate");
        let other = endpoint("kv-events");
        let configs = HashMap::from([
            (7, ModelRuntimeConfig::default()),
            (
                8,
                ModelRuntimeConfig {
                    kv_state_endpoint: Some(other.clone()),
                    ..Default::default()
                },
            ),
        ]);
        let membership = KvSourceMembership::<KvEventSource>::new();
        let view = membership.view(&serving, &configs);

        assert_eq!(
            view.endpoint_resolution,
            KvStateEndpointResolution::Ambiguous {
                endpoints: vec![serving.clone(), other.clone()]
            }
        );
        for status in view.sources.values() {
            assert_eq!(
                status,
                &KvSourceStatus::Ambiguous(KvSourceAmbiguity::EndpointMapping {
                    endpoints: vec![serving.clone(), other.clone()]
                })
            );
        }
    }

    #[test]
    fn overlapping_random_incarnations_are_ambiguous_until_one_remains() {
        let kv_endpoint = endpoint("kv-events");
        let key = KvSourceKey::new(kv_endpoint.clone(), WorkerWithDpRank::new(7, 3));
        let old = source(&kv_endpoint, 7, 3, 100);
        let new = source(&kv_endpoint, 7, 3, 205);
        let mut membership = KvSourceMembership::new();

        let initial = membership.add(old.clone()).unwrap().unwrap();
        assert!(!initial.requires_cold_reset);
        assert_eq!(initial.current, KvSourceStatus::ActiveLiveOnly(old.clone()));

        let overlap = membership.add(new.clone()).unwrap().unwrap();
        assert!(overlap.requires_cold_reset);
        assert_eq!(
            overlap.current,
            KvSourceStatus::Ambiguous(KvSourceAmbiguity::Incarnations {
                publisher_ids: vec![100, 205]
            })
        );

        let resolved = membership.remove(&old.source_id()).unwrap();
        assert!(resolved.requires_cold_reset);
        assert_eq!(
            resolved.current,
            KvSourceStatus::ActiveLiveOnly(new.clone())
        );

        assert!(membership.remove(&old.source_id()).is_none());
        assert_eq!(membership.status(&key), KvSourceStatus::ActiveLiveOnly(new));
    }

    #[test]
    fn view_is_logically_keyed_and_does_not_admit_source_only_workers() {
        let serving = endpoint("generate");
        let kv_endpoint = endpoint("kv-events");
        let configs = HashMap::from([(
            7,
            ModelRuntimeConfig {
                data_parallel_start_rank: 2,
                data_parallel_size: 2,
                kv_state_endpoint: Some(kv_endpoint.clone()),
                ..Default::default()
            },
        )]);
        let active = recoverable_source(&kv_endpoint, 7, 2, 100);
        let source_only = source(&kv_endpoint, 99, 0, 205);
        let mut membership = KvSourceMembership::new();
        membership.add(active.clone()).unwrap();
        membership.add(source_only).unwrap();

        let view = membership.view(&serving, &configs);
        assert_eq!(view.sources.len(), 2);
        assert_eq!(
            view.status(&WorkerWithDpRank::new(7, 2)),
            Some(&KvSourceStatus::ActiveRecoverable(active))
        );
        assert_eq!(
            view.status(&WorkerWithDpRank::new(7, 3)),
            Some(&KvSourceStatus::Missing)
        );
        assert!(view.status(&WorkerWithDpRank::new(99, 0)).is_none());
    }
}
