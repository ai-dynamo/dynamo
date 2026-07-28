// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::{CkfConfig, ProducerIdentity};
use dynamo_runtime::protocols::EndpointId;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::actor::{ActorFault, KvDcRelayHandle, StreamScope};
use super::host::KvDcRelayError;
use super::identity::{
    CanonicalModelId, CanonicalModelRegistration, DcPoolCatalog, DcPoolDescriptor, ModelAlias,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct PoolActorConfig {
    pub(super) expected_unique_blocks: usize,
    pub(super) publication_threshold: usize,
    pub(super) publication_delay: Duration,
}

#[derive(Debug)]
pub(super) struct PoolAttachRequest {
    pub(super) pool_id: PoolId,
    pub(super) endpoint: EndpointId,
    pub(super) registrations: Vec<CanonicalModelRegistration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PoolRetirementMode {
    Graceful,
    Fenced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PoolEntryState {
    Active,
    Withdrawn,
    Fenced,
}

struct PoolEntry {
    endpoint: EndpointId,
    handle: KvDcRelayHandle,
    identity: ProducerIdentity,
    layout_generation: u64,
    registrations: Arc<[CanonicalModelRegistration]>,
    cancel: CancellationToken,
    state: PoolEntryState,
}

struct PoolReservation {
    endpoint: EndpointId,
    layout_generation: u64,
}

struct PoolRegistryState {
    pools: HashMap<PoolId, PoolEntry>,
    reservations: HashMap<PoolId, PoolReservation>,
    next_layout_generation: u64,
    catalog_revision: u64,
    accepting: bool,
}

impl Default for PoolRegistryState {
    fn default() -> Self {
        Self {
            pools: HashMap::new(),
            reservations: HashMap::new(),
            next_layout_generation: 1,
            catalog_revision: 0,
            accepting: true,
        }
    }
}

pub(super) struct PoolAttachment {
    pub(super) pool_id: PoolId,
    pub(super) layout_generation: u64,
    pub(super) handle: KvDcRelayHandle,
    registrations: Arc<[CanonicalModelRegistration]>,
    pub(super) faults: mpsc::Receiver<ActorFault>,
    pub(super) pool_cancel: CancellationToken,
}

pub(super) struct PoolRegistry {
    process_incarnation: u64,
    actor_config: PoolActorConfig,
    state: Mutex<PoolRegistryState>,
    catalog_tx: watch::Sender<DcPoolCatalog>,
}

impl PoolRegistry {
    pub(super) fn new(process_incarnation: u64, actor_config: PoolActorConfig) -> Self {
        let (catalog_tx, _) =
            watch::channel(DcPoolCatalog::new(process_incarnation, 0, Vec::new()));
        Self {
            process_incarnation,
            actor_config,
            state: Mutex::new(PoolRegistryState::default()),
            catalog_tx,
        }
    }

    pub(super) async fn attach(
        &self,
        request: PoolAttachRequest,
    ) -> anyhow::Result<PoolAttachment> {
        anyhow::ensure!(
            !request.registrations.is_empty(),
            "pool {} requires at least one canonical model binding",
            request.pool_id
        );
        validate_registrations(&request.registrations)?;

        let layout_generation = {
            let mut state = self.state.lock();
            anyhow::ensure!(
                state.accepting,
                "KV DC Relay pool registry is shutting down"
            );
            if let Some(endpoint) = pool_owner(&state, request.pool_id) {
                anyhow::bail!(
                    "pool {} is already owned by endpoint {} and cannot also attach endpoint {}",
                    request.pool_id,
                    endpoint,
                    request.endpoint
                );
            }
            let layout_generation = allocate_layout_generation(&mut state)?;
            state.reservations.insert(
                request.pool_id,
                PoolReservation {
                    endpoint: request.endpoint.clone(),
                    layout_generation,
                },
            );
            layout_generation
        };

        // Keep allocation outside the registry mutex. There is deliberately no `.await` between
        // reservation and commit, so task cancellation cannot strand a live reservation.
        let mut config = CkfConfig::new(self.actor_config.expected_unique_blocks);
        config.publish_every_n_events = self.actor_config.publication_threshold;
        let actor = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            StreamScope {
                process_incarnation: self.process_incarnation,
                layout_generation,
                pool_id: request.pool_id,
            },
            self.actor_config.publication_delay,
        );
        let (handle, faults) = match actor {
            Ok(actor) => actor,
            Err(error) => {
                let mut state = self.state.lock();
                rollback_reservation(&mut state, request.pool_id, layout_generation);
                return Err(error.into());
            }
        };
        let identity = handle.identity();
        let registrations: Arc<[CanonicalModelRegistration]> = request.registrations.into();
        let cancel = CancellationToken::new();

        let mut state = self.state.lock();
        let reservation_matches = state
            .reservations
            .get(&request.pool_id)
            .is_some_and(|reservation| reservation.layout_generation == layout_generation);
        if !state.accepting || !reservation_matches {
            rollback_reservation(&mut state, request.pool_id, layout_generation);
            anyhow::bail!(
                "pool {} generation {} reservation was retired before commit",
                request.pool_id,
                layout_generation
            );
        }
        state.reservations.remove(&request.pool_id);
        debug_assert!(!state.pools.contains_key(&request.pool_id));
        state.pools.insert(
            request.pool_id,
            PoolEntry {
                endpoint: request.endpoint.clone(),
                handle: handle.clone(),
                identity,
                layout_generation,
                registrations: registrations.clone(),
                cancel: cancel.clone(),
                state: PoolEntryState::Active,
            },
        );
        publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);

        Ok(PoolAttachment {
            pool_id: request.pool_id,
            layout_generation,
            handle,
            registrations,
            faults,
            pool_cancel: cancel,
        })
    }

    pub(super) async fn detach(&self, attachment: PoolAttachment) -> Result<(), KvDcRelayError> {
        let PoolAttachment {
            pool_id,
            layout_generation,
            handle,
            mut faults,
            ..
        } = attachment;
        self.withdraw(pool_id, layout_generation, PoolRetirementMode::Graceful)
            .await;
        let result = drain_faults_while(pool_id, &mut faults, handle.shutdown()).await;
        self.remove(pool_id, layout_generation).await;
        result
    }

    pub(super) async fn replace_registrations(
        &self,
        attachment: &mut PoolAttachment,
        registrations: Vec<CanonicalModelRegistration>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !registrations.is_empty(),
            "pool {} requires at least one canonical model binding",
            attachment.pool_id
        );
        if attachment.registrations.as_ref() == registrations.as_slice() {
            return Ok(());
        }

        validate_registrations(&registrations)?;
        let registrations: Arc<[CanonicalModelRegistration]> = registrations.into();
        let mut state = self.state.lock();
        let entry = state
            .pools
            .get_mut(&attachment.pool_id)
            .ok_or_else(|| anyhow::anyhow!("pool {} is not attached", attachment.pool_id))?;
        anyhow::ensure!(
            entry.layout_generation == attachment.layout_generation
                && entry.state == PoolEntryState::Active,
            "pool {} generation {} is no longer active",
            attachment.pool_id,
            attachment.layout_generation
        );
        entry.registrations = registrations.clone();
        attachment.registrations = registrations;
        publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
        Ok(())
    }

    pub(super) async fn withdraw(
        &self,
        pool_id: PoolId,
        layout_generation: u64,
        mode: PoolRetirementMode,
    ) -> bool {
        let mut state = self.state.lock();
        let Some(entry) = state.pools.get_mut(&pool_id) else {
            return false;
        };
        if entry.layout_generation != layout_generation {
            return false;
        }
        let was_active = entry.state == PoolEntryState::Active;
        entry.state = match (entry.state, mode) {
            (PoolEntryState::Fenced, _) | (_, PoolRetirementMode::Fenced) => PoolEntryState::Fenced,
            _ => PoolEntryState::Withdrawn,
        };
        entry.cancel.cancel();
        if was_active {
            publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
        }
        true
    }

    pub(super) async fn remove(&self, pool_id: PoolId, layout_generation: u64) -> bool {
        let mut state = self.state.lock();
        let Some(entry) = state.pools.get(&pool_id) else {
            return false;
        };
        if entry.layout_generation != layout_generation {
            return false;
        }
        let was_active = entry.state == PoolEntryState::Active;
        let Some(entry) = state.pools.remove(&pool_id) else {
            return false;
        };
        entry.cancel.cancel();
        if was_active {
            publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
        }
        true
    }

    pub(super) fn catalog(&self) -> DcPoolCatalog {
        self.catalog_tx.borrow().clone()
    }

    pub(super) fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.catalog_tx.subscribe()
    }

    pub(super) async fn shutdown(&self) {
        let entries = {
            let mut state = self.state.lock();
            state.accepting = false;
            state.reservations.clear();
            let entries = state.pools.drain().collect::<Vec<_>>();
            publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
            entries
        };
        for (pool_id, entry) in entries {
            entry.cancel.cancel();
            if let Err(error) = entry.handle.fence().await {
                tracing::warn!(%pool_id, %error, "KV Relay pool actor failed to fence during registry shutdown");
            }
        }
    }

    #[cfg(test)]
    pub(super) async fn pool_count(&self) -> usize {
        self.state.lock().pools.len()
    }
}

pub(super) async fn drain_faults_while<T>(
    pool_id: PoolId,
    faults: &mut mpsc::Receiver<ActorFault>,
    future: impl Future<Output = T>,
) -> T {
    tokio::pin!(future);
    loop {
        tokio::select! {
            result = &mut future => return result,
            fault = faults.recv() => match fault {
                Some(fault) => tracing::debug!(
                    %pool_id,
                    worker_id = fault.worker_id,
                    dp_rank = fault.dp_rank,
                    category = ?fault.category,
                    error = %fault.message,
                    "Draining KV DC Relay actor fault while retiring its pool"
                ),
                None => return future.await,
            },
        }
    }
}

fn allocate_layout_generation(state: &mut PoolRegistryState) -> anyhow::Result<u64> {
    let generation = state.next_layout_generation;
    state.next_layout_generation = generation
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("KV DC Relay layout generation space exhausted"))?;
    Ok(generation)
}

fn pool_owner(state: &PoolRegistryState, pool_id: PoolId) -> Option<&EndpointId> {
    state
        .pools
        .get(&pool_id)
        .map(|entry| &entry.endpoint)
        .or_else(|| {
            state
                .reservations
                .get(&pool_id)
                .map(|reservation| &reservation.endpoint)
        })
}

fn rollback_reservation(state: &mut PoolRegistryState, pool_id: PoolId, layout_generation: u64) {
    if state
        .reservations
        .get(&pool_id)
        .is_some_and(|reservation| reservation.layout_generation == layout_generation)
    {
        state.reservations.remove(&pool_id);
    }
}

fn publish_catalog(
    state: &mut PoolRegistryState,
    process_incarnation: u64,
    sender: &watch::Sender<DcPoolCatalog>,
) {
    state.catalog_revision = state.catalog_revision.saturating_add(1);
    sender.send_replace(catalog_from_state(state, process_incarnation));
}

fn catalog_from_state(state: &PoolRegistryState, process_incarnation: u64) -> DcPoolCatalog {
    let mut pools: Vec<_> = state
        .pools
        .values()
        .filter(|entry| entry.state == PoolEntryState::Active)
        .map(|entry| {
            DcPoolDescriptor::new(
                entry.identity,
                entry.endpoint.clone(),
                entry.registrations.to_vec(),
            )
        })
        .collect();
    pools.sort_unstable_by_key(DcPoolDescriptor::pool_id);

    DcPoolCatalog::new(process_incarnation, state.catalog_revision, pools)
}

fn validate_registrations(registrations: &[CanonicalModelRegistration]) -> anyhow::Result<()> {
    let mut request_models = HashMap::with_capacity(registrations.len());
    for registration in registrations {
        if let Some(previous) =
            request_models.insert(registration.model().clone(), registration.target().clone())
        {
            anyhow::ensure!(
                previous == *registration.target(),
                "canonical model {} resolves to conflicting targets in the same pool",
                registration.model()
            );
            anyhow::bail!(
                "duplicate canonical model registration {}",
                registration.model()
            );
        }
    }

    let mut request_aliases = HashMap::<ModelAlias, CanonicalModelId>::new();
    for registration in registrations {
        for alias in registration.aliases() {
            let alias_as_model = CanonicalModelId::new(alias.as_str().to_string())?;
            anyhow::ensure!(
                !request_models.contains_key(&alias_as_model),
                "model alias {} conflicts with canonical model {} in the same pool",
                alias,
                alias_as_model
            );
            if let Some(owner) = request_aliases.insert(alias.clone(), registration.model().clone())
            {
                anyhow::ensure!(
                    owner == *registration.model(),
                    "model alias {} is claimed by both {} and {} in the same pool",
                    alias,
                    owner,
                    registration.model()
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, RoutingScopeId,
    };

    use super::*;
    use crate::kv_dc_relay::identity::ModelTarget;

    fn pool(seed: u8) -> PoolId {
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([seed; 16], IdentitySource::Explicit),
                RoutingScopeId::new([seed.wrapping_add(1); 16], IdentitySource::Explicit),
            ),
            DcId::new(3),
        )
    }

    fn config() -> PoolActorConfig {
        PoolActorConfig {
            expected_unique_blocks: 32,
            publication_threshold: 1,
            publication_delay: Duration::from_millis(1),
        }
    }

    fn registration(model: &str) -> CanonicalModelRegistration {
        CanonicalModelRegistration::new(
            CanonicalModelId::new(model).unwrap(),
            vec![ModelAlias::new(format!("{model}-alias")).unwrap()],
        )
    }

    fn request(pool_id: PoolId, endpoint: &str, model: &str) -> PoolAttachRequest {
        PoolAttachRequest {
            pool_id,
            endpoint: EndpointId::from(endpoint),
            registrations: vec![registration(model)],
        }
    }

    fn descriptor(catalog: &DcPoolCatalog, pool_id: PoolId) -> &DcPoolDescriptor {
        catalog
            .pools()
            .iter()
            .find(|descriptor| descriptor.pool_id() == pool_id)
            .unwrap()
    }

    async fn retire(registry: &PoolRegistry, attachment: PoolAttachment, mode: PoolRetirementMode) {
        let PoolAttachment {
            pool_id,
            layout_generation,
            handle,
            mut faults,
            ..
        } = attachment;
        assert!(registry.withdraw(pool_id, layout_generation, mode).await);
        let result = match mode {
            PoolRetirementMode::Graceful => {
                drain_faults_while(pool_id, &mut faults, handle.shutdown()).await
            }
            PoolRetirementMode::Fenced => {
                drain_faults_while(pool_id, &mut faults, handle.fence()).await
            }
        };
        result.unwrap();
        assert!(registry.remove(pool_id, layout_generation).await);
    }

    #[tokio::test]
    async fn one_model_binds_to_independent_pools() {
        let registry = PoolRegistry::new(7, config());
        let first = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        let second = registry
            .attach(request(pool(2), "slow.router.generate", "llama"))
            .await
            .unwrap();

        assert_eq!(registry.pool_count().await, 2);
        let catalog = registry.catalog();
        assert_eq!(catalog.process_incarnation(), 7);
        assert_eq!(catalog.pools().len(), 2);
        assert_eq!(
            descriptor(&catalog, pool(1)).serving_endpoint(),
            &EndpointId::from("fast.router.generate")
        );
        assert!(
            catalog
                .pools()
                .iter()
                .all(|descriptor| { descriptor.registrations()[0].model().as_str() == "llama" })
        );

        registry.detach(first).await.unwrap();
        registry.detach(second).await.unwrap();
    }

    #[tokio::test]
    async fn registration_updates_remain_pool_local() {
        let registry = PoolRegistry::new(7, config());
        let mut first = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        let mut second = registry
            .attach(request(pool(2), "slow.router.generate", "mistral"))
            .await
            .unwrap();

        registry
            .replace_registrations(
                &mut first,
                vec![CanonicalModelRegistration::new(
                    CanonicalModelId::new("llama").unwrap(),
                    vec![ModelAlias::new("mistral-alias").unwrap()],
                )],
            )
            .await
            .unwrap();
        let catalog = registry.catalog();
        assert!(catalog.pools().iter().all(|descriptor| {
            descriptor.registrations()[0].aliases() == [ModelAlias::new("mistral-alias").unwrap()]
        }));

        registry
            .replace_registrations(
                &mut second,
                vec![CanonicalModelRegistration::new(
                    CanonicalModelId::new("mistral").unwrap(),
                    vec![ModelAlias::new("llama-alias").unwrap()],
                )],
            )
            .await
            .unwrap();
        let catalog = registry.catalog();
        assert_eq!(
            descriptor(&catalog, pool(1)).registrations()[0].aliases(),
            [ModelAlias::new("mistral-alias").unwrap()]
        );
        assert_eq!(
            descriptor(&catalog, pool(2)).registrations()[0].aliases(),
            [ModelAlias::new("llama-alias").unwrap()]
        );

        registry.detach(first).await.unwrap();
        registry.detach(second).await.unwrap();
    }

    #[tokio::test]
    async fn one_pool_cannot_be_owned_by_two_endpoints() {
        let registry = PoolRegistry::new(7, config());
        let pool_id = pool(1);
        let attachment = registry
            .attach(request(pool_id, "first.router.generate", "llama"))
            .await
            .unwrap();

        let error = registry
            .attach(request(pool_id, "second.router.generate", "llama"))
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("already owned"));

        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn failed_actor_build_rolls_back_the_pool_reservation() {
        let registry = PoolRegistry::new(
            7,
            PoolActorConfig {
                expected_unique_blocks: 0,
                publication_threshold: 1,
                publication_delay: Duration::from_millis(1),
            },
        );
        let pool_id = pool(1);

        let first_error = registry
            .attach(request(pool_id, "first.router.generate", "llama"))
            .await
            .err()
            .unwrap();
        let second_error = registry
            .attach(request(pool_id, "second.router.generate", "llama"))
            .await
            .err()
            .unwrap();

        assert!(first_error.to_string().contains("greater than zero"));
        assert!(second_error.to_string().contains("greater than zero"));
        assert_eq!(registry.pool_count().await, 0);
        assert_eq!(registry.catalog().revision(), 0);
    }

    #[tokio::test]
    async fn reattaching_a_pool_allocates_a_new_layout_generation() {
        let registry = PoolRegistry::new(7, config());
        let pool_id = pool(1);
        let first = registry
            .attach(request(pool_id, "fast.router.generate", "llama"))
            .await
            .unwrap();
        let first_generation = first.layout_generation;
        registry.detach(first).await.unwrap();

        let replacement = registry
            .attach(request(pool_id, "fast.router.generate", "llama"))
            .await
            .unwrap();
        assert!(replacement.layout_generation > first_generation);

        registry.detach(replacement).await.unwrap();
    }

    #[tokio::test]
    async fn withdraw_removes_catalog_before_actor_retirement() {
        let registry = PoolRegistry::new(7, config());
        let attachment = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();

        assert!(
            registry
                .withdraw(
                    attachment.pool_id,
                    attachment.layout_generation,
                    PoolRetirementMode::Graceful,
                )
                .await
        );
        assert!(registry.catalog().pools().is_empty());
        attachment.handle.state_stats().await.unwrap();

        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn adapter_registration_changes_without_replacing_pool_generation() {
        let registry = PoolRegistry::new(7, config());
        let mut attachment = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        let generation = attachment.layout_generation;
        let base = CanonicalModelId::new("llama").unwrap();
        let adapter = CanonicalModelId::new("tenant-a").unwrap();
        registry
            .replace_registrations(
                &mut attachment,
                vec![
                    CanonicalModelRegistration::new(base.clone(), Vec::new()),
                    CanonicalModelRegistration::with_target(
                        adapter.clone(),
                        ModelTarget::Lora {
                            base_model: base,
                            adapter: adapter.clone(),
                        },
                        Vec::new(),
                    ),
                ],
            )
            .await
            .unwrap();

        assert_eq!(attachment.layout_generation, generation);
        let catalog = registry.catalog();
        assert!(
            descriptor(&catalog, pool(1))
                .registrations()
                .iter()
                .any(|registration| registration.model() == &adapter)
        );
        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn one_lora_target_binds_to_independent_pools() {
        let registry = PoolRegistry::new(7, config());
        let base = CanonicalModelId::new("llama").unwrap();
        let adapter = CanonicalModelId::new("tenant-a").unwrap();
        let registrations = || {
            vec![
                CanonicalModelRegistration::new(base.clone(), Vec::new()),
                CanonicalModelRegistration::with_target(
                    adapter.clone(),
                    ModelTarget::Lora {
                        base_model: base.clone(),
                        adapter: adapter.clone(),
                    },
                    Vec::new(),
                ),
            ]
        };
        let first = registry
            .attach(PoolAttachRequest {
                pool_id: pool(1),
                endpoint: EndpointId::from("fast.router.generate"),
                registrations: registrations(),
            })
            .await
            .unwrap();
        let second = registry
            .attach(PoolAttachRequest {
                pool_id: pool(2),
                endpoint: EndpointId::from("slow.router.generate"),
                registrations: registrations(),
            })
            .await
            .unwrap();

        let catalog = registry.catalog();
        assert_eq!(catalog.pools().len(), 2);
        assert!(catalog.pools().iter().all(|descriptor| {
            descriptor.registrations().iter().any(|registration| {
                registration.target()
                    == &ModelTarget::Lora {
                        base_model: base.clone(),
                        adapter: adapter.clone(),
                    }
            })
        }));

        registry.detach(first).await.unwrap();
        registry.detach(second).await.unwrap();
    }

    #[tokio::test]
    async fn fencing_withdraws_pool_from_catalog() {
        let registry = PoolRegistry::new(7, config());
        let attachment = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        retire(&registry, attachment, PoolRetirementMode::Fenced).await;
        assert!(registry.watch_catalog().borrow().pools().is_empty());
    }

    #[tokio::test]
    async fn fencing_withdraws_only_the_target_pool_descriptor() {
        let registry = PoolRegistry::new(7, config());
        let with_alias = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        let without_alias = registry
            .attach(PoolAttachRequest {
                pool_id: pool(2),
                endpoint: EndpointId::from("slow.router.generate"),
                registrations: vec![CanonicalModelRegistration::new(
                    CanonicalModelId::new("llama").unwrap(),
                    Vec::new(),
                )],
            })
            .await
            .unwrap();

        let catalog = registry.catalog();
        assert_eq!(
            descriptor(&catalog, pool(1)).registrations()[0]
                .aliases()
                .len(),
            1
        );
        assert!(
            descriptor(&catalog, pool(2)).registrations()[0]
                .aliases()
                .is_empty()
        );
        retire(&registry, with_alias, PoolRetirementMode::Fenced).await;
        let catalog = registry.catalog();
        assert_eq!(catalog.pools().len(), 1);
        assert_eq!(catalog.pools()[0].pool_id(), pool(2));
        assert!(catalog.pools()[0].registrations()[0].aliases().is_empty());

        registry.detach(without_alias).await.unwrap();
    }
}
