// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::{CkfBuildError, CkfConfig, DcCkfState, ProducerIdentity};
use dynamo_kv_router::protocols::{ActiveLoad, WorkerId};
use dynamo_runtime::protocols::EndpointId;
use parking_lot::Mutex;
use tokio::sync::{Semaphore, mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::actor::{ActorFault, KvDcRelayHandle, StreamScope};
use super::host::KvDcRelayError;
use super::identity::{
    CanonicalModelId, CanonicalModelRegistration, DcPoolCatalog, DcPoolDescriptor, DcRelayIdentity,
    KvQuerySemantics, ModelAlias, WorkerRole,
};
use super::load::{PoolLoadSnapshot, PoolLoadState};
use crate::local_model::runtime_config::ModelRuntimeConfig;

const DEFAULT_CKF_ALLOCATION_CONCURRENCY: usize = 2;

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
    pub(super) query_semantics: KvQuerySemantics,
    pub(super) roles: Vec<WorkerRole>,
    pub(super) serving_facts: Option<PoolServingFacts>,
}

#[derive(Debug, Clone)]
pub(super) struct PoolServingFacts {
    pub(super) runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
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
    query_semantics: KvQuerySemantics,
    roles: Arc<[WorkerRole]>,
    cancel: CancellationToken,
    state: PoolEntryState,
    serving: Option<PoolServingState>,
}

struct PoolServingState {
    load: PoolLoadState,
}

struct PoolReservation {
    endpoint: EndpointId,
    layout_generation: u64,
}

struct PoolReservationGuard<'a> {
    state: &'a Mutex<PoolRegistryState>,
    pool_id: PoolId,
    layout_generation: u64,
    is_armed: bool,
}

impl<'a> PoolReservationGuard<'a> {
    fn new(state: &'a Mutex<PoolRegistryState>, pool_id: PoolId, layout_generation: u64) -> Self {
        Self {
            state,
            pool_id,
            layout_generation,
            is_armed: true,
        }
    }

    fn disarm(&mut self) {
        self.is_armed = false;
    }
}

impl Drop for PoolReservationGuard<'_> {
    fn drop(&mut self) {
        if self.is_armed {
            rollback_reservation(&mut self.state.lock(), self.pool_id, self.layout_generation);
        }
    }
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
    roles: Arc<[WorkerRole]>,
    pub(super) faults: mpsc::Receiver<ActorFault>,
    pub(super) pool_cancel: CancellationToken,
}

pub(super) struct PoolRegistry {
    relay_identity: DcRelayIdentity,
    actor_config: PoolActorConfig,
    ckf_allocation_permits: Arc<Semaphore>,
    state: Mutex<PoolRegistryState>,
    catalog_tx: watch::Sender<DcPoolCatalog>,
    load_tx: watch::Sender<Vec<PoolLoadSnapshot>>,
}

impl PoolRegistry {
    pub(super) fn new(relay_identity: DcRelayIdentity, actor_config: PoolActorConfig) -> Self {
        let (catalog_tx, _) = watch::channel(DcPoolCatalog::new(relay_identity, 0, Vec::new()));
        let (load_tx, _) = watch::channel(Vec::new());
        Self {
            relay_identity,
            actor_config,
            ckf_allocation_permits: Arc::new(Semaphore::new(DEFAULT_CKF_ALLOCATION_CONCURRENCY)),
            state: Mutex::new(PoolRegistryState::default()),
            catalog_tx,
            load_tx,
        }
    }

    pub(super) async fn attach(
        &self,
        request: PoolAttachRequest,
    ) -> anyhow::Result<PoolAttachment> {
        self.attach_with_builder(request, DcCkfState::new).await
    }

    async fn attach_with_builder<Builder>(
        &self,
        request: PoolAttachRequest,
        builder: Builder,
    ) -> anyhow::Result<PoolAttachment>
    where
        Builder: FnOnce(CkfConfig) -> Result<DcCkfState, CkfBuildError> + Send + 'static,
    {
        anyhow::ensure!(
            !request.registrations.is_empty(),
            "pool {} requires at least one canonical model binding",
            request.pool_id
        );
        validate_registrations(&request.registrations)?;
        validate_roles(&request.roles)?;
        let serving = request
            .serving_facts
            .as_ref()
            .map(|facts| -> anyhow::Result<PoolServingState> {
                Ok(PoolServingState {
                    load: PoolLoadState::from_runtime_configs(&facts.runtime_configs)?,
                })
            })
            .transpose()?;

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
            if let Some(pool_id) = endpoint_pool(&state, &request.endpoint) {
                anyhow::bail!(
                    "endpoint {} is already assigned to pool {} and cannot attach pool {}",
                    request.endpoint,
                    pool_id,
                    request.pool_id
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
        let mut reservation =
            PoolReservationGuard::new(&self.state, request.pool_id, layout_generation);

        let mut config = CkfConfig::new(self.actor_config.expected_unique_blocks);
        config.publish_every_n_events = self.actor_config.publication_threshold;
        let permit = self
            .ckf_allocation_permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow::anyhow!("KV DC Relay pool registry is shutting down"))?;
        let ckf_state = tokio::task::spawn_blocking(move || {
            let result = builder(config);
            drop(permit);
            result
        })
        .await
        .map_err(|error| anyhow::anyhow!("KV DC Relay CKF allocation task failed: {error}"))??;

        let registrations: Arc<[CanonicalModelRegistration]> = request.registrations.into();
        let roles: Arc<[WorkerRole]> = request.roles.into();
        let cancel = CancellationToken::new();

        let mut state = self.state.lock();
        let reservation_matches = state
            .reservations
            .get(&request.pool_id)
            .is_some_and(|reservation| reservation.layout_generation == layout_generation);
        anyhow::ensure!(
            state.accepting && reservation_matches,
            "pool {} generation {} reservation was retired before commit",
            request.pool_id,
            layout_generation
        );
        let (handle, faults) = KvDcRelayHandle::spawn_with_state_and_publication_delay(
            ckf_state,
            StreamScope {
                relay_incarnation: self.relay_identity.relay_incarnation(),
                layout_generation,
                pool_id: request.pool_id,
            },
            self.actor_config.publication_delay,
        );
        let identity = handle.identity();
        let descriptor = DcPoolDescriptor::new(
            identity,
            request.endpoint.clone(),
            registrations.clone(),
            request.query_semantics,
            roles.clone(),
        );
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
                query_semantics: request.query_semantics,
                roles: roles.clone(),
                cancel: cancel.clone(),
                state: PoolEntryState::Active,
                serving,
            },
        );
        publish_catalog_upsert(&mut state, &self.catalog_tx, descriptor);
        {
            publish_load_if_changed(&state, &self.load_tx, request.pool_id);
        }
        reservation.disarm();

        Ok(PoolAttachment {
            pool_id: request.pool_id,
            layout_generation,
            handle,
            registrations,
            roles,
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
        let descriptor = DcPoolDescriptor::new(
            entry.identity,
            entry.endpoint.clone(),
            registrations.clone(),
            entry.query_semantics,
            entry.roles.clone(),
        );
        attachment.registrations = registrations;
        publish_catalog_upsert(&mut state, &self.catalog_tx, descriptor);
        Ok(())
    }

    pub(super) fn replace_roles(
        &self,
        attachment: &mut PoolAttachment,
        roles: Vec<WorkerRole>,
    ) -> anyhow::Result<()> {
        validate_roles(&roles)?;
        if attachment.roles.as_ref() == roles.as_slice() {
            return Ok(());
        }

        let roles: Arc<[WorkerRole]> = roles.into();
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
        entry.roles = roles.clone();
        let descriptor = DcPoolDescriptor::new(
            entry.identity,
            entry.endpoint.clone(),
            entry.registrations.clone(),
            entry.query_semantics,
            roles.clone(),
        );
        attachment.roles = roles;
        publish_catalog_upsert(&mut state, &self.catalog_tx, descriptor);
        Ok(())
    }

    pub(super) fn replace_load_capacity(
        &self,
        pool_id: PoolId,
        layout_generation: u64,
        runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> anyhow::Result<bool> {
        let mut state = self.state.lock();
        let Some(entry) = state.pools.get_mut(&pool_id) else {
            return Ok(false);
        };
        if entry.layout_generation != layout_generation || entry.state != PoolEntryState::Active {
            return Ok(false);
        }
        let Some(serving) = entry.serving.as_mut() else {
            return Ok(false);
        };
        if serving.load.replace_capacity(runtime_configs)? {
            publish_load_if_changed(&state, &self.load_tx, pool_id);
        }
        Ok(true)
    }

    pub(super) fn observe_load(
        &self,
        pool_id: PoolId,
        layout_generation: u64,
        load: ActiveLoad,
    ) -> bool {
        let mut state = self.state.lock();
        let Some(entry) = state.pools.get_mut(&pool_id) else {
            return false;
        };
        if entry.layout_generation != layout_generation || entry.state != PoolEntryState::Active {
            return false;
        }
        let Some(serving) = entry.serving.as_mut() else {
            return false;
        };
        if !serving.load.observe(load) {
            return false;
        }
        publish_load_if_changed(&state, &self.load_tx, pool_id);
        true
    }

    pub(super) fn clear_load_observations(&self, pool_id: PoolId, layout_generation: u64) -> bool {
        let mut state = self.state.lock();
        let Some(entry) = state.pools.get_mut(&pool_id) else {
            return false;
        };
        if entry.layout_generation != layout_generation || entry.state != PoolEntryState::Active {
            return false;
        }
        let Some(serving) = entry.serving.as_mut() else {
            return false;
        };
        if serving.load.clear_observations() {
            publish_load_if_changed(&state, &self.load_tx, pool_id);
        }
        true
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
            publish_catalog_remove(&mut state, &self.catalog_tx, pool_id);
            publish_load_if_changed(&state, &self.load_tx, pool_id);
        }
        true
    }

    pub(super) async fn remove(&self, pool_id: PoolId, layout_generation: u64) -> bool {
        let entry = {
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
                publish_catalog_remove(&mut state, &self.catalog_tx, pool_id);
            }
            {
                publish_load_if_changed(&state, &self.load_tx, pool_id);
            }
            entry
        };
        drop(entry);
        true
    }

    pub(super) fn catalog(&self) -> DcPoolCatalog {
        self.catalog_tx.borrow().clone()
    }

    pub(super) fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.catalog_tx.subscribe()
    }

    pub(super) fn load_snapshots(&self) -> Vec<PoolLoadSnapshot> {
        self.load_tx.borrow().clone()
    }

    pub(super) fn watch_load(&self) -> watch::Receiver<Vec<PoolLoadSnapshot>> {
        self.load_tx.subscribe()
    }

    pub(super) async fn shutdown(&self) {
        let entries = {
            let mut state = self.state.lock();
            state.accepting = false;
            self.ckf_allocation_permits.close();
            state.reservations.clear();
            let entries = state.pools.drain().collect::<Vec<_>>();
            publish_catalog_clear(&mut state, &self.catalog_tx);
            {
                self.load_tx.send_if_modified(|snapshots| {
                    if snapshots.is_empty() {
                        return false;
                    }
                    snapshots.clear();
                    true
                });
            }
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

fn endpoint_pool(state: &PoolRegistryState, endpoint: &EndpointId) -> Option<PoolId> {
    state
        .pools
        .iter()
        .find_map(|(&pool_id, entry)| (&entry.endpoint == endpoint).then_some(pool_id))
        .or_else(|| {
            state
                .reservations
                .iter()
                .find_map(|(&pool_id, reservation)| {
                    (&reservation.endpoint == endpoint).then_some(pool_id)
                })
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

fn advance_catalog_revision(state: &mut PoolRegistryState) -> u64 {
    state.catalog_revision = state.catalog_revision.saturating_add(1);
    state.catalog_revision
}

fn publish_catalog_upsert(
    state: &mut PoolRegistryState,
    sender: &watch::Sender<DcPoolCatalog>,
    descriptor: DcPoolDescriptor,
) {
    let revision = advance_catalog_revision(state);
    sender.send_modify(|catalog| catalog.upsert(revision, descriptor));
}

fn publish_catalog_remove(
    state: &mut PoolRegistryState,
    sender: &watch::Sender<DcPoolCatalog>,
    pool_id: PoolId,
) {
    let revision = advance_catalog_revision(state);
    sender.send_modify(|catalog| catalog.remove(revision, pool_id));
}

fn publish_catalog_clear(state: &mut PoolRegistryState, sender: &watch::Sender<DcPoolCatalog>) {
    let revision = advance_catalog_revision(state);
    sender.send_modify(|catalog| catalog.clear(revision));
}

fn publish_load_if_changed(
    state: &PoolRegistryState,
    sender: &watch::Sender<Vec<PoolLoadSnapshot>>,
    pool_id: PoolId,
) {
    let snapshot = state
        .pools
        .get(&pool_id)
        .filter(|entry| entry.state == PoolEntryState::Active)
        .and_then(|entry| {
            entry
                .serving
                .as_ref()
                .map(|serving| serving.load.snapshot(entry.identity))
        });
    sender.send_if_modified(|snapshots| {
        match (
            snapshots.binary_search_by_key(&pool_id, |item| item.producer.pool_id()),
            snapshot,
        ) {
            (Ok(index), Some(snapshot)) if snapshots[index] != snapshot => {
                snapshots[index] = snapshot;
                true
            }
            (Ok(index), None) => {
                snapshots.remove(index);
                true
            }
            (Err(index), Some(snapshot)) => {
                snapshots.insert(index, snapshot);
                true
            }
            _ => false,
        }
    });
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

fn validate_roles(roles: &[WorkerRole]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !roles.is_empty(),
        "pool requires at least one declared worker role"
    );
    let mut unique = std::collections::HashSet::with_capacity(roles.len());
    for role in roles {
        anyhow::ensure!(
            unique.insert(*role),
            "pool repeats declared worker role {role:?}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, mpsc as std_mpsc};

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, RoutingScopeId,
    };

    use super::*;
    use crate::kv_dc_relay::identity::{KvQueryHashFormat, ModelTarget};

    type TestCkfBuilder =
        Box<dyn FnOnce(CkfConfig) -> Result<DcCkfState, CkfBuildError> + Send + 'static>;

    struct GatedBuilder {
        builder: TestCkfBuilder,
        started: tokio::sync::oneshot::Receiver<()>,
        release: std_mpsc::Sender<()>,
        finished: tokio::sync::oneshot::Receiver<()>,
    }

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

    fn relay_identity() -> DcRelayIdentity {
        DcRelayIdentity::new(11, 7)
    }

    fn registration(model: &str) -> CanonicalModelRegistration {
        CanonicalModelRegistration::new(
            CanonicalModelId::new(model).unwrap(),
            vec![ModelAlias::new(format!("{model}-alias")).unwrap()],
        )
    }

    fn query_semantics() -> KvQuerySemantics {
        KvQuerySemantics::new(64, KvQueryHashFormat::DynamoStandardV1).unwrap()
    }

    fn request(pool_id: PoolId, endpoint: &str, model: &str) -> PoolAttachRequest {
        PoolAttachRequest {
            pool_id,
            endpoint: EndpointId::from(endpoint),
            registrations: vec![registration(model)],
            query_semantics: query_semantics(),
            roles: vec![WorkerRole::Aggregated],
            serving_facts: Some(PoolServingFacts {
                runtime_configs: HashMap::new(),
            }),
        }
    }

    fn gated_builder() -> GatedBuilder {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let builder = move |config| {
            let _ = started_tx.send(());
            release_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("test must release the CKF builder");
            let result = DcCkfState::new(config);
            let _ = finished_tx.send(());
            result
        };
        GatedBuilder {
            builder: Box::new(builder),
            started: started_rx,
            release: release_tx,
            finished: finished_rx,
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
        let registry = PoolRegistry::new(relay_identity(), config());
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
        assert_eq!(catalog.drt_instance_id(), 11);
        assert_eq!(catalog.relay_incarnation(), 7);
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
    async fn burst_attach_defers_catalog_materialization_until_observed() {
        let registry = PoolRegistry::new(relay_identity(), config());
        let catalog_rx = registry.watch_catalog();
        let mut attachments = Vec::new();

        for seed in 1..=32 {
            attachments.push(
                registry
                    .attach(request(
                        pool(seed),
                        &format!("pool-{seed}.router.generate"),
                        &format!("model-{seed}"),
                    ))
                    .await
                    .unwrap(),
            );
        }

        assert_eq!(catalog_rx.borrow().revision(), 32);
        assert!(!catalog_rx.borrow().is_materialized());

        let catalog = registry.catalog();
        let pool_ids: Vec<_> = catalog
            .pools()
            .iter()
            .map(DcPoolDescriptor::pool_id)
            .collect();
        assert!(catalog.is_materialized());
        assert!(pool_ids.windows(2).all(|pair| pair[0] < pair[1]));
        let serialized = serde_json::to_value(&catalog).unwrap();
        assert_eq!(serialized["drt_instance_id"], 11);
        assert_eq!(serialized["relay_incarnation"], 7);
        assert!(serialized.get("process_incarnation").is_none());
        assert_eq!(serialized["revision"], 32);
        assert_eq!(serialized["pools"].as_array().unwrap().len(), 32);

        registry.shutdown().await;
        assert_eq!(catalog.pools().len(), attachments.len());
        assert!(registry.catalog().pools().is_empty());
    }

    #[tokio::test]
    async fn catalog_publishes_and_preserves_generation_query_semantics() {
        let registry = PoolRegistry::new(relay_identity(), config());
        let expected = KvQuerySemantics::new(128, KvQueryHashFormat::DynamoEagleV1).unwrap();
        let mut attach_request = request(pool(1), "fast.router.generate", "llama");
        attach_request.query_semantics = expected;
        let mut attachment = registry.attach(attach_request).await.unwrap();
        let producer = attachment.handle.identity();

        let catalog = registry.catalog();
        let initial_descriptor = descriptor(&catalog, pool(1));
        assert_eq!(initial_descriptor.producer(), producer);
        assert_eq!(initial_descriptor.query_semantics(), expected);

        registry
            .replace_registrations(
                &mut attachment,
                vec![CanonicalModelRegistration::new(
                    CanonicalModelId::new("llama").unwrap(),
                    vec![ModelAlias::new("chat").unwrap()],
                )],
            )
            .await
            .unwrap();
        let catalog = registry.catalog();
        let updated_descriptor = descriptor(&catalog, pool(1));
        assert_eq!(updated_descriptor.producer(), producer);
        assert_eq!(updated_descriptor.query_semantics(), expected);

        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn registration_updates_remain_pool_local() {
        let registry = PoolRegistry::new(relay_identity(), config());
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
        let registry = PoolRegistry::new(relay_identity(), config());
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
    async fn endpoint_reassignment_waits_for_prior_pool_removal() {
        let registry = PoolRegistry::new(relay_identity(), config());
        let endpoint = "fast.router.generate";
        let first = registry
            .attach(request(pool(1), endpoint, "llama"))
            .await
            .unwrap();

        assert!(
            registry
                .withdraw(
                    first.pool_id,
                    first.layout_generation,
                    PoolRetirementMode::Graceful,
                )
                .await
        );
        assert!(registry.catalog().pools().is_empty());

        let error = registry
            .attach(request(pool(2), endpoint, "llama"))
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("already assigned"));
        assert!(registry.catalog().pools().is_empty());

        registry.detach(first).await.unwrap();
        let second = registry
            .attach(request(pool(2), endpoint, "llama"))
            .await
            .unwrap();
        let catalog = registry.catalog();
        assert_eq!(catalog.pools().len(), 1);
        assert_eq!(catalog.pools()[0].pool_id(), pool(2));
        assert_eq!(
            catalog.pools()[0].serving_endpoint(),
            &EndpointId::from(endpoint)
        );

        registry.detach(second).await.unwrap();
    }

    #[tokio::test]
    async fn cancelled_allocation_rolls_back_and_allows_reattach() {
        let registry = Arc::new(PoolRegistry::new(relay_identity(), config()));
        let pool_id = pool(1);
        let GatedBuilder {
            builder,
            started: started_rx,
            release: release_tx,
            finished: finished_rx,
        } = gated_builder();
        let task_registry = registry.clone();
        let attach = tokio::spawn(async move {
            task_registry
                .attach_with_builder(request(pool_id, "first.router.generate", "llama"), builder)
                .await
        });

        started_rx.await.unwrap();
        attach.abort();
        assert!(matches!(attach.await, Err(error) if error.is_cancelled()));
        assert!(registry.state.lock().reservations.is_empty());

        release_tx.send(()).unwrap();
        finished_rx.await.unwrap();
        let attachment = registry
            .attach(request(pool_id, "second.router.generate", "llama"))
            .await
            .unwrap();
        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_during_allocation_never_publishes_the_pool() {
        let registry = Arc::new(PoolRegistry::new(relay_identity(), config()));
        let GatedBuilder {
            builder,
            started: started_rx,
            release: release_tx,
            finished: finished_rx,
        } = gated_builder();
        let task_registry = registry.clone();
        let attach = tokio::spawn(async move {
            task_registry
                .attach_with_builder(request(pool(1), "first.router.generate", "llama"), builder)
                .await
        });

        started_rx.await.unwrap();
        registry.shutdown().await;
        assert!(registry.state.lock().reservations.is_empty());
        assert!(registry.catalog().pools().is_empty());

        release_tx.send(()).unwrap();
        finished_rx.await.unwrap();
        let Err(error) = attach.await.unwrap() else {
            panic!("pool attached after registry shutdown");
        };
        assert!(error.to_string().contains("retired before commit"));
        assert_eq!(registry.pool_count().await, 0);
        assert!(registry.catalog().pools().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ckf_allocation_does_not_block_the_async_executor() {
        let registry = Arc::new(PoolRegistry::new(relay_identity(), config()));
        let GatedBuilder {
            builder,
            started: started_rx,
            release: release_tx,
            finished: finished_rx,
        } = gated_builder();
        let task_registry = registry.clone();
        let attach = tokio::spawn(async move {
            task_registry
                .attach_with_builder(request(pool(1), "first.router.generate", "llama"), builder)
                .await
        });

        tokio::time::timeout(Duration::from_millis(500), started_rx)
            .await
            .expect("blocking CKF allocation did not start")
            .unwrap();
        tokio::time::timeout(
            Duration::from_millis(500),
            tokio::time::sleep(Duration::from_millis(1)),
        )
        .await
        .expect("CKF allocation blocked the async executor");

        release_tx.send(()).unwrap();
        finished_rx.await.unwrap();
        let attachment = attach.await.unwrap().unwrap();
        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn failed_actor_build_rolls_back_the_pool_reservation() {
        let registry = PoolRegistry::new(
            relay_identity(),
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
        let registry = PoolRegistry::new(relay_identity(), config());
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
    async fn relay_incarnation_fences_an_identical_pool_layout() {
        let first_registry = PoolRegistry::new(DcRelayIdentity::new(11, 7), config());
        let second_registry = PoolRegistry::new(DcRelayIdentity::new(11, 8), config());
        let pool_id = pool(1);
        let first = first_registry
            .attach(request(pool_id, "fast.router.generate", "llama"))
            .await
            .unwrap();
        let second = second_registry
            .attach(request(pool_id, "fast.router.generate", "llama"))
            .await
            .unwrap();

        assert_ne!(first.handle.identity(), second.handle.identity());
        assert_eq!(first.layout_generation, second.layout_generation);

        first_registry.detach(first).await.unwrap();
        second_registry.detach(second).await.unwrap();
    }

    #[tokio::test]
    async fn withdraw_removes_catalog_before_actor_retirement() {
        let registry = PoolRegistry::new(relay_identity(), config());
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
        let registry = PoolRegistry::new(relay_identity(), config());
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
    async fn declared_role_change_updates_catalog_without_replacing_generation() {
        let registry = PoolRegistry::new(relay_identity(), config());
        let mut attachment = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        let producer = attachment.handle.identity();

        registry
            .replace_roles(&mut attachment, vec![WorkerRole::Decode])
            .unwrap();
        let catalog = registry.catalog();
        assert_eq!(descriptor(&catalog, pool(1)).producer(), producer);
        assert_eq!(
            descriptor(&catalog, pool(1)).pool_roles(),
            [WorkerRole::Decode]
        );

        assert!(registry.replace_roles(&mut attachment, Vec::new()).is_err());
        assert_eq!(
            descriptor(&registry.catalog(), pool(1)).pool_roles(),
            [WorkerRole::Decode]
        );
        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn one_lora_target_binds_to_independent_pools() {
        let registry = PoolRegistry::new(relay_identity(), config());
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
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                serving_facts: Some(PoolServingFacts {
                    runtime_configs: HashMap::new(),
                }),
            })
            .await
            .unwrap();
        let second = registry
            .attach(PoolAttachRequest {
                pool_id: pool(2),
                endpoint: EndpointId::from("slow.router.generate"),
                registrations: registrations(),
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                serving_facts: Some(PoolServingFacts {
                    runtime_configs: HashMap::new(),
                }),
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
        let registry = PoolRegistry::new(relay_identity(), config());
        let attachment = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        retire(&registry, attachment, PoolRetirementMode::Fenced).await;
        assert!(registry.watch_catalog().borrow().pools().is_empty());
    }

    #[tokio::test]
    async fn fencing_withdraws_only_the_target_pool_descriptor() {
        let registry = PoolRegistry::new(relay_identity(), config());
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
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                serving_facts: Some(PoolServingFacts {
                    runtime_configs: HashMap::new(),
                }),
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

    #[tokio::test]
    async fn local_only_pool_publishes_no_serving_facts() {
        let registry = PoolRegistry::new(relay_identity(), config());
        let mut attach_request = request(pool(1), "fast.router.generate", "llama");
        attach_request.serving_facts = None;
        let attachment = registry.attach(attach_request).await.unwrap();

        assert_eq!(registry.catalog().pools().len(), 1);
        assert!(registry.load_snapshots().is_empty());
        assert!(
            !registry
                .replace_load_capacity(
                    attachment.pool_id,
                    attachment.layout_generation,
                    &HashMap::new(),
                )
                .unwrap()
        );
        assert!(!registry.observe_load(
            attachment.pool_id,
            attachment.layout_generation,
            ActiveLoad::default(),
        ));

        attachment.handle.state_stats().await.unwrap();
        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn load_is_generation_scoped_and_withdrawn_with_the_pool() {
        let registry = PoolRegistry::new(relay_identity(), config());
        let runtime_configs = HashMap::from([(
            1,
            ModelRuntimeConfig {
                total_kv_blocks: Some(100),
                max_num_batched_tokens: Some(2_048),
                ..ModelRuntimeConfig::default()
            },
        )]);
        let attachment = registry
            .attach(PoolAttachRequest {
                pool_id: pool(1),
                endpoint: EndpointId::from("fast.router.generate"),
                registrations: vec![registration("llama")],
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                serving_facts: Some(PoolServingFacts { runtime_configs }),
            })
            .await
            .unwrap();
        let old_generation = attachment.layout_generation;
        let old_producer = attachment.handle.identity();

        let initial_load = registry.load_snapshots();
        assert_eq!(initial_load.len(), 1);
        assert_eq!(initial_load[0].kv_expected_ranks, 1);
        assert_eq!(initial_load[0].kv_observed_ranks, 0);

        let mut load = ActiveLoad {
            worker_id: 1,
            dp_rank: 0,
            ..ActiveLoad::default()
        };
        load.kv_used_blocks = Some(40);
        load.active_decode_blocks = Some(30);
        load.active_prefill_tokens = Some(512);
        assert!(registry.observe_load(pool(1), old_generation, load));
        let observed = registry.load_snapshots()[0];
        assert_eq!(observed.kv_used_blocks, 40);
        assert_eq!(observed.active_decode_blocks, 30);
        assert_eq!(observed.active_prefill_tokens, 512);
        assert!(!observed.has_degraded_coverage());

        assert!(
            registry
                .withdraw(pool(1), old_generation, PoolRetirementMode::Graceful)
                .await
        );
        assert!(registry.load_snapshots().is_empty());
        assert!(!registry.observe_load(
            pool(1),
            old_generation,
            ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                kv_used_blocks: Some(99),
                ..ActiveLoad::default()
            },
        ));
        registry.detach(attachment).await.unwrap();

        let replacement = registry
            .attach(request(pool(1), "fast.router.generate", "llama"))
            .await
            .unwrap();
        assert_ne!(replacement.layout_generation, old_generation);
        assert_ne!(replacement.handle.identity(), old_producer);
        assert!(
            !registry
                .replace_load_capacity(
                    pool(1),
                    old_generation,
                    &HashMap::from([(
                        1,
                        ModelRuntimeConfig {
                            total_kv_blocks: Some(999),
                            ..ModelRuntimeConfig::default()
                        },
                    )]),
                )
                .unwrap()
        );
        assert_eq!(registry.load_snapshots()[0].kv_expected_ranks, 0);
        registry.detach(replacement).await.unwrap();
    }
}
