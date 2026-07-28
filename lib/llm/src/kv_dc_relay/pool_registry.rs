// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::{CkfConfig, ProducerIdentity};
use dynamo_runtime::protocols::EndpointId;
use tokio::sync::{Mutex, mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::actor::{ActorFault, KvDcRelayHandle, StreamScope};
use super::host::KvDcRelayError;
use super::identity::{
    CanonicalModelId, CanonicalModelRegistration, DcPoolCatalog, DcPoolDescriptor, ModelAlias,
    ModelAliasBinding, ModelPoolBinding,
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

struct PoolEntry {
    endpoint: EndpointId,
    handle: KvDcRelayHandle,
    identity: ProducerIdentity,
    layout_generation: u64,
    registrations: Arc<[CanonicalModelRegistration]>,
    cancel: CancellationToken,
    fenced: bool,
}

struct PoolRegistryState {
    pools: HashMap<PoolId, PoolEntry>,
    next_layout_generation: u64,
    catalog_revision: u64,
}

impl Default for PoolRegistryState {
    fn default() -> Self {
        Self {
            pools: HashMap::new(),
            next_layout_generation: 1,
            catalog_revision: 0,
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

        let mut state = self.state.lock().await;
        if let Some(existing) = state.pools.get(&request.pool_id) {
            anyhow::bail!(
                "pool {} is already owned by endpoint {} and cannot also attach endpoint {}",
                request.pool_id,
                existing.endpoint,
                request.endpoint
            );
        }
        validate_registrations(&request.registrations)?;

        let layout_generation = allocate_layout_generation(&mut state)?;
        let mut config = CkfConfig::new(self.actor_config.expected_unique_blocks);
        config.publish_every_n_events = self.actor_config.publication_threshold;
        let (handle, faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            StreamScope {
                process_incarnation: self.process_incarnation,
                layout_generation,
                pool_id: request.pool_id,
            },
            self.actor_config.publication_delay,
        )?;
        let identity = handle.identity();
        let registrations: Arc<[CanonicalModelRegistration]> = request.registrations.into();
        let cancel = CancellationToken::new();

        state.pools.insert(
            request.pool_id,
            PoolEntry {
                endpoint: request.endpoint.clone(),
                handle: handle.clone(),
                identity,
                layout_generation,
                registrations: registrations.clone(),
                cancel: cancel.clone(),
                fenced: false,
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
        let pool_id = attachment.pool_id;
        let layout_generation = attachment.layout_generation;
        let faults = attachment.faults;
        let entry = {
            let mut state = self.state.lock().await;
            let Some(current) = state.pools.get(&pool_id) else {
                return Ok(());
            };
            if current.layout_generation != layout_generation {
                return Ok(());
            }
            let entry = state
                .pools
                .remove(&pool_id)
                .ok_or(KvDcRelayError::ActorStopped)?;
            publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
            entry
        };

        entry.cancel.cancel();
        if entry.fenced {
            Ok(())
        } else {
            shutdown_while_draining_faults(pool_id, entry.handle, faults).await
        }
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
        let mut state = self.state.lock().await;
        let entry = state
            .pools
            .get_mut(&attachment.pool_id)
            .ok_or_else(|| anyhow::anyhow!("pool {} is not attached", attachment.pool_id))?;
        anyhow::ensure!(
            entry.layout_generation == attachment.layout_generation && !entry.fenced,
            "pool {} generation {} is no longer active",
            attachment.pool_id,
            attachment.layout_generation
        );
        entry.registrations = registrations.clone();
        attachment.registrations = registrations;
        publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
        Ok(())
    }

    pub(super) async fn fence(&self, pool_id: PoolId) -> Result<(), KvDcRelayError> {
        let handle = {
            let mut state = self.state.lock().await;
            let Some(entry) = state.pools.get_mut(&pool_id) else {
                return Ok(());
            };
            if entry.fenced {
                return Ok(());
            }
            entry.fenced = true;
            entry.cancel.cancel();
            let handle = entry.handle.clone();
            publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
            handle
        };
        handle.fence().await
    }

    pub(super) async fn model_pool_bindings(&self) -> Vec<ModelPoolBinding> {
        let catalog = self.catalog_tx.borrow();
        let mut bindings: Vec<_> = catalog
            .pools()
            .iter()
            .flat_map(|pool| {
                pool.registrations().iter().map(|registration| {
                    ModelPoolBinding::with_target(
                        registration.model().clone(),
                        pool.pool_id(),
                        registration.target().clone(),
                    )
                })
            })
            .collect();
        bindings.sort_unstable();
        bindings
    }

    pub(super) async fn model_alias_bindings(&self) -> Vec<ModelAliasBinding> {
        let catalog = self.catalog_tx.borrow();
        let mut bindings: Vec<_> = catalog
            .pools()
            .iter()
            .flat_map(|pool| {
                pool.registrations().iter().flat_map(|registration| {
                    registration.aliases().iter().map(|alias| {
                        ModelAliasBinding::new(alias.clone(), registration.model().clone())
                    })
                })
            })
            .collect();
        bindings.sort_unstable();
        bindings.dedup();
        bindings
    }

    pub(super) async fn model_pool_bindings_for(
        &self,
        requested_model: &str,
    ) -> Vec<ModelPoolBinding> {
        let catalog = self.catalog_tx.borrow();
        let mut bindings: Vec<_> = catalog
            .pools()
            .iter()
            .flat_map(|pool| {
                pool.registrations()
                    .iter()
                    .filter(|registration| {
                        registration.model().as_str() == requested_model
                            || registration
                                .aliases()
                                .iter()
                                .any(|alias| alias.as_str() == requested_model)
                    })
                    .map(|registration| {
                        ModelPoolBinding::with_target(
                            registration.model().clone(),
                            pool.pool_id(),
                            registration.target().clone(),
                        )
                    })
            })
            .collect();
        bindings.sort_unstable();
        bindings
    }

    pub(super) fn catalog(&self) -> DcPoolCatalog {
        self.catalog_tx.borrow().clone()
    }

    pub(super) fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.catalog_tx.subscribe()
    }

    pub(super) async fn shutdown(&self) {
        let entries = {
            let mut state = self.state.lock().await;
            let entries = state.pools.drain().collect::<Vec<_>>();
            publish_catalog(&mut state, self.process_incarnation, &self.catalog_tx);
            entries
        };
        for (pool_id, entry) in entries {
            entry.cancel.cancel();
            if !entry.fenced
                && let Err(error) = entry.handle.fence().await
            {
                tracing::warn!(%pool_id, %error, "KV Relay pool actor failed to fence during registry shutdown");
            }
        }
    }

    #[cfg(test)]
    async fn pool_count(&self) -> usize {
        self.state.lock().await.pools.len()
    }
}

async fn shutdown_while_draining_faults(
    pool_id: PoolId,
    handle: KvDcRelayHandle,
    mut faults: mpsc::Receiver<ActorFault>,
) -> Result<(), KvDcRelayError> {
    let shutdown = handle.shutdown();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            result = &mut shutdown => return result,
            fault = faults.recv() => match fault {
                Some(fault) => tracing::debug!(
                    %pool_id,
                    worker_id = fault.worker_id,
                    dp_rank = fault.dp_rank,
                    category = ?fault.category,
                    error = %fault.message,
                    "Draining KV DC Relay actor fault while retiring its pool"
                ),
                None => return shutdown.await,
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
        .filter(|entry| !entry.fenced)
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

        let bindings = registry.model_pool_bindings_for("llama").await;
        assert_eq!(bindings.len(), 2);
        assert_ne!(bindings[0].pool_id(), bindings[1].pool_id());
        assert_eq!(registry.pool_count().await, 2);
        let catalog = registry.catalog();
        assert_eq!(catalog.process_incarnation(), 7);
        assert_eq!(catalog.pools().len(), 2);
        assert_eq!(
            catalog
                .pools()
                .iter()
                .find(|descriptor| descriptor.pool_id() == pool(1))
                .unwrap()
                .serving_endpoint(),
            &EndpointId::from("fast.router.generate")
        );

        registry.detach(first).await.unwrap();
        registry.detach(second).await.unwrap();
    }

    #[tokio::test]
    async fn registration_updates_do_not_coordinate_aliases_across_pools() {
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
        assert_eq!(
            registry
                .model_pool_bindings_for("mistral-alias")
                .await
                .len(),
            2
        );

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
        assert_eq!(
            registry.model_pool_bindings_for("mistral-alias").await[0]
                .model()
                .as_str(),
            "llama"
        );
        assert_eq!(
            registry.model_pool_bindings_for("llama-alias").await[0]
                .model()
                .as_str(),
            "mistral"
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
        assert_eq!(
            registry
                .model_pool_bindings_for(adapter.as_str())
                .await
                .len(),
            1
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

        let bindings = registry.model_pool_bindings_for(adapter.as_str()).await;
        assert_eq!(bindings.len(), 2);
        assert!(bindings.iter().all(|binding| {
            binding.target()
                == &ModelTarget::Lora {
                    base_model: base.clone(),
                    adapter: adapter.clone(),
                }
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
        let pool_id = attachment.pool_id;

        registry.fence(pool_id).await.unwrap();
        assert!(registry.model_pool_bindings().await.is_empty());
        assert!(registry.watch_catalog().borrow().pools().is_empty());

        registry.detach(attachment).await.unwrap();
    }

    #[tokio::test]
    async fn fencing_withdraws_aliases_owned_only_by_that_pool() {
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

        assert_eq!(registry.model_alias_bindings().await.len(), 1);
        registry.fence(with_alias.pool_id).await.unwrap();
        assert!(registry.model_alias_bindings().await.is_empty());
        assert_eq!(registry.model_pool_bindings().await.len(), 1);

        registry.detach(with_alias).await.unwrap();
        registry.detach(without_alias).await.unwrap();
    }
}
