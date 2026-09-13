// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet};

use futures::StreamExt;
use tokio::sync::watch;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{
    DiscoveryEvent, DiscoveryInstanceId, DiscoveryQuery, DiscoveryStream,
};
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use tokio_util::sync::CancellationToken;

use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;
use dynamo_kv_router::protocols::WorkerId;

/// Type alias for the runtime config watch receiver.
pub type RuntimeConfigWatch = watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>;

// `lifecycle` bounds this task directly rather than leaving it to notice its
// receiver is gone. That receiver-drop signal only reaches this task via a
// failed `tx.send`, and `tx.send` is only attempted when a discovery event
// actually changes `configs` — so on a quiescent endpoint (no discovery
// events, the case a retired WorkerSet is usually in) this task can sit in
// `stream.next()` forever, past every consumer's exit, past `lifecycle`
// cancelling. See the "WorkerSet churn" test below.
fn base_runtime_config_watch(
    mut stream: DiscoveryStream,
    lifecycle: CancellationToken,
) -> watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>> {
    let (tx, rx) = watch::channel(HashMap::new());

    tokio::spawn(async move {
        let mut configs = HashMap::new();
        loop {
            let result = tokio::select! {
                _ = lifecycle.cancelled() => break,
                event = stream.next() => match event {
                    Some(result) => result,
                    None => break,
                },
            };
            match result {
                Ok(DiscoveryEvent::Added(instance)) => {
                    let DiscoveryInstanceId::Model(id) = instance.id() else {
                        continue;
                    };
                    let card = match instance.deserialize_model::<ModelDeploymentCard>() {
                        Ok(card) => card,
                        Err(error) => {
                            tracing::warn!(
                                instance_id = id.instance_id,
                                %error,
                                "Failed to deserialize base model runtime config"
                            );
                            continue;
                        }
                    };
                    if id.model_suffix.is_some() || card.lora.is_some() {
                        tracing::debug!(
                            instance_id = id.instance_id,
                            reason = if id.model_suffix.is_some() {
                                "model_suffix"
                            } else {
                                "lora"
                            },
                            "Skipping non-base model card; it defines no base runtime config"
                        );
                        continue;
                    }
                    if let Err(error) = card.runtime_config.data_parallel_rank_range() {
                        tracing::warn!(
                            instance_id = id.instance_id,
                            %error,
                            "Ignoring base model runtime config with invalid data-parallel rank range"
                        );
                        configs.remove(&id.instance_id);
                    } else {
                        configs.insert(id.instance_id, card.runtime_config);
                    }
                }
                Ok(DiscoveryEvent::ModelTaintsUpdated(update)) => {
                    if update.id.model_suffix.is_some() {
                        continue;
                    }
                    let Some(config) = configs.get_mut(&update.id.instance_id) else {
                        tracing::warn!(
                            instance_id = update.id.instance_id,
                            "Ignoring taint update for an unknown base model card"
                        );
                        continue;
                    };
                    let taints = update.taints.into_iter().collect();
                    if config.taints == taints {
                        continue;
                    }
                    config.taints = taints;
                }
                Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::Model(id))) => {
                    if id.model_suffix.is_none() {
                        configs.remove(&id.instance_id);
                    }
                }
                Ok(DiscoveryEvent::Removed(_)) => continue,
                Err(error) => {
                    tracing::error!(%error, "Base model runtime-config discovery stream failed");
                    continue;
                }
            }

            if *tx.borrow() != configs && tx.send(configs.clone()).is_err() {
                break;
            }
        }
    });

    rx
}

/// The single predicate deciding which workers a router may route to.
///
/// Returns the admitted workers — present in both the endpoint's availability set and the
/// base model runtime configs — and, separately, the instance ids that were available but
/// carried no base runtime config. The excluded half exists because a worker dropped here
/// leaves no trace of its own: the failure surfaces later, in another crate, as
/// `KvSchedulerError::NoEndpoints`, with nothing naming which side of this join was empty.
fn join_available_instances_with_configs(
    instances: &HashSet<WorkerId>,
    configs: &HashMap<WorkerId, ModelRuntimeConfig>,
) -> (HashMap<WorkerId, ModelRuntimeConfig>, BTreeSet<WorkerId>) {
    let mut admitted = HashMap::new();
    let mut excluded = BTreeSet::new();
    for id in instances {
        match configs.get(id) {
            Some(config) => {
                admitted.insert(*id, config.clone());
            }
            None => {
                excluded.insert(*id);
            }
        }
    }
    (admitted, excluded)
}

/// Join instance availability and config discovery into a single watch.
///
/// Only includes workers that have BOTH an instance registration AND a runtime config.
/// Spawns a background task that recomputes the joined state whenever either source changes.
/// The returned `watch::Receiver` always contains the latest joined snapshot.
///
/// `lifecycle` bounds `Source 2`'s `base_runtime_config_watch` task directly, and this
/// function's own join task below. `Source 1`'s `Client` already scopes its own
/// `monitor_instance_source` task to the lifetime of its last `instance_avail_watcher`
/// receiver, so it needs no token here — dropping this function's receivers already
/// stops it. A caller scoped to something narrower than the process, such as a monitor
/// bound to one `WorkerSet`'s lifecycle, must still pass that scope's own token, or
/// `base_runtime_config_watch`'s task outlives every dropped reference the caller holds
/// and leaks until process shutdown.
pub async fn runtime_config_watch(
    endpoint: &Endpoint,
    lifecycle: CancellationToken,
) -> anyhow::Result<RuntimeConfigWatch> {
    let component = endpoint.component();
    let cancel_token = component.drt().primary_token();

    // Source 1: instance availability (watches DiscoveryQuery::Endpoint)
    let client = endpoint.client().await?;
    let mut instance_ids_rx = client.instance_avail_watcher();

    // Source 2: runtime configs from discovery (watches DiscoveryQuery::EndpointModels)
    let discovery = component.drt().discovery();
    let eid = endpoint.id();
    let stream = discovery
        .list_and_watch(
            DiscoveryQuery::EndpointModels {
                namespace: eid.namespace.clone(),
                component: eid.component.clone(),
                endpoint: eid.name.clone(),
            },
            Some(cancel_token.clone()),
        )
        .await?;
    let mut configs_rx = base_runtime_config_watch(stream, lifecycle.clone());

    let (tx, rx) = watch::channel(HashMap::new());

    tokio::spawn(async move {
        // Remembered across ticks so the warning below fires on a transition into a new
        // exclusion set, not on every recomputation of an unchanged one.
        let mut reported_exclusions = BTreeSet::new();
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => break,
                _ = lifecycle.cancelled() => break,
                _ = tx.closed() => break,
                result = instance_ids_rx.changed() => { if result.is_err() { break; } }
                result = configs_rx.changed() => { if result.is_err() { break; } }
            }

            let instances: HashSet<WorkerId> = instance_ids_rx
                .borrow_and_update()
                .iter()
                .copied()
                .collect();
            let configs = configs_rx.borrow_and_update().clone();

            let (ready, excluded) = join_available_instances_with_configs(&instances, &configs);

            if excluded != reported_exclusions {
                if !excluded.is_empty() {
                    tracing::warn!(
                        endpoint = %eid,
                        excluded_instance_ids = ?excluded,
                        "Discovered instances have no base model runtime config and cannot be routed"
                    );
                }
                reported_exclusions = excluded;
            }

            // Only send if the joined result actually changed, to avoid waking
            // downstream consumers (wait_for, changed) on no-op recomputations.
            if *tx.borrow() == ready {
                continue;
            }

            // Break if all receivers dropped (e.g., TOCTOU in model_manager discards a duplicate).
            if tx.send(ready).is_err() {
                break;
            }
        }
    });

    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::discovery::{DiscoveryInstance, ModelCardInstanceId, ModelTaintsUpdate};

    fn model_instance(
        instance_id: u64,
        model_suffix: Option<&str>,
        card: &ModelDeploymentCard,
    ) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id,
            card_json: serde_json::to_value(card).unwrap(),
            model_suffix: model_suffix.map(str::to_string),
        }
    }

    /// A worker registered on the endpoint but with no base model runtime config is
    /// silently absent from the routable set, and the request that later fails carries no
    /// hint of which of the two discovery sources was missing it. It must be reported.
    #[test]
    fn available_instances_without_a_runtime_config_are_reported_as_excluded() {
        let instances = HashSet::from([7, 8]);
        let configs = HashMap::from([(7, ModelRuntimeConfig::default())]);

        let (admitted, excluded) = join_available_instances_with_configs(&instances, &configs);

        assert_eq!(admitted.keys().copied().collect::<HashSet<_>>(), [7].into());
        assert_eq!(excluded, BTreeSet::from([8]));
    }

    /// The reported deployment's worker shape: no `--kv_events_config`, so the card
    /// advertises `kv_event_publishing_enabled: Some(false)` and no `router_config`.
    /// KV-event capability must never gate admission — such a worker is routable and
    /// simply scores zero KV overlap.
    #[test]
    fn a_worker_that_publishes_no_kv_events_is_still_admitted() {
        let mut card = ModelDeploymentCard::default();
        card.runtime_config.kv_event_publishing_enabled = Some(false);
        assert!(card.router_config.is_none(), "bare worker advertises none");
        let instances = HashSet::from([0x1a0dcb6265db46]);
        let configs = HashMap::from([(0x1a0dcb6265db46, card.runtime_config)]);

        let (admitted, excluded) = join_available_instances_with_configs(&instances, &configs);

        assert_eq!(
            admitted
                .get(&0x1a0dcb6265db46)
                .unwrap()
                .kv_event_publishing_enabled,
            Some(false)
        );
        assert!(excluded.is_empty());
    }

    /// The join must hand the scheduler the worker's advertised capacity untouched;
    /// flattening it to defaults would silently corrupt KV-aware scoring.
    #[test]
    fn admitted_configs_keep_their_kv_aware_scoring_inputs() {
        let config = ModelRuntimeConfig {
            kv_event_publishing_enabled: Some(true),
            total_kv_blocks: Some(8192),
            data_parallel_start_rank: 2,
            data_parallel_size: 4,
            ..Default::default()
        };
        let instances = HashSet::from([11]);
        let configs = HashMap::from([(11, config)]);

        let (admitted, excluded) = join_available_instances_with_configs(&instances, &configs);

        let admitted = admitted.get(&11).unwrap();
        assert_eq!(admitted.kv_event_publishing_enabled, Some(true));
        assert_eq!(admitted.total_kv_blocks, Some(8192));
        assert_eq!(admitted.data_parallel_start_rank, 2);
        assert_eq!(admitted.data_parallel_size, 4);
        assert!(excluded.is_empty());
    }

    /// The instance side drives the join: a leftover card for a worker that is no longer
    /// available is neither routable nor an exclusion worth reporting.
    #[test]
    fn a_config_without_an_available_instance_is_neither_admitted_nor_reported() {
        let instances = HashSet::from([7]);
        let configs = HashMap::from([
            (7, ModelRuntimeConfig::default()),
            (12, ModelRuntimeConfig::default()),
        ]);

        let (admitted, excluded) = join_available_instances_with_configs(&instances, &configs);

        assert_eq!(admitted.keys().copied().collect::<HashSet<_>>(), [7].into());
        assert!(excluded.is_empty());
    }

    /// Regression test for the "WorkerSet churn" leak this fix closes:
    /// `base_runtime_config_watch`'s task must exit when its `lifecycle` token
    /// cancels, even on a quiescent stream that never emits an event again.
    ///
    /// Before this fix the task's only exit paths were the stream ending or a
    /// failed `tx.send` — and `tx.send` runs only when a discovery event
    /// changes `configs`, so on a quiescent endpoint (the state a retired
    /// WorkerSet's discovery stream is normally left in) neither path ever
    /// fires. The task then outlived every dropped reference to its receiver
    /// and leaked until process shutdown.
    #[tokio::test]
    async fn base_runtime_config_watch_exits_on_lifecycle_cancellation_with_no_stream_activity() {
        // `_tx` stays alive for the whole test, so the stream never ends on its
        // own — the only way the task below can exit is `lifecycle` cancelling.
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream: DiscoveryStream =
            Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
        let lifecycle = CancellationToken::new();
        let mut configs = base_runtime_config_watch(stream, lifecycle.clone());

        lifecycle.cancel();

        // The task drops its `watch::Sender` when it exits — the one signal a
        // caller outside this module can observe. `changed()` on the paired
        // `Receiver` returns an error once every `Sender` is gone.
        tokio::time::timeout(std::time::Duration::from_secs(5), configs.changed())
            .await
            .expect("base_runtime_config_watch's task must exit within the timeout")
            .expect_err("the watch::Sender must be dropped once the task exits");
    }

    #[tokio::test]
    async fn only_base_cards_define_runtime_config_expectations() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream: DiscoveryStream =
            Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
        let mut configs = base_runtime_config_watch(stream, CancellationToken::new());
        let mut base = ModelDeploymentCard::default();
        base.runtime_config.data_parallel_start_rank = 3;
        base.runtime_config.data_parallel_size = 2;
        let mut lora = ModelDeploymentCard::default();
        lora.lora = Some(crate::model_card::LoraInfo {
            name: "adapter".to_string(),
            max_gpu_lora_count: Some(4),
        });
        lora.runtime_config.data_parallel_start_rank = 99;
        lora.runtime_config.data_parallel_size = 8;
        let base_instance = model_instance(7, None, &base);
        let lora_instance = model_instance(7, Some("adapter"), &lora);

        tx.send(Ok(DiscoveryEvent::Added(lora_instance.clone())))
            .unwrap();
        tx.send(Ok(DiscoveryEvent::Added(base_instance.clone())))
            .unwrap();
        configs.changed().await.unwrap();
        let config = configs.borrow().get(&7).cloned().unwrap();
        assert_eq!(config.data_parallel_start_rank, 3);
        assert_eq!(config.data_parallel_size, 2);

        tx.send(Ok(DiscoveryEvent::Removed(lora_instance.id())))
            .unwrap();
        tx.send(Ok(DiscoveryEvent::Removed(base_instance.id())))
            .unwrap();
        configs.changed().await.unwrap();
        assert!(configs.borrow().is_empty());
    }

    #[tokio::test]
    async fn invalid_data_parallel_ranges_are_ignored_before_runtime_config_watch() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream: DiscoveryStream =
            Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
        let mut configs = base_runtime_config_watch(stream, CancellationToken::new());
        let mut zero_size = ModelDeploymentCard::default();
        zero_size.runtime_config.data_parallel_size = 0;
        let mut oversized = ModelDeploymentCard::default();
        oversized.runtime_config.data_parallel_size = 4097;
        let mut overflowing = ModelDeploymentCard::default();
        overflowing.runtime_config.data_parallel_start_rank = u32::MAX;
        let valid = ModelDeploymentCard::default();

        for (instance_id, card) in [(6, &zero_size), (7, &oversized), (8, &overflowing)] {
            tx.send(Ok(DiscoveryEvent::Added(model_instance(
                instance_id,
                None,
                card,
            ))))
            .unwrap();
        }
        tx.send(Ok(DiscoveryEvent::Added(model_instance(9, None, &valid))))
            .unwrap();

        configs.changed().await.unwrap();
        assert!(!configs.borrow().contains_key(&6));
        assert!(!configs.borrow().contains_key(&7));
        assert!(!configs.borrow().contains_key(&8));
        assert_eq!(configs.borrow().get(&9).unwrap().data_parallel_size, 1);
    }

    #[tokio::test]
    async fn scoped_taint_updates_replace_only_known_base_worker_taints() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream: DiscoveryStream =
            Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
        let mut configs = base_runtime_config_watch(stream, CancellationToken::new());
        let mut base = ModelDeploymentCard::default();
        base.runtime_config.taints = HashSet::from(["old".to_string()]);
        let base_instance = model_instance(7, None, &base);
        let DiscoveryInstanceId::Model(id) = base_instance.id() else {
            unreachable!()
        };

        tx.send(Ok(DiscoveryEvent::Added(base_instance))).unwrap();
        configs.changed().await.unwrap();
        configs.borrow_and_update();

        let updated_taints = vec!["blue".to_string(), "gpu".to_string()];
        tx.send(Ok(DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
            id: id.clone(),
            taints: updated_taints.clone(),
        })))
        .unwrap();
        configs.changed().await.unwrap();
        assert_eq!(
            configs.borrow_and_update().get(&7).unwrap().taints,
            updated_taints.iter().cloned().collect()
        );

        tx.send(Ok(DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
            id: id.clone(),
            taints: updated_taints,
        })))
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), configs.changed())
                .await
                .is_err()
        );

        tx.send(Ok(DiscoveryEvent::ModelTaintsUpdated(ModelTaintsUpdate {
            id: ModelCardInstanceId {
                instance_id: 99,
                ..id
            },
            taints: vec!["unknown".to_string()],
        })))
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), configs.changed())
                .await
                .is_err()
        );
        assert_eq!(configs.borrow().len(), 1);
    }
}
