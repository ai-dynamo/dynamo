// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

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
                        continue;
                    }
                    configs.insert(id.instance_id, card.runtime_config);
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

/// Join instance availability and config discovery into a single watch.
///
/// Only includes workers that have BOTH an instance registration AND a runtime config.
/// Spawns a background task that recomputes the joined state whenever either source changes.
/// The returned `watch::Receiver` always contains the latest joined snapshot.
///
/// `lifecycle` bounds every task and subscription this function creates, directly —
/// the outer join task below, `Source 1`'s `Client` (and the `monitor_instance_source`
/// task it owns), and `Source 2`'s `base_runtime_config_watch` task. A caller scoped to
/// something narrower than the process, such as a monitor bound to one `WorkerSet`'s
/// lifecycle, must pass that scope's own token, or these background tasks outlive every
/// dropped reference the caller holds and leak until process shutdown.
pub async fn runtime_config_watch(
    endpoint: &Endpoint,
    lifecycle: CancellationToken,
) -> anyhow::Result<RuntimeConfigWatch> {
    let component = endpoint.component();
    let cancel_token = component.drt().primary_token();

    // Source 1: instance availability (watches DiscoveryQuery::Endpoint)
    let client = endpoint.client_with_cancellation(lifecycle.clone()).await?;
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

            let ready: HashMap<WorkerId, ModelRuntimeConfig> = instances
                .into_iter()
                .filter_map(|id| configs.get(&id).map(|cfg| (id, cfg.clone())))
                .collect();

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
