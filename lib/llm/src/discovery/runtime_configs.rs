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

use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;
use dynamo_kv_router::protocols::WorkerId;

/// Type alias for the runtime config watch receiver.
pub type RuntimeConfigWatch = watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>;

fn base_runtime_config_watch(
    mut stream: DiscoveryStream,
) -> watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>> {
    let (tx, rx) = watch::channel(HashMap::new());

    tokio::spawn(async move {
        let mut configs = HashMap::new();
        while let Some(result) = stream.next().await {
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
pub async fn runtime_config_watch(endpoint: &Endpoint) -> anyhow::Result<RuntimeConfigWatch> {
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
    let mut configs_rx = base_runtime_config_watch(stream);

    let (tx, rx) = watch::channel(HashMap::new());

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => break,
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
    use dynamo_runtime::discovery::DiscoveryInstance;

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

    #[tokio::test]
    async fn only_base_cards_define_runtime_config_expectations() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream: DiscoveryStream =
            Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
        let mut configs = base_runtime_config_watch(stream);
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
}
