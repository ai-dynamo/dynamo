// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::stream::StreamExt;

use dynamo_runtime::{
    DistributedRuntime,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, DiscoveryStream,
    },
};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::standalone_indexer::registry::WorkerRegistry;

/// Minimal subset of ModelDeploymentCard — only the fields the indexer needs.
/// Using `#[serde(default)]` on optional fields lets us safely ignore the rest.
#[derive(Deserialize, Debug)]
struct PartialModelCard {
    pub display_name: String,
    #[serde(default)]
    pub kv_cache_block_size: u32,
}

/// Spawn a background task that watches MDC discovery for worker additions/removals
/// and updates the WorkerRegistry accordingly.
pub async fn spawn_discovery_watcher(
    drt: &DistributedRuntime,
    registry: Arc<WorkerRegistry>,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    let discovery = drt.discovery();
    let mut stream: DiscoveryStream = discovery
        .list_and_watch(DiscoveryQuery::AllModels, Some(cancel_token.clone()))
        .await?;

    tokio::spawn(async move {
        tracing::info!("Discovery watcher started");

        while let Some(result) = stream.next().await {
            let event = match result {
                Ok(event) => event,
                Err(err) => {
                    tracing::error!(%err, "Error in discovery stream");
                    continue;
                }
            };

            match event {
                DiscoveryEvent::Added(instance) => {
                    let (instance_id, namespace, card) = match &instance {
                        DiscoveryInstance::Model {
                            instance_id,
                            namespace,
                            ..
                        } => match instance.deserialize_model::<PartialModelCard>() {
                            Ok(card) => (*instance_id, namespace.clone(), card),
                            Err(err) => {
                                tracing::error!(%err, instance_id, "Failed to deserialize model card");
                                continue;
                            }
                        },
                        _ => {
                            tracing::debug!("Ignoring non-model discovery instance");
                            continue;
                        }
                    };

                    let model_name = card.display_name.clone();
                    let block_size = card.kv_cache_block_size;
                    // Use the Dynamo namespace as the tenant_id
                    let tenant_id = namespace;

                    if block_size == 0 {
                        tracing::warn!(
                            instance_id,
                            model_name,
                            "Skipping worker with kv_cache_block_size=0"
                        );
                        continue;
                    }

                    tracing::info!(
                        instance_id,
                        model_name,
                        tenant_id,
                        block_size,
                        "Discovery: adding worker"
                    );

                    if let Err(e) = registry.add_worker_from_discovery(
                        instance_id,
                        model_name.clone(),
                        tenant_id,
                        block_size,
                    ) {
                        tracing::error!(
                            instance_id,
                            model_name,
                            error = %e,
                            "Failed to add discovered worker"
                        );
                    }
                }
                DiscoveryEvent::Removed(id) => {
                    let instance_id = match &id {
                        DiscoveryInstanceId::Model(mcid) => mcid.instance_id,
                        _ => {
                            tracing::debug!("Ignoring non-model discovery removal");
                            continue;
                        }
                    };

                    tracing::info!(instance_id, "Discovery: removing worker");
                    registry.remove_worker_from_discovery(instance_id).await;
                }
            }
        }

        tracing::info!("Discovery watcher exiting");
    });

    Ok(())
}
