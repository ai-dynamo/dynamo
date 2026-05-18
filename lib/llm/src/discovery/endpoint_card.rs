// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryQuery};
use dynamo_runtime::prelude::DistributedRuntimeProvider;

use crate::model_card::ModelDeploymentCard;

/// Wait for a worker on `endpoint` to publish its `ModelDeploymentCard`.
///
/// Uses the watch-based discovery API so the wait is event-driven (no polling)
/// and returns as soon as the first card is observed. Existing registrations
/// are delivered as `Added` events at the start of the stream, so callers do
/// not need to issue a separate `list` first.
///
/// Returns `Ok(Some(card))` once a card is observed, or `Ok(None)` if `timeout`
/// elapses, the supplied `cancel_token` fires, or the discovery stream ends
/// without ever delivering a deserializable model card. When `cancel_token` is
/// `None`, the runtime's primary token is used so the wait aborts on shutdown.
pub async fn wait_for_endpoint_model_card(
    endpoint: &Endpoint,
    timeout: Duration,
    cancel_token: Option<CancellationToken>,
) -> Result<Option<ModelDeploymentCard>> {
    let cancel_token = cancel_token.unwrap_or_else(|| endpoint.drt().primary_token());
    let eid = endpoint.id();
    let query = DiscoveryQuery::EndpointModels {
        namespace: eid.namespace,
        component: eid.component,
        endpoint: eid.name,
    };

    let mut stream = endpoint
        .drt()
        .discovery()
        .list_and_watch(query, Some(cancel_token.clone()))
        .await?;

    let find_card = async {
        while let Some(event) = stream.next().await {
            match event {
                Ok(DiscoveryEvent::Added(instance)) => {
                    if let Ok(card) = instance.deserialize_model::<ModelDeploymentCard>() {
                        return Some(card);
                    }
                }
                Ok(DiscoveryEvent::Removed(_)) => {}
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "Discovery event error while waiting for endpoint model card; continuing"
                    );
                }
            }
        }
        None
    };

    Ok(tokio::select! {
        card = find_card => card,
        _ = tokio::time::sleep(timeout) => None,
        _ = cancel_token.cancelled() => None,
    })
}
