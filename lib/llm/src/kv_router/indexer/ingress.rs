// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's KV-event ingress: builds the router's index and feeds it from
//! the Dynamo runtime (event plane subscription with per-rank recovery, and the
//! served-indexer endpoint when configured). Worker membership comes from the
//! KV-source membership watch, so the per-worker catalog hooks are no-ops.

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::services::indexer::registry::WorkerRegistry;
use dynamo_kv_router::services::selection::KvEventIngress;
use dynamo_runtime::component::Endpoint;
use tokio_util::sync::CancellationToken;

use super::{
    Indexer, KvEventSubscriptionHandle, ServedIndexerMode, ensure_served_indexer_service,
    start_subscriber,
};
use crate::discovery::KvSourceMembershipWatch;
use crate::kv_router::KvEventSourceRequirement;

pub(crate) struct RuntimeIngressArgs<'a> {
    pub endpoint: &'a Endpoint,
    pub kv_router_config: &'a KvRouterConfig,
    pub block_size: u32,
    pub model_name: Option<&'a str>,
    pub worker_role: Option<WorkerType>,
    pub metric_worker_type: &'static str,
    /// Whether any consumer needs KV knowledge; without it the index is `None`.
    pub cache_required: bool,
    pub kv_event_source_requirement: KvEventSourceRequirement,
    pub kv_source_membership: Option<KvSourceMembershipWatch>,
    pub cancellation_token: CancellationToken,
}

pub(crate) struct RuntimeIngress {
    indexer: Indexer,
    subscription: parking_lot::Mutex<Option<KvEventSubscriptionHandle>>,
    _served: Option<super::ServedIndexerHandle>,
}

impl RuntimeIngress {
    pub(crate) async fn start(args: RuntimeIngressArgs<'_>) -> Result<Arc<Self>> {
        let component = args.endpoint.component();
        let indexer = if args.cache_required {
            super::build(
                component,
                args.kv_router_config,
                args.block_size,
                args.model_name,
                args.cancellation_token.child_token(),
            )
            .await?
        } else {
            Indexer::None
        };

        let subscription = if args.cache_required
            && args
                .kv_event_source_requirement
                .should_subscribe(args.kv_router_config)
        {
            let membership_watch = args.kv_source_membership.ok_or_else(|| {
                anyhow::anyhow!(
                    "KV source membership watch is required when local KV event subscription is enabled"
                )
            })?;
            Some(
                start_subscriber(
                    args.endpoint.clone(),
                    indexer.clone(),
                    membership_watch,
                    args.block_size,
                    args.model_name.unwrap_or("unknown").to_string(),
                    args.worker_role,
                    args.kv_event_source_requirement,
                    args.metric_worker_type,
                    args.cancellation_token.child_token(),
                )
                .await?,
            )
        } else {
            tracing::info!(
                requirement = %args.kv_event_source_requirement,
                cache_required = args.cache_required,
                "Skipping KV event subscription (use_kv_events={}, overlap_score_credit={}, use_remote_indexer={})",
                args.kv_router_config.use_kv_events,
                args.kv_router_config.overlap_score_credit,
                args.kv_router_config.use_remote_indexer,
            );
            None
        };

        let served = if args.kv_router_config.serve_indexer {
            let model_name = args.model_name.ok_or_else(|| {
                anyhow::anyhow!("model_name is required when serve_indexer is configured")
            })?;
            Some(
                ensure_served_indexer_service(
                    component.clone(),
                    ServedIndexerMode::from_use_kv_events(args.kv_router_config.use_kv_events),
                    model_name.to_string(),
                    indexer.clone(),
                )
                .await?,
            )
        } else {
            None
        };

        Ok(Arc::new(Self {
            indexer,
            subscription: parking_lot::Mutex::new(subscription),
            _served: served,
        }))
    }

    pub(crate) fn indexer(&self) -> &Indexer {
        &self.indexer
    }

    #[cfg(test)]
    pub(crate) fn has_subscription(&self) -> bool {
        self.subscription.lock().is_some()
    }

    pub(crate) fn set_task_guard(&self, task_guard: dynamo_runtime::engine::EngineContextGuard) {
        if let Some(subscription) = self.subscription.lock().as_mut() {
            subscription.set_task_guard(task_guard);
        }
    }

    /// Stop event ingestion and wait for it to drain.
    pub(crate) async fn shutdown(&self) {
        let subscription = self.subscription.lock().take();
        if let Some(subscription) = subscription {
            subscription.shutdown().await;
        }
    }
}

#[async_trait::async_trait]
impl KvEventIngress for RuntimeIngress {
    fn open(
        &self,
        _registry: &WorkerRegistry,
        _key: &RoutingPartitionId,
        _block_size: u32,
    ) -> Indexer {
        self.indexer.clone()
    }
}
