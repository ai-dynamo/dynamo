// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::discovery::KvSourceMembershipWatch;

mod broker_zmq;
mod direct_zmq;
mod recovery_lane;
mod source_health;
mod state_agent;
mod subscriber;
mod target;
mod worker_query;
mod worker_query_endpoint;
mod worker_query_state;
mod worker_query_transport;

pub(crate) use state_agent::start_state_agent_router;
pub(crate) use subscriber::{
    KvEventSubscriptionHandle, RecoverySupervisor, start_subscriber, start_target_subscriber,
};
pub(crate) use target::{IndexerRecoveryTarget, RecoveryResetReason, RecoveryTarget};
pub(crate) use worker_query::DEFAULT_RECOVERY_ATTEMPT_TIMEOUT;
pub(crate) use worker_query::TargetFaultDisposition;
#[cfg(test)]
pub(crate) use worker_query::WorkerQueryClient;
#[cfg(feature = "ckf-diagnostics")]
pub(crate) use worker_query::WorkerQueryHealthSnapshot;
pub(crate) use worker_query_endpoint::{
    start_worker_kv_query_endpoint, start_worker_kv_query_endpoint_with_status,
};
pub(crate) use worker_query_transport::RuntimeWorkerQueryTransport;
#[cfg(test)]
pub(crate) use worker_query_transport::WorkerQueryTransport;

/// Waits for `delay`, a membership change, or cancellation, whichever comes first.
/// Returns `false` on cancellation or a closed membership channel.
async fn wait_for_retry(
    delay: Duration,
    membership_watch: &mut KvSourceMembershipWatch,
    cancellation_token: &CancellationToken,
) -> bool {
    tokio::select! {
        _ = cancellation_token.cancelled() => false,
        changed = membership_watch.changed() => changed.is_ok(),
        _ = tokio::time::sleep(delay) => true,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use dynamo_runtime::{
        discovery::{Discovery, MockDiscovery, SharedMockRegistry},
        protocols::EndpointId,
    };
    use tokio::sync::watch;

    use super::*;
    use crate::discovery::{KvSourceMembershipCoordinator, KvSourceMembershipView};

    const LONG_DELAY: Duration = Duration::from_secs(3600);

    /// Returns a membership watch backed by a test-owned channel, so each test
    /// controls exactly which `select!` arm can become ready.
    fn membership_watch() -> (
        watch::Sender<KvSourceMembershipView>,
        KvSourceMembershipWatch,
    ) {
        let serving = EndpointId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            name: "generate".to_string(),
        };
        let (_configs_tx, configs_rx) = watch::channel(HashMap::new());
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));
        let base = KvSourceMembershipCoordinator::start(serving, configs_rx, discovery).subscribe();
        let (tx, rx) = watch::channel(base.borrow().clone());
        (tx, base.with_receiver(rx))
    }

    #[tokio::test]
    async fn wait_for_retry_returns_false_on_cancellation() {
        let (_tx, mut watch) = membership_watch();
        let cancellation_token = CancellationToken::new();
        cancellation_token.cancel();
        assert!(!wait_for_retry(LONG_DELAY, &mut watch, &cancellation_token).await);
    }

    #[tokio::test]
    async fn wait_for_retry_returns_false_on_closed_membership_channel() {
        let (tx, mut watch) = membership_watch();
        drop(tx);
        assert!(!wait_for_retry(LONG_DELAY, &mut watch, &CancellationToken::new()).await);
    }

    #[tokio::test]
    async fn wait_for_retry_returns_true_on_membership_update() {
        let (tx, mut watch) = membership_watch();
        tx.send_modify(|_| {});
        assert!(wait_for_retry(LONG_DELAY, &mut watch, &CancellationToken::new()).await);
    }

    #[tokio::test(start_paused = true)]
    async fn wait_for_retry_returns_true_after_delay() {
        let (_tx, mut watch) = membership_watch();
        assert!(
            wait_for_retry(
                Duration::from_millis(100),
                &mut watch,
                &CancellationToken::new()
            )
            .await
        );
    }
}
