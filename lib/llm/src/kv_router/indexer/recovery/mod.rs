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
