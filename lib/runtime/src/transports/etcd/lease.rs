// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::connector::Connector;
use crate::runtime::Runtime;
use anyhow::Context;
use etcd_client::{LeaseKeepAliveStream, LeaseKeeper};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Create an etcd lease with the given TTL, tie its lifetime to the [`Runtime`],
/// spawn a keep-alive task, and return the lease id (u64).
///
/// Note: this function spawns a background task that maintains the lease until the runtime is
/// shut down or an unrecoverable error occurs. On an unrecoverable error the runtime is shut
/// down, honoring the contract that a lost lease shuts the worker down.
pub async fn create_lease(
    connector: Arc<Connector>,
    ttl: u64,
    runtime: Runtime,
) -> anyhow::Result<u64> {
    // Enroll before granting a lease: teardown must also await construction
    // that races cancellation, not just already-running keep-alive tasks.
    let teardown_guard = runtime.teardown_guard()?;
    let token = runtime.primary_token();
    if token.is_cancelled() {
        let _ = teardown_guard.send(Ok(()));
        anyhow::bail!("lease creation cancelled");
    }

    let mut lease_client = connector.get_client().lease_client();
    let lease = match lease_client.grant(ttl as i64, None).await {
        Ok(lease) => lease,
        Err(error) => {
            // No lease ID is available to revoke; preserve the grant failure
            // for shutdown as well as for the constructor's caller.
            let _ = teardown_guard.send(Err(anyhow::anyhow!(error.to_string())));
            return Err(error.into());
        }
    };

    let id = lease.id() as u64;
    if token.is_cancelled() {
        let result = lease_client
            .revoke(id as i64)
            .await
            .map(|_| ())
            .context("Failed to revoke lease after cancellation during creation");
        let error = result.as_ref().err().map(|error| format!("{error:#}"));
        let _ = teardown_guard.send(result);
        if let Some(error) = error {
            anyhow::bail!(error);
        }
        anyhow::bail!("lease creation cancelled");
    }

    let ttl = lease.ttl() as u64;
    let child = token.child_token();

    tokio::spawn(async move {
        let result = keep_alive(connector, id, ttl, child).await;
        match &result {
            Ok(_) => tracing::trace!("keep alive task exited successfully"),
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "Unable to maintain lease. Check etcd server status"
                );
                // Phased shutdown (endpoint drain -> backend teardown), not a
                // bare primary-token cancel, so teardown is ordered.
                runtime.shutdown();
            }
        }
        let _ = teardown_guard.send(result);
    });

    Ok(id)
}

/// Task to keep leases alive with reconnection support.
///
/// If this task returns an error, the cancellation token will be invoked on the runtime.
async fn keep_alive(
    connector: Arc<Connector>,
    lease_id: u64,
    ttl: u64,
    token: CancellationToken,
) -> anyhow::Result<()> {
    // Deadline when the lease expires
    let mut deadline = Instant::now() + Duration::from_secs(ttl);

    let mut reconnect = true;
    while reconnect {
        // Try to establish or re-establish the keep-alive stream
        let (sender, receiver) =
            match new_keep_alive_stream(&connector, lease_id, &deadline, &token).await? {
                Some(stream) => stream,
                None => break, // cancelled
            };

        // Keep-alive loop with the established stream
        reconnect =
            keep_alive_with_stream(sender, receiver, lease_id, &mut deadline, &token).await?;
    }
    // Both active streams and cancelled connection/reconnection attempts
    // converge here, and shutdown observes the revocation result.
    tracing::debug!(lease_id, "cancellation token triggered; revoking lease");
    connector
        .get_client()
        .lease_client()
        .revoke(lease_id as i64)
        .await
        .context("Failed to revoke lease during cancellation")?;
    Ok(())
}

/// Establish a new keep-alive stream with automatic retry and reconnection.
///
/// Returns:
///     `Ok(Some((LeaseKeeper, LeaseKeepAliveStream)))` on success.
///     `Ok(None)` if cancelled.
///     `Err` for unrecoverable errors such as deadline exceeded.
async fn new_keep_alive_stream(
    connector: &Arc<Connector>,
    lease_id: u64,
    deadline: &Instant,
    token: &CancellationToken,
) -> anyhow::Result<Option<(LeaseKeeper, LeaseKeepAliveStream)>> {
    loop {
        let mut lease_client = connector.get_client().lease_client();
        let result = tokio::select! {
            biased;
            _ = token.cancelled() => return Ok(None),
            result = lease_client.keep_alive(lease_id as i64) => result,
        };
        match result {
            Ok((sender, receiver)) => {
                tracing::debug!(lease_id, "Established keep-alive stream");
                return Ok(Some((sender, receiver))); // success
            }
            Err(e) => {
                tracing::warn!(lease_id, error = %e, "Failed to establish keep-alive stream");

                // Try to reconnect with the deadline, but also check for cancellation
                tokio::select! {
                    biased;

                    _ = token.cancelled() => {
                        tracing::debug!(lease_id, "Cancellation token triggered during reconnection");
                        return Ok(None); // cancelled
                    }

                    reconnect_result = connector.reconnect(*deadline) => {
                        match reconnect_result {
                            Err(e) => return Err(e), // cannot reconnect
                            _ => continue, // retry
                        }
                    }
                }
            }
        };
    }
}

/// Keep-alive loop that maintains the lease using the provided sender and receiver.
///
/// Returns:
///     `Ok(true)` for recoverable errors such as stream closure that warrant reconnection attempts.
///     `Ok(false)` if cancelled.
///     `Err` for unrecoverable errors such as lease already expired.
async fn keep_alive_with_stream(
    mut sender: LeaseKeeper,
    mut receiver: LeaseKeepAliveStream,
    lease_id: u64,
    deadline: &mut Instant,
    token: &CancellationToken,
) -> anyhow::Result<bool> {
    loop {
        let next_renewal = deadline
            .saturating_duration_since(Instant::now())
            .div_f64(2.0);

        tokio::select! {
            biased;

            status = receiver.message() => {
                match status {
                    Ok(Some(resp)) => {
                        tracing::trace!(lease_id, "keep alive response received: {:?}", resp);
                        // Update deadline from response
                        let ttl = resp.ttl();
                        if ttl <= 0 {
                            tracing::error!(lease_id, "Keep-alive lease expired");
                            anyhow::bail!("Unable to maintain lease - expired or revoked. Check etcd server status");
                        }
                        *deadline = Instant::now() + Duration::from_secs(ttl as u64);
                    }
                    Ok(None) => {
                        tracing::warn!(lease_id, "Keep-alive stream unexpectedly ended");
                        return Ok(true); // Exit to reconnect
                    }
                    Err(e) => {
                        tracing::warn!(lease_id, error = %e, "Keep-alive stream error");
                        return Ok(true); // Exit to reconnect
                    }
                }
            }

            _ = token.cancelled() => {
                return Ok(false);
            }

            _ = tokio::time::sleep(next_renewal) => {
                tracing::trace!(lease_id, "sending keep alive");
                if let Err(e) = sender.keep_alive().await {
                    tracing::warn!(
                        lease_id,
                        error = %e,
                        "Unable to send lease heartbeat. Check etcd server status"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lease-loss teardown path (`runtime.shutdown()`) must be phased:
    /// Phase 1 cancels the endpoint token promptly, but Phase 3 (primary token
    /// + backend teardown) waits for outstanding graceful tasks.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lease_loss_shutdown_is_phased() {
        let runtime = Runtime::from_current().unwrap();

        // Stands in for a serving endpoint; Phase 1 cancels the endpoint
        // shutdown token, cascading to this child.
        let endpoint_token = runtime.child_token();
        let primary_token = runtime.primary_token();

        // Hold a graceful task so Phase 2 cannot advance to Phase 3 yet.
        let guard = runtime.graceful_shutdown_tracker().register_task();

        runtime.shutdown();

        // Phase 1 cancels the endpoint token promptly.
        tokio::time::timeout(Duration::from_secs(5), endpoint_token.cancelled())
            .await
            .expect("Phase 1 must cancel the endpoint token on lease loss");

        // While the graceful task is outstanding, Phase 3 must not have torn
        // down the primary token. A bare-cancel regression fails this assert.
        assert!(
            !primary_token.is_cancelled(),
            "primary token must not be cancelled while a graceful task is outstanding"
        );

        // Releasing the task lets Phase 3 proceed and cancel the primary token.
        drop(guard);
        tokio::time::timeout(Duration::from_secs(5), primary_token.cancelled())
            .await
            .expect("Phase 3 must cancel the primary token once graceful tasks complete");
    }
}
