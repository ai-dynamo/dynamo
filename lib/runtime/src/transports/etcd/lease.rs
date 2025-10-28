// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::connector::Connector;
use super::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};

/// Create a [`Lease`] with a given time-to-live (TTL) attached to the [`CancellationToken`].
pub async fn create_lease(
    connector: Arc<Connector>,
    ttl: u64,
    token: CancellationToken,
) -> Result<Lease> {
    let mut lease_client = connector.get_client().lease_client();
    let lease = lease_client.grant(ttl as i64, None).await?;

    let id = lease.id() as u64;
    let ttl = lease.ttl() as u64;
    let child = token.child_token();
    let clone = token.clone();

    tokio::spawn(async move {
        match keep_alive(connector, id, ttl, child).await {
            Ok(_) => tracing::trace!("keep alive task exited successfully"),
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "Unable to maintain lease. Check etcd server status"
                );
                token.cancel();
            }
        }
    });

    Ok(Lease {
        id,
        cancel_token: clone,
    })
}

/// Revoke a lease given its lease id. A wrapper over etcd_client::LeaseClient::revoke
pub async fn revoke_lease(connector: Arc<Connector>, lease_id: u64) -> Result<()> {
    let mut lease_client = connector.get_client().lease_client();
    match lease_client.revoke(lease_id as i64).await {
        Ok(_) => Ok(()),
        Err(e) => {
            tracing::warn!("failed to revoke lease: {:?}", e);
            Err(e.into())
        }
    }
}

/// Task to keep leases alive with reconnection support.
///
/// If this task returns an error, the cancellation token will be invoked on the runtime.
async fn keep_alive(
    connector: Arc<Connector>,
    lease_id: u64,
    mut ttl: u64,
    token: CancellationToken,
) -> Result<()> {
    let mut deadline = create_deadline(ttl)?;

    loop {
        // Try to establish or re-establish the keep-alive stream
        let mut lease_client = connector.get_client().lease_client();
        let (mut heartbeat_sender, mut heartbeat_receiver) = match lease_client
            .keep_alive(lease_id as i64)
            .await
        {
            Ok((sender, receiver)) => {
                tracing::debug!(lease_id, "Established keep-alive stream");
                (sender, receiver)
            }
            Err(e) => {
                tracing::warn!(lease_id, error = %e, "Failed to establish keep-alive stream");

                // Try to reconnect with the deadline, but also check for cancellation
                tokio::select! {
                    biased;

                    reconnect_result = connector.reconnect(deadline) => {
                        match reconnect_result {
                            Err(e) => return Err(e),
                            _ => continue,
                        }
                    }

                    _ = token.cancelled() => {
                        tracing::debug!(lease_id, "Cancellation token triggered during reconnection");
                        return Ok(());
                    }
                }
            }
        };

        // Keep-alive loop with the established stream
        loop {
            if deadline < std::time::Instant::now() {
                return Err(error!(
                    "Unable to refresh lease - deadline exceeded. Check etcd server status"
                ));
            }

            tokio::select! {
                biased;

                status = heartbeat_receiver.message() => {
                    match status {
                        Ok(Some(resp)) => {
                            tracing::trace!(lease_id, "keep alive response received: {:?}", resp);

                            // Update ttl and deadline from response
                            ttl = resp.ttl() as u64;
                            deadline = create_deadline(ttl)?;

                            if resp.ttl() == 0 {
                                return Err(error!("Unable to maintain lease - expired or revoked. Check etcd server status"));
                            }
                        }
                        Ok(None) => {
                            tracing::warn!(lease_id, "Keep-alive stream unexpectedly ended");
                            break;
                        }
                        Err(e) => {
                            tracing::warn!(lease_id, error = %e, "Keep-alive stream error");
                            break;
                        }
                    }
                }

                _ = token.cancelled() => {
                    tracing::debug!(lease_id, "cancellation token triggered; revoking lease");
                    if let Err(e) = lease_client.revoke(lease_id as i64).await {
                        tracing::warn!(
                            lease_id,
                            error = %e,
                            "Failed to revoke lease during cancellation. Cleanup may be incomplete."
                        );
                    }
                    return Ok(());
                }

                _ = tokio::time::sleep(Duration::from_secs(ttl / 2)) => {
                    tracing::trace!(lease_id, "sending keep alive");

                    // if we get a error issuing the heartbeat, set the ttl to 0
                    // this will allow us to poll the response stream once and the cancellation
                    // token once, then immediately try to tick the heartbeat
                    // this will repeat until either the heartbeat is reestablished or the deadline
                    // is exceeded
                    if let Err(e) = heartbeat_sender.keep_alive().await {
                        tracing::warn!(
                            lease_id,
                            error = %e,
                            "Unable to send lease heartbeat. Check etcd server status"
                        );
                        ttl = 0;
                    }
                }
            }
        }
    }
}

/// Create a deadline for a given time-to-live (TTL).
fn create_deadline(ttl: u64) -> Result<std::time::Instant> {
    Ok(std::time::Instant::now() + std::time::Duration::from_secs(ttl))
}
