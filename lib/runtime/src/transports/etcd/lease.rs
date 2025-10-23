// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Create a [`Lease`] with a given time-to-live (TTL) attached to the [`CancellationToken`].
pub async fn create_lease(
    mut lease_client: LeaseClient,
    ttl: u64,
    token: CancellationToken,
) -> Result<Lease> {
    let lease = lease_client.grant(ttl as i64, None).await?;

    let id = lease.id() as u64;
    let ttl = lease.ttl() as u64;
    let child = token.child_token();
    let clone = token.clone();

    tokio::spawn(async move {
        match keep_alive(lease_client, id, ttl, child).await {
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
pub async fn revoke_lease(mut lease_client: LeaseClient, lease_id: u64) -> Result<()> {
    match lease_client.revoke(lease_id as i64).await {
        Ok(_) => Ok(()),
        Err(e) => {
            tracing::warn!("failed to revoke lease: {:?}", e);
            Err(e.into())
        }
    }
}

/// Task to keep leases alive.
///
/// If this task returns an error, the cancellation token will be invoked on the runtime.
/// If
pub async fn keep_alive(
    client: LeaseClient,
    lease_id: u64,
    ttl: u64,
    token: CancellationToken,
) -> Result<()> {
    let mut ttl = ttl;
    let mut deadline = create_deadline(ttl)?;

    let mut client = client;
    let (mut heartbeat_sender, mut heartbeat_receiver) = client.keep_alive(lease_id as i64).await?;

    loop {
        // if the deadline is exceeded, then we have failed to issue a heartbeat in time
        // we may be permanently disconnected from the etcd server, so we are now officially done
        if deadline < std::time::Instant::now() {
            return Err(error!(
                "Unable to refresh lease - deadline exceeded. Check etcd server status"
            ));
        }

        tokio::select! {
            biased;

            status = heartbeat_receiver.message() => {
                if let Some(resp) = status? {
                    tracing::trace!(lease_id, "keep alive response received: {:?}", resp);

                    // update ttl and deadline
                    ttl = resp.ttl() as u64;
                    deadline = create_deadline(ttl)?;

                    if resp.ttl() == 0 {
                        return Err(error!("Unable to maintain lease - expired or revoked. Check etcd server status"));
                    }

                }
            }

            _ = token.cancelled() => {
                tracing::trace!(lease_id, "cancellation token triggered; revoking lease");
                let _ = client.revoke(lease_id as i64).await?;
                return Ok(());
            }

            _ = tokio::time::sleep(tokio::time::Duration::from_secs(ttl / 2)) => {
                tracing::trace!(lease_id, "sending keep alive");

                // if we get a error issuing the heartbeat, set the ttl to 0
                // this will allow us to poll the response stream once and the cancellation token once, then
                // immediately try to tick the heartbeat
                // this will repeat until either the heartbeat is reestablished or the deadline is exceeded
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

/// Create a deadline for a given time-to-live (TTL).
fn create_deadline(ttl: u64) -> Result<std::time::Instant> {
    Ok(std::time::Instant::now() + std::time::Duration::from_secs(ttl))
}
