// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

/// Create a [`Lease`] with a given time-to-live (TTL) attached to the [`CancellationToken`].
pub async fn create_lease(
    mut lease_client: LeaseClient,
    ttl: i64,
    token: CancellationToken,
) -> Result<Lease> {
    let lease = lease_client.grant(ttl, None).await?;

    let id = lease.id();
    let ttl = lease.ttl();
    let child = token.child_token();
    let clone = token.clone();

    tokio::spawn(async move {
        match keep_alive(lease_client, id, ttl, child).await {
            Ok(_) => tracing::trace!("keep alive task exited successfully"),
            Err(e) => {
                tracing::info!("keep alive task failed: {:?}", e);
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
pub async fn revoke_lease(mut lease_client: LeaseClient, lease_id: i64) -> Result<()> {
    match lease_client.revoke(lease_id).await {
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
    lease_id: i64,
    ttl: i64,
    token: CancellationToken,
) -> Result<()> {
    let mut ttl = ttl;
    let mut deadline = create_deadline(ttl)?;

    let mut client = client;
    let (mut heartbeat_sender, mut heartbeat_receiver) = client.keep_alive(lease_id).await?;

    loop {
        // if the deadline is exceeded, then we have failed to issue a heartbeat in time
        // we maybe be permanently disconnected from the etcd server, so we are now officially done
        if deadline < std::time::Instant::now() {
            return Err(error!("failed to issue heartbeat in time"));
        }

        tokio::select! {
            biased;

            status = heartbeat_receiver.message() => {
                if let Some(resp) = status? {
                    tracing::trace!(lease_id, "keep alive response received: {:?}", resp);

                    // update ttl and deadline
                    ttl = resp.ttl();
                    deadline = create_deadline(ttl)?;

                    if resp.ttl() == 0 {
                        return Err(error!("lease expired or revoked"));
                    }

                }
            }

            _ = token.cancelled() => {
                tracing::trace!(lease_id, "cancellation token triggered; revoking lease");
                let _ = client.revoke(lease_id).await?;
                return Ok(());
            }

            _ = tokio::time::sleep(tokio::time::Duration::from_secs(ttl as u64 / 2)) => {
                tracing::trace!(lease_id, "sending keep alive");

                // if we get a error issuing the heartbeat, set the ttl to 0
                // this will allow us to poll the response stream once and the cancellation token once, then
                // immediately try to tick the heartbeat
                // this will repeat until either the heartbeat is reestablished or the deadline is exceeded
                if let Err(e) = heartbeat_sender.keep_alive().await {
                    tracing::warn!(lease_id, "keep alive failed: {:?}", e);
                    ttl = 0;
                }
            }

        }
    }
}

/// Create a deadline for a given time-to-live (TTL).
fn create_deadline(ttl: i64) -> Result<std::time::Instant> {
    if ttl <= 0 {
        return Err(error!("invalid ttl: {}", ttl));
    }
    Ok(std::time::Instant::now() + std::time::Duration::from_secs(ttl as u64))
}
