// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-indexer remote-search driver.
//!
//! Spawned by [`InstanceLeader::find_matches_with_options`](super::InstanceLeader::find_matches_with_options)
//! when remote search is enabled and a [`RemoteBlockDiscovery`](super::RemoteBlockDiscovery)
//! is injected. The driver runs the targeted remote pull entirely over the
//! engine's transfer control plane:
//!
//! 1. Ask discovery (the hub's KV indexer) which instances hold the
//!    locally-uncached prefix.
//! 2. `open_session` on a candidate (holder) — pins its matching G2 prefix and
//!    returns a session endpoint.
//! 3. `pull_from_session` locally — attaches, RDMA-pulls, and **registers the
//!    pulled blocks into local G2**.
//! 4. `close_session` on the candidate (best-effort teardown).
//! 5. Re-match local G2 over the full slice (now including the pulled blocks)
//!    to fill the [`AsyncSessionResult`](super::AsyncSessionResult) block holder
//!    and set the terminal [`OnboardingStatus::Complete`].
//!
//! Any failure, timeout, or cancellation degrades to the local match — the
//! driver always reaches a terminal `Complete` so the connector's shard never
//! stays non-terminal (which would wedge `get_num_new_matched_tokens`).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;

use kvbm_logical::blocks::ImmutableBlock;
use kvbm_protocols::control::client::LeaderControlClient;
use kvbm_protocols::control::modules::transfer::{
    CloseTransferSessionRequest, FindMode, OpenTransferSessionRequest, OpenTransferSessionResponse,
    PullFromSessionRequest, SearchMode, TierSelection,
};

use super::InstanceLeader;
use super::discovery::RemoteDiscoveryHandle;
use super::onboarding::OnboardingStatus;
use super::session::SessionId;
use super::types::MatchBreakdown;
use crate::{G2, InstanceId, SequenceHash};

/// Wall-clock bound on the whole remote-search attempt. Sized to the holder's
/// transfer-session watchdog so a wedged pull degrades to the local match
/// rather than holding `get_num_new_matched_tokens` at `(None, false)`.
const REMOTE_SEARCH_WATCHDOG: Duration = Duration::from_secs(30);

/// Owns everything the background remote-search task needs. Built and spawned
/// by `find_matches_with_options`.
pub(super) struct RemoteSearchDriver {
    pub leader: Arc<InstanceLeader>,
    pub discovery: RemoteDiscoveryHandle,
    /// Full ascending-position sequence-hash slice for this find.
    pub sequence_hashes: Vec<SequenceHash>,
    /// Number of leading blocks already resident in local G2 (the synchronous
    /// local prefix match). The remote search targets the range after this.
    pub local_block_count: usize,
    /// Issue a remote search only when remaining uncached blocks exceed this.
    pub min_remote_blocks: usize,
    pub status_tx: watch::Sender<OnboardingStatus>,
    pub blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
    pub match_breakdown: Arc<Mutex<MatchBreakdown>>,
    /// Engine session id for this `AsyncSessionResult` (logging only).
    pub session_id: SessionId,
    pub cancel: CancellationToken,
}

impl RemoteSearchDriver {
    /// Run to terminal `Complete`. Always sends a terminal status.
    pub(super) async fn run(self) {
        let matched_blocks = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => {
                tracing::debug!(session_id = %self.session_id, "remote search cancelled");
                self.rematch_local().await
            }
            res = tokio::time::timeout(REMOTE_SEARCH_WATCHDOG, self.search()) => match res {
                Ok(Ok(n)) => n,
                Ok(Err(e)) => {
                    tracing::warn!(
                        error = %e, session_id = %self.session_id,
                        "remote search failed; degrading to local match"
                    );
                    self.rematch_local().await
                }
                Err(_) => {
                    tracing::warn!(
                        session_id = %self.session_id,
                        "remote search timed out; degrading to local match"
                    );
                    self.rematch_local().await
                }
            },
        };

        self.status_tx
            .send(OnboardingStatus::Complete { matched_blocks })
            .ok();
    }

    /// The happy-path search: discover → open → pull → close. Returns the
    /// contiguous local-G2 prefix length after the pull.
    async fn search(&self) -> Result<usize> {
        let total = self.sequence_hashes.len();
        let start = self.local_block_count.min(total);
        let remaining = &self.sequence_hashes[start..];

        // Threshold gate: search only when at least `min_remote_blocks` full
        // remote blocks remain (a sub-threshold round-trip never pays off).
        if remaining.len() < self.min_remote_blocks {
            return Ok(self.rematch_local().await);
        }

        let candidates = match self.discovery.discover(remaining.to_vec()).await? {
            Some(c) => c,
            None => return Ok(self.rematch_local().await),
        };

        // Target = the remaining prefix up to and including the indexer's
        // deepest placed hash.
        let Some(pos) = remaining.iter().position(|h| *h == candidates.deepest) else {
            return Ok(self.rematch_local().await);
        };
        let target: Vec<SequenceHash> = remaining[..=pos].to_vec();

        let self_id = self.leader.messenger().instance_id();
        for candidate in candidates.instances.iter().copied() {
            if candidate == self_id {
                continue;
            }
            if self.cancel.is_cancelled() {
                break;
            }
            match self.pull_from(candidate, &target).await {
                Ok(true) => break,
                Ok(false) => continue,
                Err(e) => {
                    tracing::debug!(
                        error = %e, %candidate,
                        "remote pull attempt failed; trying next candidate"
                    );
                    continue;
                }
            }
        }

        // Re-match local G2 over the full slice — the pulled blocks (and any
        // partial chunks registered before an error) are now resident, so the
        // contiguous prefix grows to include them.
        Ok(self.rematch_local().await)
    }

    /// Open a session on `candidate`, pull its committed prefix into local G2,
    /// and close the session. `Ok(true)` if blocks were committed/pulled,
    /// `Ok(false)` if the holder had nothing, `Err` on RPC/pull failure
    /// (partial chunks may still have landed in local G2).
    async fn pull_from(&self, candidate: InstanceId, target: &[SequenceHash]) -> Result<bool> {
        let client = LeaderControlClient::new(self.leader.messenger().clone(), candidate);

        let open = client
            .transfer()
            .open_session(OpenTransferSessionRequest {
                sequence_hashes: target.to_vec(),
                search_mode: SearchMode::Prefix,
                find_mode: FindMode::Sync,
                tiers: TierSelection::default(),
                watchdog_ms: None,
            })
            .await
            .map_err(|e| anyhow!("open_session on {candidate}: {e}"))?;

        let (capability, committed) = match open {
            OpenTransferSessionResponse::Sync {
                capability,
                committed,
                ..
            } => (capability, committed),
            OpenTransferSessionResponse::NoBlocksFound => return Ok(false),
            OpenTransferSessionResponse::Async { capability } => {
                // We requested Sync; an Async reply means nothing usable inline.
                self.close(&client, capability.session_id, "unexpected async open")
                    .await;
                return Ok(false);
            }
        };

        if committed.is_empty() {
            self.close(&client, capability.session_id, "no committed blocks")
                .await;
            return Ok(false);
        }

        // `selector: None` pulls every committed hash — i.e. the holder's
        // contiguous G2 prefix of `target` (its authoritative deepest match).
        let pull = self
            .leader
            .pull_from_session(PullFromSessionRequest {
                session_id: capability.session_id,
                source_instance_id: candidate,
                endpoint: Some(capability.endpoint),
                selector: None,
            })
            .await;

        self.close(&client, capability.session_id, "remote search complete")
            .await;

        match pull {
            Ok(_) => Ok(true),
            Err(e) => Err(anyhow!("pull_from_session from {candidate}: {e}")),
        }
    }

    /// Best-effort holder-side session teardown.
    async fn close(&self, client: &LeaderControlClient, session_id: uuid::Uuid, reason: &str) {
        if let Err(e) = client
            .transfer()
            .close_session(CloseTransferSessionRequest {
                session_id,
                reason: Some(reason.to_string()),
            })
            .await
        {
            tracing::debug!(error = %e, %session_id, "close_session failed (watchdog will reclaim)");
        }
    }

    /// Re-match local G2 over the full slice, fill the block holder + breakdown,
    /// and return the contiguous prefix length.
    async fn rematch_local(&self) -> usize {
        let matched = self.leader.g2_manager().match_blocks(&self.sequence_hashes);
        let n = matched.len();
        *self.match_breakdown.lock().await = MatchBreakdown {
            host_blocks: n,
            disk_blocks: 0,
            object_blocks: 0,
        };
        *self.blocks.lock().await = Some(matched);
        n
    }
}
