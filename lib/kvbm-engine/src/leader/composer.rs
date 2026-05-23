// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Onboarding composer: layered tier composition for AsyncSession finds.
//!
//! Spawned by [`InstanceLeader::find_matches_with_options`] whenever the
//! synchronous local match leaves work to do (local G3 to stage, a remote
//! tail to pull, or both). The composer overlaps the discovery RPC with the
//! G3→G2 staging transfer, then issues the remote pull against the **actual**
//! post-staging local prefix — a final single
//! [`BlockManager::match_blocks`](kvbm_logical::manager::BlockManager::match_blocks)
//! over the full sequence collapses whatever each tier registered into the
//! contiguous prefix the caller sees.
//!
//! ## Tier composition
//!
//! - **G3→G2 staging**: when [`InstanceLeader::parallel_worker`] is present and
//!   the synchronous G3 match returned blocks, stage them into G2 via
//!   [`stage_g3_to_g2`]. Staging signals `staging_settled` on completion
//!   (success or failure) via a [`Drop`] guard so the remote-pull continuation
//!   can compute its target from the truthful post-staging G2 prefix.
//! - **Hub-indexer remote pull**: when remote search is enabled the composer
//!   builds a [`remote_search::DiscoveryPlan`] over the *expected* post-
//!   staging remaining range, fires the (decimated, ≤64-hash) query
//!   concurrently with staging, then — after staging has settled and we have
//!   the truthful post-staging local prefix — resolves the indexer's
//!   `deepest` sample into a pin range and pulls from the first reachable
//!   candidate. The decimation + pin-target arithmetic lives in
//!   [`DiscoveryPlan`](remote_search::DiscoveryPlan) so the composer is just
//!   glue.
//!
//! Decoupling the pull target from the *expected* post-staging position
//! (`local_g2_count + local_g3_count`) closes two races the optimistic
//! projection would hit: (1) the bridge-hole case — G2 already holds blocks
//! past the staging range so the post-staging prefix is longer than the sum;
//! (2) the staging-failure case — the prefix is just `local_g2_count` because
//! G3 contributed nothing. Either way, the re-match below feeds the pull the
//! truthful starting position.
//!
//! ## Failure model
//!
//! Either child failing (staging error, discovery miss, pull RPC failure,
//! cancellation, timeout) is logged and degrades to "no contribution from
//! that tier". The composer always reaches a terminal
//! [`OnboardingStatus::Complete`] so the connector's shard never wedges
//! non-terminal — the contiguous prefix that did land determines
//! `matched_blocks`.

use std::sync::Arc;

use kvbm_logical::blocks::ImmutableBlock;
use tokio::sync::{Mutex, Notify, watch};
use tokio_util::sync::CancellationToken;

use super::InstanceLeader;
use super::blocks::BlockHolder;
use super::onboarding::OnboardingStatus;
use super::remote_search;
use super::staging::stage_g3_to_g2;
use super::types::{MatchBreakdown, SessionId};
use crate::{G2, G3, SequenceHash};

/// Orchestrates the AsyncSession onboarding path. Built and spawned by
/// [`InstanceLeader::find_matches_with_options`].
pub(super) struct OnboardingComposer {
    pub leader: Arc<InstanceLeader>,
    /// Full ascending-position sequence-hash slice for this find.
    pub sequence_hashes: Vec<SequenceHash>,
    /// Local G3 blocks matched synchronously (the staging input).
    pub matched_g3_blocks: Vec<ImmutableBlock<G3>>,
    /// Synchronous local G2 contiguous-prefix length. Combined with
    /// `matched_g3_blocks.len()` to anchor [`remote_search::DiscoveryPlan`]
    /// at the expected post-staging remaining range — the indexer never
    /// walks hashes the requester already holds (or expects to hold once
    /// staging settles). The pull target's *lower* bound uses the post-
    /// staging re-matched prefix (`current_prefix`) instead, which can be
    /// longer when staging bridges a hole.
    pub local_g2_count: usize,
    /// When true, the composer issues a hub-indexer discovery + remote pull.
    pub use_remote_search: bool,
    /// Threshold (in blocks) under which the remote pull is skipped.
    pub min_remote_blocks: usize,
    pub status_tx: watch::Sender<OnboardingStatus>,
    pub all_g2_blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
    pub match_breakdown: Arc<Mutex<MatchBreakdown>>,
    pub cancel: CancellationToken,
    pub session_id: SessionId,
    /// Signalled by the staging future's `Drop` guard once G3→G2 has settled
    /// (success, failure, or no-op). The remote-pull continuation awaits this
    /// after discovery returns so its target projection sees the truthful
    /// local prefix rather than the optimistic `local_g2_count + local_g3_count`.
    pub staging_settled: Arc<Notify>,
}

/// RAII signal: fires `staging_settled.notify_one()` on drop so every exit
/// path of `stage_g3_into_g2` (success, early-return, error, panic) wakes the
/// remote-pull continuation. The notification is paired with exactly one
/// `notified()` await; a `notify_one` permit is stored if the await hasn't
/// started yet, so ordering between staging and discovery doesn't matter.
struct StagingSignal<'a> {
    notify: &'a Notify,
}

impl Drop for StagingSignal<'_> {
    fn drop(&mut self) {
        self.notify.notify_one();
    }
}

impl OnboardingComposer {
    /// Run to terminal `Complete`. Always emits a terminal status, so a
    /// cancelled or fully-failed composition still drains its watch receiver.
    pub(super) async fn run(self) {
        self.run_tiers().await;
        self.finalize().await;
    }

    /// Run the active tiers concurrently. The discovery RPC overlaps with the
    /// G3→G2 staging transfer; the remote pull only starts once staging has
    /// settled, so its target projection sees the truthful post-staging G2
    /// prefix (closes the bridge-hole and staging-failure races).
    ///
    /// **Cancellation contract.** The [`CancellationToken`] gates *idle*
    /// points only — it must never abort a mid-flight DMA. Two surfaces care:
    ///
    /// - `stage_g3_into_g2` runs unconditionally to completion. Aborting
    ///   `stage_g3_to_g2` mid-flight would return G2 destination blocks to
    ///   the allocator pool while the underlying transfer is still writing
    ///   into them. `release_session` waits for staging to settle (a local
    ///   memcpy, sub-ms in practice) and then reaches `finalize`.
    /// - `discover_and_pull` spawns each `remote_search::pull_from` as a
    ///   detachable task. On cancel mid-pull, the `JoinHandle` is dropped
    ///   (the spawned task continues, finishes its RDMA safely, and
    ///   registers whatever blocks it pulled into the G2 registry for
    ///   future use); the composer's main task proceeds to `finalize`
    ///   immediately so the `AsyncSession` reaches `Complete` promptly
    ///   instead of waiting on the holder's session watchdog.
    async fn run_tiers(&self) {
        tokio::join!(self.stage_g3_into_g2(), self.discover_and_pull());
    }

    /// Stage local G3 matches into G2. No-op when the synchronous G3 match
    /// returned nothing or no parallel worker is configured. The RAII signal
    /// fires on every exit path so the remote-pull continuation never blocks
    /// forever.
    async fn stage_g3_into_g2(&self) {
        let _signal = StagingSignal {
            notify: &self.staging_settled,
        };
        if self.matched_g3_blocks.is_empty() {
            return;
        }
        let Some(parallel_worker) = self.leader.parallel_worker() else {
            tracing::warn!(
                session_id = %self.session_id,
                "no parallel worker configured; cannot stage G3, delivering G2 prefix only"
            );
            return;
        };
        let holder = BlockHolder::new(self.matched_g3_blocks.clone());
        if let Err(e) = stage_g3_to_g2(&holder, self.leader.g2_manager(), &*parallel_worker).await {
            tracing::warn!(
                session_id = %self.session_id,
                error = %e,
                "local G3→G2 staging failed; delivering G2 prefix only"
            );
        }
    }

    /// Hub-indexer discovery + targeted remote pull, bounded by
    /// [`remote_search::REMOTE_SEARCH_WATCHDOG`]. The discovery RPC overlaps
    /// with the G3 staging transfer; the pull target is computed only after
    /// staging settles, against the truthful post-staging local G2 prefix.
    async fn discover_and_pull(&self) {
        if !self.use_remote_search {
            return;
        }
        let Some(discovery) = self.leader.remote_discovery() else {
            return;
        };

        let outcome = tokio::time::timeout(
            remote_search::REMOTE_SEARCH_WATCHDOG,
            self.do_discover_and_pull(discovery),
        )
        .await;
        if outcome.is_err() {
            tracing::warn!(
                session_id = %self.session_id,
                "remote search timed out; degrading to local match"
            );
        }
    }

    async fn do_discover_and_pull(&self, discovery: super::discovery::RemoteDiscoveryHandle) {
        // Build the decimated discovery plan against the *expected* post-
        // staging remaining range. Two synchronously-available counts —
        // `local_g2_count` + `matched_g3_blocks.len()` — define the start of
        // remaining; assuming staging succeeds is the optimistic case, and
        // the post-staging re-match below corrects via `current_prefix`.
        let expected_remaining_start = self.local_g2_count + self.matched_g3_blocks.len();
        let plan =
            remote_search::DiscoveryPlan::new(&self.sequence_hashes, expected_remaining_start);
        if plan.is_empty() || plan.remaining_len() < self.min_remote_blocks {
            return;
        }

        // Fire the discovery RPC. Cancellable — read-only, safe to drop.
        // Overlaps with `stage_g3_into_g2` in the parent join.
        let candidates = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => {
                tracing::debug!(session_id = %self.session_id, "remote search cancelled before discovery");
                return;
            }
            res = discovery.discover(plan.query_hashes().to_vec()) => match res {
                Ok(Some(c)) => c,
                Ok(None) => return,
                Err(e) => {
                    tracing::warn!(
                        session_id = %self.session_id,
                        error = %e,
                        "discovery RPC failed; degrading to local match"
                    );
                    return;
                }
            }
        };

        // Wait for staging to settle (success, failure, or no-op) so the pin
        // target's lower bound reflects the truthful local prefix.
        // Cancellation here only bypasses the pull — the staging future in
        // the parent join continues to completion (see `run_tiers` note).
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => {
                tracing::debug!(session_id = %self.session_id, "remote search cancelled before pull");
                return;
            }
            _ = self.staging_settled.notified() => {}
        }
        let current_prefix = self
            .leader
            .g2_manager()
            .match_blocks(&self.sequence_hashes)
            .len();

        // Resolve the indexer's deepest sample into a pin range. None means
        // either staging fully covered the indexer's reach or the reply was
        // unusable; either way we degrade to local match.
        let Some(pin_range) = plan.resolve_pin_target(candidates.deepest, current_prefix) else {
            return;
        };
        let target: Vec<SequenceHash> = self.sequence_hashes[pin_range].to_vec();

        let self_id = self.leader.messenger().instance_id();
        for candidate in candidates.instances.iter().copied() {
            if candidate == self_id {
                continue;
            }
            if self.cancel.is_cancelled() {
                break;
            }

            // Spawn each `pull_from` so cancellation can **detach** the
            // in-flight task rather than aborting it mid-RDMA. Aborting a
            // `session.pull(..)` mid-flight would return G2 destination
            // blocks to the allocator pool while the underlying NIXL
            // transfer is still writing into them — the same corruption
            // risk as aborting staging. Detaching lets the current pull run
            // to safe completion in the background; the composer's main
            // task proceeds to `finalize` and the AsyncSession reaches
            // `Complete` promptly (rather than hanging until the holder's
            // session watchdog or this driver's `REMOTE_SEARCH_WATCHDOG`).
            let leader = self.leader.clone();
            let target_for_task = target.clone();
            let pull_handle = tokio::spawn(async move {
                remote_search::pull_from(&leader, candidate, &target_for_task).await
            });

            let pull_outcome = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => {
                    tracing::debug!(
                        session_id = %self.session_id,
                        %candidate,
                        "remote pull cancelled mid-flight; detaching task to complete safely \
                         in background"
                    );
                    return;
                }
                res = pull_handle => res,
            };

            match pull_outcome {
                Ok(Ok(true)) => break,
                Ok(Ok(false)) => continue,
                Ok(Err(e)) => {
                    tracing::debug!(
                        error = %e,
                        %candidate,
                        "remote pull attempt failed; trying next candidate"
                    );
                    continue;
                }
                Err(join_err) => {
                    tracing::debug!(
                        error = %join_err,
                        %candidate,
                        "pull_from task aborted unexpectedly; trying next candidate"
                    );
                    continue;
                }
            }
        }
    }

    /// Single final re-match: collapse whatever each tier registered in G2
    /// into the contiguous prefix the caller will see. Writes the result
    /// holder + breakdown and emits the terminal status.
    async fn finalize(self) {
        let matched = self.leader.g2_manager().match_blocks(&self.sequence_hashes);
        let n = matched.len();
        *self.match_breakdown.lock().await = MatchBreakdown {
            host_blocks: n,
            disk_blocks: 0,
            object_blocks: 0,
        };
        *self.all_g2_blocks.lock().await = Some(matched);
        self.status_tx
            .send(OnboardingStatus::Complete { matched_blocks: n })
            .ok();
    }
}
