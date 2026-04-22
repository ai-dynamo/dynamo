// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_engine::offload::{TransferHandle, TransferStatus};

use super::*;
use slot::OnboardingState;

impl ConnectorLeader {
    /// Mark a request as finished, returning the status.
    ///
    /// Dispatches on the slot's current `TransactionState`:
    /// - `Inactive`: the request has no outstanding work; the slot is removed
    ///   synchronously and `Finished` is returned.
    /// - `PreparingToOnboard`: the request was cancelled mid-search, before
    ///   onboarding was committed to the scheduler or the workers. The
    ///   `OnboardingState` is extracted, the slot is removed synchronously,
    ///   and a fire-and-forget task drains every shard's find session and
    ///   releases it. No worker callbacks are emitted because no worker ever
    ///   learned about this request. Returns `Finished`.
    /// - `Offloading`: outstanding offload handles must complete before
    ///   workers can be signalled. If all handles are already complete the
    ///   slot is removed and `Finished` is returned; otherwise a cleanup
    ///   task is spawned, `mark_offloading_complete` will be issued on
    ///   completion, and `Pending` is returned.
    /// - `Onboarding` / `Error`: not handled in this path yet; the caller
    ///   sees the pre-existing log-only behaviour.
    #[tracing::instrument(level = "debug", skip(self), fields(?request_id), ret)]
    pub fn request_finished(&self, request_id: &str) -> FinishedStatus {
        tracing::debug!("evaluating finished status");

        let Some(shared_slot) = self.slots.get(request_id).map(|slot| slot.clone()) else {
            return FinishedStatus::UntrackedRequest;
        };

        let mut slot = shared_slot.lock();

        // Cancel path: request is finished while still in PreparingToOnboard.
        // No worker callback will ever arrive (scheduler never committed), so we
        // own the whole cleanup here — remove the slot synchronously and hand
        // the OnboardingState to a background task for session release.
        if matches!(slot.txn_state(), TransactionState::PreparingToOnboard(_)) {
            match slot.txn_take_preparing_to_onboard() {
                Ok(onboarding_state) => {
                    drop(slot);
                    self.remove_slot(request_id);
                    self.spawn_preparing_to_onboard_cleanup(request_id, onboarding_state);
                    return FinishedStatus::Finished;
                }
                Err(e) => {
                    // txn_state() said PreparingToOnboard, so this branch should
                    // be unreachable. Fall through to the legacy path (which
                    // will log and return Pending) rather than panicking.
                    tracing::error!(
                        "Failed to take PreparingToOnboard state for request ID {}: {}",
                        request_id,
                        e
                    );
                }
            }
        }

        // Mark the slot for deletion
        let initial_status = slot.slot_mark_finished();

        // If immediately finished (no active transaction), we're done
        if matches!(initial_status, FinishedStatus::Finished) {
            drop(slot);
            self.remove_slot(request_id);
            return FinishedStatus::Finished;
        }

        // Try to take the offloading state to check handles
        match slot.txn_take_offloading() {
            Ok(offloading_state) => {
                let handles = offloading_state.handles;

                // Check if all handles are already complete
                let all_complete = handles.iter().all(|h| h.is_complete());

                if all_complete {
                    // All transfers are done, we can finish immediately
                    tracing::debug!(
                        "all {} transfer handles complete for request_id: {}",
                        handles.len(),
                        request_id
                    );
                    drop(slot);
                    self.remove_slot(request_id);
                    return FinishedStatus::Finished;
                }

                // Some handles are still in progress - spawn cleanup task
                let incomplete_count = handles.iter().filter(|h| !h.is_complete()).count();
                tracing::debug!(
                    "{} of {} transfer handles still in progress for request_id: {}, spawning cleanup task",
                    incomplete_count,
                    handles.len(),
                    request_id
                );

                let request_id_owned = request_id.to_string();
                let workers = self.workers.get().unwrap().clone();

                self.runtime.tokio().spawn(async move {
                    cleanup_offloading_handles(handles, &request_id_owned).await;

                    if let Err(e) = workers
                        .mark_offloading_complete(request_id_owned.clone())
                        .await
                    {
                        tracing::error!(
                            "Error marking offloading complete for request ID: {}: {}",
                            request_id_owned,
                            e
                        );
                    }
                    tracing::debug!(
                        "marked offloading complete for request ID: {}",
                        request_id_owned
                    );
                });

                FinishedStatus::Pending
            }
            Err(e) => {
                // No offloading state or in wrong state - just return the initial status
                // This could happen if the slot was in a different transaction state
                tracing::error!(
                    "Error taking offloading state for request ID: {}: {}",
                    request_id,
                    e
                );
                initial_status
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self), fields(finished_sending = finished_sending.len(), finished_recving = finished_recving.len()))]
    pub fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        // process the requests that have finished onboarding
        // recving ==> remote kv storage -> worker g1 memory
        for request_id in finished_recving {
            match self.process_finished_onboarding(&request_id) {
                Ok(()) => {
                    tracing::debug!("finished onboarding for request ID: {}", request_id);
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to process finished onboarding for request ID: {}: {}",
                        request_id,
                        e
                    );
                }
            }
        }

        // Process the requests that have finished offloading
        // These requests should be marked for deletion but are waiting for the outstanding operations
        // to be complete. This is that signal.
        for request_id in finished_sending {
            match self.process_finished_offloading(&request_id) {
                Ok(()) => {
                    tracing::debug!("finished offloading for request ID: {}", request_id);
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to process finished offloading for request ID: {}: {}",
                        request_id,
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Spawn a fire-and-forget task that drains the find sessions of a
    /// cancelled-in-`PreparingToOnboard` request and releases each one.
    ///
    /// The caller must have already removed the slot from the map and taken
    /// ownership of the `OnboardingState`; this function only touches the
    /// extracted state and the `InstanceLeader`.
    fn spawn_preparing_to_onboard_cleanup(
        &self,
        request_id: &str,
        onboarding_state: OnboardingState,
    ) {
        let Some(instance_leader) = self.instance_leader.get().cloned() else {
            // InstanceLeader is set during worker initialization, well before
            // any request reaches PreparingToOnboard. If it is missing we have
            // a serious lifecycle bug — log and drop the state (RAII releases
            // Ready shards; AsyncSession server-side state will be cleaned up
            // when the leader shuts down).
            tracing::error!(
                "InstanceLeader not available; dropping OnboardingState without releasing sessions for request_id: {}",
                request_id
            );
            return;
        };

        let request_id_owned = request_id.to_string();
        self.runtime.tokio().spawn(async move {
            cleanup_preparing_to_onboard(onboarding_state, instance_leader, request_id_owned)
                .await;
        });
    }

    fn remove_slot(&self, request_id: &str) {
        let Some((_, shared_slot)) = self.slots.remove(request_id) else {
            tracing::debug!("Slot not found for request ID: {}", request_id);
            return;
        };

        // todo: if we need to access anything on the slot to clean up, we should do it here.
        drop(shared_slot);
    }

    fn process_finished_onboarding(&self, request_id: &str) -> Result<()> {
        let shared_slot = self
            .slots
            .get(request_id)
            .map(|slot| slot.clone())
            .ok_or_else(|| anyhow!("Slot not found for request ID: {}", request_id))?;

        let mut slot = shared_slot.lock();
        let onboarding_state = slot.txn_take_onboarding()?;
        if let Some(session_id) = onboarding_state.find_session.session_id() {
            self.instance_leader
                .get()
                .unwrap()
                .release_session(session_id);
        }
        Ok(())
    }

    fn process_finished_offloading(&self, request_id: &str) -> Result<()> {
        let shared_slot = self
            .slots
            .get(request_id)
            .map(|slot| slot.clone())
            .ok_or_else(|| anyhow!("Slot not found for request ID: {}", request_id))?;

        {
            let slot = shared_slot.lock();

            // The slot should be Inactive - the cleanup task spawned by request_finished()
            // already took the offloading state and validated handle completion.
            // By the time we get here via update_connector_output(finished_sending),
            // the cleanup task has completed and the slot is Inactive.
            if !slot.txn_state().is_inactive() {
                debug_assert!(
                    false,
                    "process_finished_offloading called but slot is in {:?} state, expected Inactive. \
                     This indicates the cleanup task did not complete properly for request_id: {}",
                    slot.txn_state().name(),
                    request_id
                );
            }
        }

        // Remove the slot from the map - this is the final cleanup step
        self.remove_slot(request_id);

        tracing::debug!(
            "removed slot for finished offloading request_id: {}",
            request_id
        );

        Ok(())
    }
}

/// Async cleanup task for offloading handles.
///
/// This function:
/// 1. Requests cancellation on all incomplete handles and collects confirmations
/// 2. Awaits all cancellation confirmations (draining complete)
/// 3. Awaits ALL handles to reach a terminal state (Complete, Cancelled, or Failed)
/// 4. Panics on hard failures (for now - unrecoverable error)
async fn cleanup_offloading_handles(mut handles: Vec<TransferHandle>, request_id: &str) {
    tracing::debug!(
        "starting cleanup of {} transfer handles for request_id: {}",
        handles.len(),
        request_id
    );

    // todo: add a configuration option to enable cancellation of incomplete handles on request finished

    // // 1. Request cancellation on all incomplete handles and collect confirmations
    // let mut confirmations = Vec::new();
    // for handle in &handles {
    //     if !handle.is_complete() {
    //         tracing::debug!(
    //             "requesting cancellation for handle {} (status: {:?})",
    //             handle.id(),
    //             handle.status()
    //         );
    //         confirmations.push(handle.cancel());
    //     }
    // }

    // // 2. Await all cancellation confirmations (draining complete)
    // // This ensures all queued items are swept and in-flight transfers finish
    // for confirmation in confirmations {
    //     confirmation.wait().await;
    // }

    // tracing::debug!(
    //     "all cancellation confirmations received for request_id: {}",
    //     request_id
    // );

    // 3. Await ALL handles to terminal state (Complete, Cancelled, or Failed)
    for handle in &mut handles {
        if handle.is_complete() {
            continue;
        }

        tracing::debug!(
            "awaiting handle {} to complete for request_id: {}",
            handle.id(),
            request_id
        );

        match handle.wait().await {
            Ok(result) => {
                if matches!(result.status, TransferStatus::Failed) {
                    // Hard failure - panic for now until we determine recovery policy
                    panic!(
                        "Hard failure on offload transfer {} for request_id: {}: {:?}",
                        result.id, request_id, result.error
                    );
                }
                tracing::debug!(
                    "handle {} completed with status {:?} for request_id: {}",
                    result.id,
                    result.status,
                    request_id
                );
            }
            Err(e) => {
                // Channel closed unexpectedly - this is also a hard failure
                panic!(
                    "Transfer handle channel closed unexpectedly for request_id: {}: {}",
                    request_id, e
                );
            }
        }
    }

    tracing::debug!(
        "cleanup complete for {} transfer handles for request_id: {}",
        handles.len(),
        request_id
    );
}

/// Async cleanup task for a request cancelled while in `PreparingToOnboard`.
///
/// Each shard's find session is driven to a terminal state before its
/// server-side state is released. `FindMatchesResult::Ready` variants resolve
/// immediately (blocks are held via RAII and will be dropped when the state
/// falls out of scope at the end of this task); `AsyncSession` variants await
/// `wait_for_completion` — already-terminal sessions complete on first poll,
/// and in-flight sessions drain naturally. A session-drain error is logged
/// but the task still attempts to release the session, because leaving the
/// leader-side session state mapped would be worse than a failed RPC.
///
/// Shards are drained concurrently via `join_all` so that ready / already-
/// terminal shards free their resources immediately without waiting on the
/// slowest in-flight shard.
///
/// No worker callback (`mark_onboarding_complete` / `mark_failed_onboarding` /
/// `mark_offloading_complete`) is emitted — the scheduler never committed
/// this request, so no worker ever learned about it.
async fn cleanup_preparing_to_onboard(
    onboarding_state: OnboardingState,
    instance_leader: InstanceLeader,
    request_id: String,
) {
    let shard_count = onboarding_state.shards.len();
    tracing::debug!(
        "starting PreparingToOnboard cleanup of {} shard(s) for request_id: {}",
        shard_count,
        request_id
    );

    let drain_futures = onboarding_state
        .shards
        .iter()
        .map(|shard| {
            let wait = shard.find_session.wait_for_completion();
            let session_id = shard.find_session.session_id();
            let instance_leader = instance_leader.clone();
            let request_id = request_id.clone();
            async move {
                if let Err(e) = wait.await {
                    tracing::warn!(
                        "find_session drain failed for request_id {} (session {:?}): {}; \
                         releasing session anyway",
                        request_id,
                        session_id,
                        e
                    );
                }
                if let Some(session_id) = session_id {
                    instance_leader.release_session(session_id);
                }
            }
        })
        .collect::<Vec<_>>();

    futures::future::join_all(drain_futures).await;

    // Dropping `onboarding_state` here releases any `Ready`-variant blocks
    // held by RAII.
    drop(onboarding_state);

    tracing::debug!(
        "PreparingToOnboard cleanup complete for request_id: {} ({} shard(s))",
        request_id,
        shard_count
    );
}

#[cfg(all(test, feature = "testing"))]
mod cleanup_tests {
    //! Unit tests for `cleanup_preparing_to_onboard`.
    //!
    //! These cover the observable contract of the cancel-path cleanup task:
    //! - every `AsyncSession` shard has its session released via
    //!   `InstanceLeader::release_session`,
    //! - `Ready` shards are dropped via RAII (no extra release call),
    //! - non-terminal `AsyncSession` shards block release until they become
    //!   terminal,
    //! - shards drain concurrently via `join_all`,
    //! - sessions unrelated to the state are left alone.
    //!
    //! Observability is via the `#[cfg(any(test, feature = "testing"))]`
    //! `has_session` accessor on `InstanceLeader` — `release_session` removes
    //! from all three internal session maps; `has_session` returns `true` if
    //! any of them still sees the id.
    use super::*;
    use kvbm_engine::leader::{
        AsyncSessionResult, FindMatchesResult, MatchBreakdown, OnboardingStatus, ReadyResult,
        SessionId,
    };
    use kvbm_engine::testing::{managers, messenger};
    use slot::{OnboardingShard, OnboardingState};
    use std::time::Duration;
    use tokio::sync::{Mutex as TokioMutex, watch};
    use uuid::Uuid;

    use crate::G2;

    const TEST_BLOCK_COUNT: usize = 8;
    const TEST_BLOCK_SIZE: usize = 4;

    /// Build a minimal `InstanceLeader` backed by a local TCP messenger. No
    /// handlers are registered — the tests only touch the session maps.
    async fn build_leader() -> InstanceLeader {
        let m = messenger::create_messenger_tcp()
            .await
            .expect("tcp messenger");
        let registry = managers::TestRegistryBuilder::new().build();
        let g2 = Arc::new(
            managers::TestManagerBuilder::<G2>::new()
                .block_count(TEST_BLOCK_COUNT)
                .block_size(TEST_BLOCK_SIZE)
                .registry(registry.clone())
                .build(),
        );
        InstanceLeader::builder()
            .messenger(m)
            .registry(registry)
            .g2_manager(g2)
            .workers(vec![])
            .build()
            .expect("instance leader")
    }

    /// A terminal `AsyncSession` (status = Complete) with a known session_id.
    fn complete_async(session_id: SessionId) -> FindMatchesResult {
        let (status_tx, status_rx) =
            watch::channel(OnboardingStatus::Complete { matched_blocks: 0 });
        drop(status_tx);
        FindMatchesResult::AsyncSession(AsyncSessionResult::new(
            session_id,
            status_rx,
            Arc::new(TokioMutex::new(Some(Vec::new()))),
            Arc::new(TokioMutex::new(MatchBreakdown::default())),
            None,
        ))
    }

    /// A pending `AsyncSession` (status = Searching) with a known session_id.
    /// Returns the shard and the watch sender so the test can drive it terminal.
    fn pending_async(
        session_id: SessionId,
    ) -> (FindMatchesResult, watch::Sender<OnboardingStatus>) {
        let (status_tx, status_rx) = watch::channel(OnboardingStatus::Searching);
        let session = AsyncSessionResult::new(
            session_id,
            status_rx,
            Arc::new(TokioMutex::new(None)),
            Arc::new(TokioMutex::new(MatchBreakdown::default())),
            None,
        );
        (FindMatchesResult::AsyncSession(session), status_tx)
    }

    /// A `Ready` shard with no blocks — no session_id, RAII-only release.
    fn ready_empty() -> FindMatchesResult {
        FindMatchesResult::Ready(ReadyResult::new(vec![], MatchBreakdown::default()))
    }

    /// Build an `OnboardingState` from a list of shards, numbering them
    /// contiguously starting at block 0 with `num_queried_blocks = 1` each.
    fn state_from(shards: Vec<FindMatchesResult>) -> OnboardingState {
        assert!(!shards.is_empty());
        let mut it = shards.into_iter().enumerate();
        let (i0, first) = it.next().unwrap();
        let mut state = OnboardingState::new(
            0,
            0,
            OnboardingShard {
                start_block: i0,
                num_queried_blocks: 1,
                find_session: first,
            },
        );
        for (i, find_session) in it {
            state.shards.push(OnboardingShard {
                start_block: i,
                num_queried_blocks: 1,
                find_session,
            });
        }
        state
    }

    fn fresh_id() -> SessionId {
        Uuid::new_v4()
    }

    /// Test 1 — a single terminal `AsyncSession` shard: cleanup must release
    /// the session.
    #[tokio::test]
    async fn cleanup_releases_async_session() {
        let leader = build_leader().await;
        let sid = fresh_id();
        leader.insert_test_session_marker(sid);
        assert!(leader.has_session(sid));

        let state = state_from(vec![complete_async(sid)]);
        cleanup_preparing_to_onboard(state, leader.clone(), "r1".into()).await;

        assert!(
            !leader.has_session(sid),
            "cleanup must release AsyncSession shards"
        );
    }

    /// Test 2 — mixed shards: `Ready` shard needs no session call; the
    /// `AsyncSession` shard is released; an unrelated session is untouched.
    #[tokio::test]
    async fn cleanup_skips_ready_shards_and_scoped_to_state() {
        let leader = build_leader().await;
        let sid = fresh_id();
        let unrelated = fresh_id();
        leader.insert_test_session_marker(sid);
        leader.insert_test_session_marker(unrelated);

        let state = state_from(vec![ready_empty(), complete_async(sid)]);
        cleanup_preparing_to_onboard(state, leader.clone(), "r2".into()).await;

        assert!(!leader.has_session(sid), "AsyncSession shard must release");
        assert!(
            leader.has_session(unrelated),
            "cleanup must not touch sessions outside the OnboardingState"
        );
    }

    /// Test 3 — a non-terminal `AsyncSession` shard blocks cleanup until the
    /// status transitions to a terminal state, then releases.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cleanup_drains_pending_async_before_release() {
        let leader = build_leader().await;
        let sid = fresh_id();
        leader.insert_test_session_marker(sid);

        let (find, status_tx) = pending_async(sid);
        let state = state_from(vec![find]);

        let leader_for_task = leader.clone();
        let task =
            tokio::spawn(cleanup_preparing_to_onboard(state, leader_for_task, "r3".into()));

        // With status still Searching, cleanup is blocked on wait_for_completion.
        // Give it a tick to ensure it has polled.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            leader.has_session(sid),
            "cleanup must wait for AsyncSession to reach terminal status before releasing"
        );
        assert!(!task.is_finished(), "cleanup task should still be pending");

        // Transition the session to terminal; cleanup should complete and release.
        status_tx
            .send(OnboardingStatus::Complete { matched_blocks: 0 })
            .expect("send terminal status");
        task.await.expect("cleanup task");

        assert!(
            !leader.has_session(sid),
            "cleanup must release after status reaches terminal"
        );
    }

    /// Test 4 — `join_all` drains shards concurrently: a terminal shard does
    /// not wait for a pending sibling.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cleanup_drains_concurrently() {
        let leader = build_leader().await;
        let terminal_id = fresh_id();
        let pending_id = fresh_id();
        leader.insert_test_session_marker(terminal_id);
        leader.insert_test_session_marker(pending_id);

        let (pending_find, status_tx) = pending_async(pending_id);
        let state = state_from(vec![complete_async(terminal_id), pending_find]);

        let leader_for_task = leader.clone();
        let task = tokio::spawn(cleanup_preparing_to_onboard(
            state,
            leader_for_task,
            "r4".into(),
        ));

        // Drive the test runtime so the terminal shard's release_session can
        // land even though the pending shard is still blocked.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !leader.has_session(terminal_id),
            "terminal shard must be released before the pending shard transitions"
        );
        assert!(
            leader.has_session(pending_id),
            "pending shard must still be registered while its find session is non-terminal"
        );
        assert!(!task.is_finished());

        status_tx
            .send(OnboardingStatus::Complete { matched_blocks: 0 })
            .expect("send terminal status");
        task.await.expect("cleanup task");

        assert!(!leader.has_session(pending_id));
    }
}
