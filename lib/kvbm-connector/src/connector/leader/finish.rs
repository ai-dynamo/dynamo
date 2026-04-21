// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_engine::offload::{TransferHandle, TransferStatus};

use super::*;
use super::slot::TransactionState;

impl ConnectorLeader {
    /// Mark a request as finished, returning the status.
    ///
    /// This method:
    /// 1. Marks the slot for deletion
    /// 2. If there's an OffloadingState, takes it and checks all transfer handles
    /// 3. If all handles are complete, returns Finished immediately
    /// 4. If any handles are incomplete, spawns a cleanup task and returns Pending
    #[tracing::instrument(level = "debug", skip(self), fields(?request_id), ret)]
    pub fn request_finished(&self, request_id: &str) -> FinishedStatus {
        tracing::debug!("evaluating finished status");

        let Some(shared_slot) = self.slots.get(request_id).map(|slot| slot.clone()) else {
            return FinishedStatus::UntrackedRequest;
        };

        let mut slot = shared_slot.lock();

        // Mark the slot for deletion
        let initial_status = slot.slot_mark_finished();

        // If immediately finished (no active transaction), we're done
        if matches!(initial_status, FinishedStatus::Finished) {
            drop(slot);
            self.remove_slot(request_id);
            return FinishedStatus::Finished;
        }

        // vLLM may call `request_finished` while we are still searching or
        // loading from external KV cache (client timeout/disconnect, scheduler
        // preemption, eviction restore). Without this guard the code below
        // falls through to `txn_take_offloading`, which is only valid from
        // `Offloading` and returns `InvalidTransition`. We then return
        // `initial_status` (`Pending`) without cancelling the find session or
        // releasing its `session_id`, leaving the slot pinned in
        // `{PreparingToOnboard, Onboarding} + MarkedForDeletion` forever —
        // which in turn keeps G2/G3 blocks pinned so `/reset/g2` reports
        // `BlockCountMismatch` indefinitely (observed
        // 2026-04-21_15-00-13: 8 leaked slots → 60 s client timeout on
        // `clear` → SLURM SIGTERM).
        //
        // Cancel the in-flight onboarding, release the session, drop the
        // slot, and report `Finished`. Mirrors the cleanup in
        // `onboard.rs::execute_onboarding` after a successful load.
        if matches!(
            slot.txn_state(),
            TransactionState::PreparingToOnboard(_) | TransactionState::Onboarding(_)
        ) {
            match slot.txn_cancel_onboarding() {
                Ok(onboarding_state) => {
                    let session_id = onboarding_state.find_session.session_id();
                    // Defense in depth: drain any queued intra-pass load data
                    // so we don't leave stale G2/G1 block-id pairs on a slot
                    // that is about to be dropped. In practice this is always
                    // `None` here — `process_scheduler_output` drains it at
                    // the end of the same step that sets it — but consuming
                    // it explicitly keeps cleanup symmetrical with the happy
                    // path and guards against future intra-pass schedulers
                    // that might queue data across multiple steps.
                    let _ = slot.take_pending_intra_pass();
                    drop(slot);
                    if let Some(session_id) = session_id {
                        if let Some(leader) = self.instance_leader.get() {
                            leader.release_session(session_id);
                        }
                    }
                    self.remove_slot(request_id);
                    tracing::debug!(
                        "cancelled in-flight onboarding on request_finished for {}",
                        request_id
                    );
                    return FinishedStatus::Finished;
                }
                Err(e) => {
                    // Unreachable: state matched above under the slot lock.
                    // Fall through to the existing error path rather than
                    // panicking mid-teardown.
                    tracing::error!(
                        "Unexpected failure cancelling onboarding for {}: {}",
                        request_id,
                        e
                    );
                }
            }
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
