// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::distributed::offload::{TransferHandle, TransferStatus};

use super::*;

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

        let mut slot = shared_slot.lock();
        let offloading_state = slot.txn_take_offloading()?;

        // Verify all handles are complete
        for handle in &offloading_state.handles {
            if !handle.is_complete() {
                tracing::warn!(
                    "process_finished_offloading called but handle {} is not complete for request_id: {}",
                    handle.id(),
                    request_id
                );
            }
        }

        tracing::debug!(
            "finished offloading {} blocks for request_id: {}",
            offloading_state.block_mappings.len(),
            request_id
        );

        Ok(())
    }
}

/// Async cleanup task for offloading handles.
///
/// This function:
/// 1. Requests cancellation on all handles (only affects queued transfers, not inflight)
/// 2. Awaits ALL handles to reach a terminal state (Complete, Cancelled, or Failed)
/// 3. Panics on hard failures (for now - unrecoverable error)
async fn cleanup_offloading_handles(mut handles: Vec<TransferHandle>, request_id: &str) {
    tracing::debug!(
        "starting cleanup of {} transfer handles for request_id: {}",
        handles.len(),
        request_id
    );

    // 1. Request cancellation on all handles (only affects queued, not inflight)
    for handle in &handles {
        if !handle.is_complete() {
            tracing::debug!(
                "requesting cancellation for handle {} (status: {:?})",
                handle.id(),
                handle.status()
            );
            // Cancel returns a future for confirmation, but we don't await individual cancellations
            // We'll await all handles below
            // let _ = handle.cancel();
        }
    }

    // 2. Await ALL handles to terminal state (Complete, Cancelled, or Failed)
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
