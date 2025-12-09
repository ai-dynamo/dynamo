// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConnectorLeader {
    /// Mark a request as finished, returning the status.
    #[tracing::instrument(level = "debug", skip(self), fields(?request_id), ret)]
    pub fn request_finished(&self, request_id: &str) -> FinishedStatus {
        tracing::debug!("evaluating finished status");
        if let Some(shared_slot) = self.slots.get(request_id).map(|slot| slot.clone()) {
            match shared_slot.lock().slot_mark_finished() {
                FinishedStatus::Finished => {
                    self.remove_slot(request_id);
                    return FinishedStatus::Finished;
                }
                FinishedStatus::Pending => return FinishedStatus::Pending,
                FinishedStatus::UntrackedRequest => unreachable!(),
            }
        }
        FinishedStatus::UntrackedRequest
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
        let _offloading_state = slot.txn_take_offloading()?;
        todo!("clean up session and free resources");
    }
}
