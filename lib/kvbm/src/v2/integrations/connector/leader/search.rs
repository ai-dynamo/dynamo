// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConnectorLeader {
    /// Internal helper to check match status and determine the outcome.
    /// This function contains all the logic for determining whether blocks match,
    /// but does not perform state transitions - that's handled by the caller.
    pub(crate) fn process_match(
        &self,
        slot: &mut RequestSlot,
        num_computed_tokens: usize,
    ) -> Result<MatchCheckOutcome> {
        let block_size = slot.block_size();
        let total_tokens = slot.sequence.total_tokens();

        // Early exit if we cannot match a full block
        if (total_tokens - num_computed_tokens) < block_size {
            return Ok(MatchCheckOutcome::NoMatch);
        }

        let instance_leader = self
            .instance_leader
            .get()
            .ok_or_else(|| anyhow!("InstanceLeader not set; called before initialized"))?;

        // Check if we have an active find session
        if !slot.has_onboarding_state() {
            // Start a new find operation
            let sequence_hashes = slot.all_sequence_hashes();
            assert!(!sequence_hashes.is_empty());

            // Remove the already computed tokens from the search list
            assert!(num_computed_tokens.is_multiple_of(block_size));
            let num_device_blocks = num_computed_tokens / block_size;

            // If the total number of tokens is an even multiple of the block size,
            // then we do not include the last full block in the search.
            let last_block_index = if total_tokens.is_multiple_of(block_size) {
                (total_tokens / block_size) - 1
            } else {
                total_tokens / block_size
            };

            let search_sequence_hashes = &sequence_hashes[num_device_blocks..last_block_index];

            let options = FindMatchesOptions {
                search_remote: true,
                staging_mode: StagingMode::Full,
            };

            tracing::debug!(
                num_hashes = sequence_hashes.len(),
                "Starting find_matches_with_options"
            );

            match instance_leader.find_matches_with_options(search_sequence_hashes, options) {
                Ok(result) => {
                    if let Err(e) = slot.txn_prepare_to_onboard(num_computed_tokens, result) {
                        tracing::error!("Failed to set find session: {}", e);
                        bail!("Failed to set find session: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to start find operation: {}", e);
                    bail!("Failed to start find operation: {}", e);
                }
            }
        }

        debug_assert!(matches!(
            slot.txn_state(),
            TransactionState::PreparingToOnboard(_)
        ));

        // Check the status of the find session and produce outcome
        let session = slot.onboarding_state_mut().expect("session should exist");

        let outcome = match &session.find_session {
            FindMatchesResult::Ready(ready) => {
                // Ready result means immediate completion (local only, no async work)
                let matched_blocks = ready.g2_count();
                let matched_tokens = matched_blocks * block_size;

                tracing::debug!(
                    matched_blocks,
                    matched_tokens,
                    "Find completed immediately (Ready)"
                );

                MatchCheckOutcome::Found { matched_tokens }
            }
            FindMatchesResult::AsyncSession(async_session) => {
                // Check if the async session is complete
                let status = async_session.status();

                match status {
                    OnboardingStatus::Searching
                    | OnboardingStatus::Preparing { .. }
                    | OnboardingStatus::Staging { .. } => {
                        tracing::trace!(?status, "Find operation still in progress");
                        MatchCheckOutcome::InProgress
                    }
                    OnboardingStatus::Complete { matched_blocks } => {
                        // Completed - blocks are now in local G2
                        let matched_tokens = matched_blocks * block_size;

                        tracing::debug!(
                            matched_blocks,
                            matched_tokens,
                            "Find completed (Full mode)"
                        );

                        MatchCheckOutcome::Found { matched_tokens }
                    }
                    OnboardingStatus::Holding { .. } | OnboardingStatus::Prepared { .. } => {
                        unreachable!("Should not be in a holding or prepared state");
                    }
                }
            }
        };

        Ok(outcome)
    }

    /// Recover from a match error by transitioning to error state and extracting state for cleanup.
    pub(crate) fn recover_from_match_error(&self, slot: &mut RequestSlot) {
        // Transition to error state to preserve any active state data
        slot.txn_to_error();

        // Take the error state and clean up
        if let Ok(active_data) = slot.txn_take_error() {
            match active_data {
                slot::ActiveStateData::Onboarding(onboarding_state) => {
                    match onboarding_state.find_session {
                        FindMatchesResult::Ready(_) => {
                            // no-op - Ready sessions have no async cleanup
                        }
                        FindMatchesResult::AsyncSession(_async_session) => {
                            // todo: cancel and clean up the async session
                            tracing::warn!(
                                "Async session cleanup not yet implemented - session may leak"
                            );
                        }
                    }
                }
                slot::ActiveStateData::Offloading(_offloading_state) => {
                    // Offloading cleanup if needed
                    tracing::warn!("Offloading error recovery - cleanup may be needed");
                    todo!("implement offloading error recovery");
                }
            }
        }
        // Slot is now in Inactive state
    }
}
