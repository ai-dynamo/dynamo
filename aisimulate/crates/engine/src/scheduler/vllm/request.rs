// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use uuid::Uuid;

use crate::common::sequence::RequestSequence;
use crate::kv_manager::BlockRequestLease;

/// Request-local token state and its native-G1 physical lease.
pub(crate) struct RequestKvState {
    pub(super) sequence: RequestSequence,
    pub(super) lease: BlockRequestLease,
}

impl RequestKvState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn native(
        owner: Uuid,
        tokens: Vec<u32>,
        max_output_tokens: usize,
        output_capacity_hint: usize,
        block_size: usize,
        enable_prefix_caching: bool,
        retain_local_hashes: bool,
        emit_token_ids: bool,
        planned_output_ids: Option<Vec<u32>>,
    ) -> Self {
        let (sequence, identities) = RequestSequence::new(
            tokens,
            max_output_tokens,
            output_capacity_hint,
            block_size,
            enable_prefix_caching,
            retain_local_hashes,
            emit_token_ids,
            planned_output_ids,
        );
        Self {
            sequence,
            lease: BlockRequestLease::new(owner, identities),
        }
    }

    #[cfg(test)]
    pub(super) fn native_parts(&self) -> (&RequestSequence, &BlockRequestLease) {
        (&self.sequence, &self.lease)
    }

    pub(super) fn len(&self) -> usize {
        self.sequence.len()
    }

    pub(super) fn max_output_tokens(&self) -> usize {
        self.sequence.max_output_tokens()
    }

    pub(super) fn generated_tokens(&self) -> usize {
        self.sequence.generated_tokens()
    }

    pub(super) fn num_input_tokens(&self) -> usize {
        self.sequence.num_input_tokens()
    }

    pub(super) fn num_allocated_tokens(&self) -> usize {
        self.lease.allocated_tokens()
    }

    pub(super) fn current_known_blocks(&self) -> usize {
        self.sequence.current_known_blocks()
    }

    pub(super) fn to_completion_blocks(&self) -> usize {
        self.sequence.to_completion_blocks()
    }

    pub(super) fn generate_token(&mut self) -> u32 {
        let (token, opened_partial) = self.sequence.generate_token();
        if opened_partial {
            self.lease.append_partial();
        }
        token
    }

    #[cfg(test)]
    pub(super) const fn uses_flat_tokens(&self) -> bool {
        true
    }

    #[cfg(test)]
    pub(super) fn native_storage_capacities(&self) -> (usize, usize) {
        (self.sequence.token_capacity(), self.lease.entry_capacity())
    }
}
