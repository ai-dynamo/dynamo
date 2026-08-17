// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "kvbm-offload")]
use dynamo_tokens::blocks::UniqueBlock;
#[cfg(feature = "kvbm-offload")]
use dynamo_tokens::{BlockHash, PositionalLineageHash};
use uuid::Uuid;

use crate::common::protocols::MoveBlock;
use crate::common::sequence::{ActiveSequence, RequestSequence};
use crate::kv_manager::BlockRequestLease;

/// Backend-specific request KV state selected once when the request enters the
/// shared vLLM/TRT scheduler.
pub(crate) enum RequestKvState {
    Native {
        sequence: RequestSequence,
        lease: BlockRequestLease,
    },
    Kvbm(ActiveSequence),
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
        Self::Native {
            sequence,
            lease: BlockRequestLease::new(owner, identities),
        }
    }

    pub(super) fn kvbm(sequence: ActiveSequence) -> Self {
        Self::Kvbm(sequence)
    }

    #[cfg(feature = "kvbm-offload")]
    pub(super) fn legacy(&self) -> Option<&ActiveSequence> {
        match self {
            Self::Native { .. } => None,
            Self::Kvbm(sequence) => Some(sequence),
        }
    }

    #[cfg(test)]
    pub(super) fn native_parts(&self) -> Option<(&RequestSequence, &BlockRequestLease)> {
        match self {
            Self::Native { sequence, lease } => Some((sequence, lease)),
            Self::Kvbm(_) => None,
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Native { sequence, .. } => sequence.len(),
            Self::Kvbm(sequence) => sequence.len(),
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(super) fn block_size(&self) -> usize {
        match self {
            Self::Native { sequence, .. } => sequence.block_size(),
            Self::Kvbm(sequence) => sequence.block_size(),
        }
    }

    pub(super) fn max_output_tokens(&self) -> usize {
        match self {
            Self::Native { sequence, .. } => sequence.max_output_tokens(),
            Self::Kvbm(sequence) => sequence.max_output_tokens(),
        }
    }

    pub(super) fn generated_tokens(&self) -> usize {
        match self {
            Self::Native { sequence, .. } => sequence.generated_tokens(),
            Self::Kvbm(sequence) => sequence.generated_tokens(),
        }
    }

    pub(super) fn num_input_tokens(&self) -> usize {
        match self {
            Self::Native { sequence, .. } => sequence.num_input_tokens(),
            Self::Kvbm(sequence) => sequence.num_input_tokens(),
        }
    }

    pub(super) fn num_allocated_tokens(&self) -> usize {
        match self {
            Self::Native { lease, .. } => lease.allocated_tokens(),
            Self::Kvbm(sequence) => sequence.num_allocated_tokens(),
        }
    }

    pub(super) fn current_known_blocks(&self) -> usize {
        match self {
            Self::Native { sequence, .. } => sequence.current_known_blocks(),
            Self::Kvbm(sequence) => sequence.current_known_blocks(),
        }
    }

    pub(super) fn to_completion_blocks(&self) -> usize {
        match self {
            Self::Native { sequence, .. } => sequence.to_completion_blocks(),
            Self::Kvbm(sequence) => sequence.to_completion_blocks(),
        }
    }

    pub(super) fn generate_token(&mut self) -> (u32, Vec<MoveBlock>) {
        match self {
            Self::Native { sequence, lease } => {
                let (token, opened_partial) = sequence.generate_token();
                if opened_partial {
                    lease.append_partial();
                }
                (token, Vec::new())
            }
            Self::Kvbm(sequence) => sequence.generate_token(),
        }
    }

    pub(super) fn pop(&mut self) {
        match self {
            Self::Native { .. } => {
                unreachable!("native decode never rolls back a sampled token")
            }
            Self::Kvbm(sequence) => sequence.pop(),
        }
    }

    pub(super) fn terminal_signals(&self) -> Vec<MoveBlock> {
        match self {
            Self::Native { .. } => Vec::new(),
            Self::Kvbm(sequence) => sequence.terminal_signals(),
        }
    }

    pub(super) fn free_signal(&self) -> Vec<MoveBlock> {
        match self {
            Self::Native { .. } => Vec::new(),
            Self::Kvbm(sequence) => sequence.free_signal(),
        }
    }

    pub(super) fn reset_legacy_with_signal(&mut self) -> Vec<MoveBlock> {
        match self {
            Self::Native { .. } => Vec::new(),
            Self::Kvbm(sequence) => sequence.reset_with_signal(),
        }
    }

    #[cfg(test)]
    pub(super) fn uses_flat_tokens(&self) -> bool {
        match self {
            Self::Native { .. } => true,
            Self::Kvbm(_) => false,
        }
    }

    #[cfg(test)]
    pub(super) fn native_storage_capacities(&self) -> Option<(usize, usize)> {
        match self {
            Self::Native { sequence, lease } => {
                Some((sequence.token_capacity(), lease.entry_capacity()))
            }
            Self::Kvbm(_) => None,
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(super) fn unique_blocks(&self) -> &[UniqueBlock] {
        self.legacy()
            .expect("KVBM metadata requested from a native request")
            .unique_blocks()
    }

    #[cfg(feature = "kvbm-offload")]
    pub(super) fn positional_lineage_hashes(&self) -> &[PositionalLineageHash] {
        self.legacy()
            .expect("KVBM metadata requested from a native request")
            .positional_lineage_hashes()
    }

    #[cfg(feature = "kvbm-offload")]
    pub(super) fn block_hashes(&self) -> &[BlockHash] {
        self.legacy()
            .expect("KVBM metadata requested from a native request")
            .block_hashes()
    }

    #[cfg(feature = "kvbm-offload")]
    pub(super) fn block_token_ids(&self) -> Vec<Vec<u32>> {
        self.legacy()
            .expect("KVBM metadata requested from a native request")
            .block_token_ids()
    }
}
