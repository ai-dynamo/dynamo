// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::num::NonZeroUsize;

use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};

/// Number of physical prompt blocks represented by one tracked sequence hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ActiveSequenceStride(NonZeroUsize);

impl ActiveSequenceStride {
    /// Dense active-sequence tracking with one hash per complete prompt block.
    pub const ONE: Self = Self(NonZeroUsize::MIN);

    /// Create a validated active-sequence stride.
    pub const fn new(value: usize) -> Result<Self, ActiveSequenceStrideError> {
        match NonZeroUsize::new(value) {
            Some(value) => Ok(Self(value)),
            None => Err(ActiveSequenceStrideError::Zero),
        }
    }

    /// Return the stride as a primitive integer.
    pub const fn get(self) -> usize {
        self.0.get()
    }

    /// Encode the stride for replica events, omitting the legacy stride of one.
    pub const fn event_value(self) -> Option<usize> {
        if self.get() == 1 {
            None
        } else {
            Some(self.get())
        }
    }

    /// Return the number of retained hashes for a count of complete prompt blocks.
    pub const fn tracked_len(self, complete_blocks: usize) -> usize {
        complete_blocks / self.get()
    }

    /// Convert a complete dense rolling-hash chain into tracked active-sequence hashes.
    pub fn sample_dense(self, sequence: Vec<SequenceHash>) -> TrackedSequenceHashes {
        if self == Self::ONE {
            return TrackedSequenceHashes(sequence);
        }
        TrackedSequenceHashes(
            sequence
                .into_iter()
                .skip(self.get() - 1)
                .step_by(self.get())
                .collect(),
        )
    }

    /// Generate one independent tracked hash for each complete stride group.
    pub fn generate_for_complete_blocks(
        self,
        complete_blocks: usize,
        mut generate: impl FnMut() -> SequenceHash,
    ) -> TrackedSequenceHashes {
        TrackedSequenceHashes(
            (0..self.tracked_len(complete_blocks))
                .map(|_| generate())
                .collect(),
        )
    }

    /// Normalize a replica-event stride, treating a missing field as legacy stride one.
    pub(crate) const fn from_event_value(
        value: Option<usize>,
    ) -> Result<Self, ActiveSequenceStrideError> {
        match value {
            None => Ok(Self::ONE),
            Some(value) => Self::new(value),
        }
    }
}

impl Default for ActiveSequenceStride {
    fn default() -> Self {
        Self::ONE
    }
}

impl TryFrom<usize> for ActiveSequenceStride {
    type Error = ActiveSequenceStrideError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<NonZeroUsize> for ActiveSequenceStride {
    fn from(value: NonZeroUsize) -> Self {
        Self(value)
    }
}

impl fmt::Display for ActiveSequenceStride {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

/// Rolling hashes that have crossed the active-sequence stride boundary exactly once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackedSequenceHashes(Vec<SequenceHash>);

impl TrackedSequenceHashes {
    /// Borrow the tracked hashes as a slice.
    pub fn as_slice(&self) -> &[SequenceHash] {
        &self.0
    }

    /// Return the number of tracked hashes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return whether no prompt hashes are tracked.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Accept already-sampled hashes after their replica-event stride was validated.
    pub(crate) fn from_validated_event(sequence: Vec<SequenceHash>) -> Self {
        Self(sequence)
    }

    /// Consume the wrapper at the final sparsity-agnostic tracker boundary.
    pub(crate) fn into_inner(self) -> Vec<SequenceHash> {
        self.0
    }
}

impl AsRef<[SequenceHash]> for TrackedSequenceHashes {
    fn as_ref(&self) -> &[SequenceHash] {
        self.as_slice()
    }
}

#[cfg(test)]
pub(crate) fn tracked_sequence_hashes(sequence: Vec<SequenceHash>) -> TrackedSequenceHashes {
    ActiveSequenceStride::ONE.sample_dense(sequence)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum ActiveSequenceStrideError {
    #[error("active-sequence stride must be greater than zero")]
    Zero,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_rejects_zero_and_defaults_to_one() {
        assert_eq!(ActiveSequenceStride::default(), ActiveSequenceStride::ONE);
        assert_eq!(
            ActiveSequenceStride::new(0),
            Err(ActiveSequenceStrideError::Zero)
        );
    }

    #[test]
    fn sampling_retains_stride_endpoints_and_omits_trailing_group() {
        let stride = ActiveSequenceStride::new(3).unwrap();
        let tracked = stride.sample_dense((0..10).collect());
        assert_eq!(tracked.as_slice(), &[2, 5, 8]);
        assert_eq!(stride.tracked_len(10), 3);
    }

    #[test]
    fn stride_one_preserves_the_dense_chain() {
        let dense = vec![1, 2, 3];
        let allocation = dense.as_ptr();
        let tracked = ActiveSequenceStride::ONE.sample_dense(dense);
        assert_eq!(tracked.as_slice(), &[1, 2, 3]);
        assert_eq!(tracked.as_slice().as_ptr(), allocation);
        assert_eq!(ActiveSequenceStride::ONE.event_value(), None);
    }

    #[test]
    fn direct_generation_uses_the_tracked_length() {
        let stride = ActiveSequenceStride::new(3).unwrap();
        let mut next = 0;
        let tracked = stride.generate_for_complete_blocks(10, || {
            next += 1;
            next
        });
        assert_eq!(tracked.as_slice(), &[1, 2, 3]);
        assert_eq!(next, 3);
    }

    #[test]
    fn stride_serializes_as_an_integer() {
        let stride = ActiveSequenceStride::new(4).unwrap();
        assert_eq!(serde_json::to_string(&stride).unwrap(), "4");
        assert_eq!(
            serde_json::from_str::<ActiveSequenceStride>("4").unwrap(),
            stride
        );
        assert!(serde_json::from_str::<ActiveSequenceStride>("0").is_err());
    }
}
