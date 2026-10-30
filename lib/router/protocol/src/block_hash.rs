// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block hash identities. The functions that compute them live with the hashing contract, not here.

use serde::{Deserialize, Serialize};

/// A [`LocalBlockHash`] is a hash computed from the token IDs, optional multimodal metadata,
/// and optional LoRA adapter name of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct LocalBlockHash(pub u64);

/// A sequence-aware hash of a block computed by the engine from token IDs, optional metadata,
/// and the hash of the parent block.
///
/// In this case, the hashing function is external and unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExternalSequenceBlockHash(pub u64);

// Implement From trait for convenient conversion
impl From<u64> for ExternalSequenceBlockHash {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<i64> for ExternalSequenceBlockHash {
    /// Bitwise reinterpretation: preserves all bits, including negatives.
    /// This is lossless, but negative i64 values will appear as large u64 values.
    fn from(value: i64) -> Self {
        Self(value as u64)
    }
}

impl Serialize for LocalBlockHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for LocalBlockHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(LocalBlockHash(value))
    }
}

impl Serialize for ExternalSequenceBlockHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for ExternalSequenceBlockHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(ExternalSequenceBlockHash(value))
    }
}
