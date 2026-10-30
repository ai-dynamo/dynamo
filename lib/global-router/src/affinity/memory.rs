// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dashmap::{DashMap, mapref::entry::Entry};

use super::{AffinityBinding, AffinityError, AffinityKey, AffinityStore, WriteOutcome};

#[derive(Debug, Default)]
pub struct InMemoryAffinityStore {
    bindings: DashMap<AffinityKey, AffinityBinding>,
}

impl InMemoryAffinityStore {
    /// Includes expired records until the next sweep.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Off-request-path cleanup, with Unix seconds supplied by the host.
    pub fn sweep(&self, now: u64) {
        self.bindings.retain(|_, binding| binding.expires_at > now);
    }
}

#[async_trait]
impl AffinityStore for InMemoryAffinityStore {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError> {
        Ok(self.bindings.get(key).map(|binding| binding.clone()))
    }

    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected_revision: Option<&str>,
        binding: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError> {
        match self.bindings.entry(key.clone()) {
            Entry::Occupied(mut entry)
                if expected_revision == Some(entry.get().revision.as_str()) =>
            {
                entry.insert(binding.clone());
                Ok(WriteOutcome::Applied)
            }
            Entry::Vacant(entry) if expected_revision.is_none() => {
                entry.insert(binding.clone());
                Ok(WriteOutcome::Applied)
            }
            _ => Ok(WriteOutcome::Conflict),
        }
    }
}
