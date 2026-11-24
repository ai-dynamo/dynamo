// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request ids inside a worker's engine call.
//!
//! A migration retry keeps its request's id. When the retry reaches a worker
//! that is still inside the engine call for the first attempt, the worker
//! refuses it with [`PipelineError::DuplicateRequestId`], which the frontend
//! reads as a `Conflict`: it retries on another worker and does not mark this
//! one down. An engine never sees two calls with one id at once.
//!
//! An id is held only until the engine returns its stream, not for the whole
//! stream: the stream is the call's return value, so two streams with one id
//! cannot cross.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::pipeline::PipelineError;

/// The set of ids whose engine call has not returned yet.
#[derive(Default)]
pub(crate) struct InFlightIds {
    ids: Mutex<HashSet<Arc<str>>>,
}

impl InFlightIds {
    /// Hold `id` until the returned claim is dropped.
    ///
    /// # Errors
    ///
    /// [`PipelineError::DuplicateRequestId`] if `id` is already held.
    pub(crate) fn claim(&self, id: &str) -> Result<InFlightClaim<'_>, PipelineError> {
        let mut ids = self.ids.lock();
        if ids.contains(id) {
            return Err(PipelineError::DuplicateRequestId(id.to_string()));
        }
        // One allocation, shared by the set and the claim that removes it.
        let id: Arc<str> = Arc::from(id);
        ids.insert(id.clone());
        Ok(InFlightClaim { owner: self, id })
    }
}

/// Frees its id when dropped: after the engine call returns or fails, or when
/// the request is dropped inside the call.
pub(crate) struct InFlightClaim<'a> {
    owner: &'a InFlightIds,
    id: Arc<str>,
}

impl Drop for InFlightClaim<'_> {
    fn drop(&mut self) {
        self.owner.ids.lock().remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_held_id_is_refused_until_its_claim_is_dropped() {
        let ids = InFlightIds::default();
        let claim = ids.claim("a").unwrap();
        assert!(matches!(
            ids.claim("a"),
            Err(PipelineError::DuplicateRequestId(id)) if id == "a"
        ));
        let other = ids.claim("b").expect("other ids are independent");
        drop(claim);
        drop(ids.claim("a").expect("a dropped claim frees its id"));
        drop(other);
        assert!(ids.ids.lock().is_empty());
    }
}
