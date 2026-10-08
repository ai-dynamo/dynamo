// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process prototype of a single scheduling-domain admission owner.
//!
//! The prototype serializes booking decisions for one domain and makes retries
//! idempotent by request ID. It deliberately does not define distributed owner
//! discovery, fencing, or recovery; those require the Phase 2 DEP decision.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

/// Result of forwarding an admission intent to a domain owner.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum DomainAdmissionError<E> {
    #[error("the scheduling-domain owner is unavailable")]
    OwnerUnavailable,

    #[error("the scheduling-domain owner could not book the request: {0}")]
    Booking(E),
}

/// Serializes one domain owner's booking decisions.
///
/// A successful decision is cached by stable request ID, so a retry after a
/// lost response returns the original booking without executing a second booking.
pub(crate) struct DomainOwner<Decision> {
    available: AtomicBool,
    decisions: Mutex<HashMap<String, Decision>>,
}

impl<Decision> DomainOwner<Decision> {
    pub(crate) fn new() -> Self {
        Self {
            available: AtomicBool::new(true),
            decisions: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn set_available(&self, available: bool) {
        self.available.store(available, Ordering::Release);
    }
}

impl<Decision: Clone> DomainOwner<Decision> {
    /// Return the cached booking for a retry or create one new booking decision.
    ///
    /// The owner holds its decision lock while booking so duplicate admissions
    /// cannot execute the booking closure concurrently. Failed bookings are not
    /// cached and may be retried once the caller resolves the underlying error.
    pub(crate) fn admit<E>(
        &self,
        request_id: &str,
        book: impl FnOnce() -> Result<Decision, E>,
    ) -> Result<Decision, DomainAdmissionError<E>> {
        if !self.available.load(Ordering::Acquire) {
            return Err(DomainAdmissionError::OwnerUnavailable);
        }

        let mut decisions = self.decisions.lock();
        if let Some(decision) = decisions.get(request_id) {
            return Ok(decision.clone());
        }

        let decision = book().map_err(DomainAdmissionError::Booking)?;
        decisions.insert(request_id.to_owned(), decision.clone());
        Ok(decision)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{DomainAdmissionError, DomainOwner};

    #[test]
    fn retry_returns_the_original_booking_without_rebooking() {
        let owner = DomainOwner::new();
        let booking_calls = Cell::new(0);

        let first = owner
            .admit("request-1", || {
                booking_calls.set(booking_calls.get() + 1);
                Ok::<_, ()>("worker-a")
            })
            .unwrap();
        let retry = owner
            .admit("request-1", || {
                booking_calls.set(booking_calls.get() + 1);
                Ok::<_, ()>("worker-b")
            })
            .unwrap();

        assert_eq!(first, "worker-a");
        assert_eq!(retry, "worker-a");
        assert_eq!(booking_calls.get(), 1);
    }

    #[test]
    fn unavailable_owner_fails_closed_without_booking() {
        let owner = DomainOwner::<&str>::new();
        owner.set_available(false);

        let result = owner.admit("request-1", || -> Result<&str, ()> {
            panic!("unavailable owner must not book")
        });

        assert_eq!(result, Err(DomainAdmissionError::OwnerUnavailable));
    }

    #[test]
    fn failed_booking_is_not_cached() {
        let owner = DomainOwner::new();

        let failed = owner.admit("request-1", || Err::<&str, _>("worker unavailable"));
        let retry = owner.admit("request-1", || Ok::<_, &str>("worker-a"));

        assert_eq!(
            failed,
            Err(DomainAdmissionError::Booking("worker unavailable"))
        );
        assert_eq!(retry, Ok("worker-a"));
    }
}
