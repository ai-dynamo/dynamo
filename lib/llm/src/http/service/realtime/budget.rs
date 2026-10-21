// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(in crate::http::service) struct Budget {
    limit: usize,
    used: AtomicUsize,
}

impl Budget {
    pub(in crate::http::service) fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
        }
    }

    pub(super) fn reservation(self: &Arc<Self>) -> Reservation {
        Reservation {
            budget: self.clone(),
            bytes: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }
}

pub(super) struct Reservation {
    budget: Arc<Budget>,
    pub(super) bytes: usize,
}

impl Reservation {
    pub(super) fn try_grow(&mut self, bytes: usize) -> bool {
        if bytes == 0 {
            return true;
        }
        let reserved =
            self.budget
                .used
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                    used.checked_add(bytes)
                        .filter(|&sum| sum <= self.budget.limit)
                });
        if let Err(used) = reserved {
            tracing::debug!(
                used,
                limit = self.budget.limit,
                requested = bytes,
                "Realtime journal budget exhausted"
            );
            return false;
        }
        self.bytes += bytes;
        true
    }

    pub(super) fn release(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.bytes -= bytes;
        self.budget.used.fetch_sub(bytes, Ordering::Relaxed);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn concurrent_reservations_cannot_overcommit_and_drop_releases_them() {
        let budget = Arc::new(Budget::new(100));
        let barrier = Barrier::new(16);
        std::thread::scope(|scope| {
            for _ in 0..16 {
                scope.spawn(|| {
                    let mut reservation = budget.reservation();
                    barrier.wait();
                    reservation.try_grow(25);
                    barrier.wait();
                    let used = budget.used();
                    barrier.wait();
                    assert_eq!(used, 100);
                });
            }
        });
        assert_eq!(budget.used(), 0);
        let mut reservation = budget.reservation();
        assert!(!reservation.try_grow(usize::MAX));
        assert!(reservation.try_grow(100));
        reservation.release(25);
        assert_eq!(budget.used(), 75);
    }
}
