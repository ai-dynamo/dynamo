// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The lock-free protocol primitives of the arena index, in one file that the loom models
//! in `lib/kv-router/arena-loom` compile against loom's types through `super::sync`.
//!
//! Every function here is production code. The `_with` variants take `const` switches that
//! exist only so a model can remove one fix and show that the model then fails; production
//! calls the plain wrappers, which pass every fix.
//!
//! - Version steps (spec B 5.2, W1 to W3 and W7) and reader validation (R1, R5, R7).
//! - The child-claim handshake against a step that hands off a child table (5.3).
//! - The release fence after a free-list pop (A1).
//! - The promotion order (bit, then tombstone) and the reader's double coverage read.
//! - The stepless append's `len` publication.
//! - The rank mailbox state machine of the lane pool (8.1).

use std::collections::VecDeque;

use super::sync::{AtomicU8, AtomicU32, AtomicU64, Mutex, Ordering, fence, spin_hint};

pub(crate) const FLAG_SEALED: u32 = 1;
pub(crate) const FLAG_DEAD: u32 = 1 << 1;
pub(crate) const FLAG_ANCHOR: u32 = 1 << 2;
pub(crate) const FLAG_POISONED: u32 = 1 << 3;

/// An empty slot in a packed-entry table.
pub(crate) const EMPTY: u64 = 0;
/// A tombstoned entry.
pub(crate) const TOMB: u64 = u64::MAX;

// ----------------------------------------------------------------------------
// Version steps
// ----------------------------------------------------------------------------

/// An open version step. Closing it (dropping it normally) makes the version even again.
/// Dropping it while the thread unwinds leaves the version odd and marks the run poisoned
/// (fix 4), unless `POISON_ON_UNWIND` is off, which only the negative loom model does.
pub(crate) struct Step<'a, const POISON_ON_UNWIND: bool> {
    version: &'a AtomicU64,
    flags: &'a AtomicU32,
    odd: u64,
}

/// The production step.
pub(crate) type VersionStep<'a> = Step<'a, true>;

impl<'a, const POISON_ON_UNWIND: bool> Step<'a, POISON_ON_UNWIND> {
    /// W1: makes the version odd, then fences so a reader that sees any write of the step
    /// also sees the odd version (fix 1). `FENCE = false` is the negative model.
    #[inline]
    pub(crate) fn open_with<const FENCE: bool>(
        version: &'a AtomicU64,
        flags: &'a AtomicU32,
    ) -> Self {
        let odd = version.fetch_add(1, Ordering::SeqCst).wrapping_add(1);
        if FENCE {
            fence(Ordering::Release);
        }
        Self {
            version,
            flags,
            odd,
        }
    }

    /// W1 for steps that replace or hand off the child table: after the odd step, waits
    /// until no lock-free claim is in flight. The `SeqCst` fence pairs with the claimer's
    /// in [`claim_enter`], so either the claimer sees the odd version or this sees it.
    #[inline]
    pub(crate) fn open_excluding_claims_with<const FENCE: bool>(
        version: &'a AtomicU64,
        flags: &'a AtomicU32,
        inflight: &AtomicU32,
    ) -> Self {
        let odd = version.fetch_add(1, Ordering::SeqCst).wrapping_add(1);
        if FENCE {
            fence(Ordering::SeqCst);
        }
        // Claimers never wait while counted, so this wait is bounded by their probes.
        while inflight.load(Ordering::Acquire) != 0 {
            spin_hint();
        }
        Self {
            version,
            flags,
            odd,
        }
    }

    fn finish(&self, unwinding: bool) {
        if unwinding && POISON_ON_UNWIND {
            // W7: never even on unwind. Readers fall back to the lock and stop there.
            self.flags.fetch_or(FLAG_POISONED, Ordering::Release);
            return;
        }
        self.version
            .store(self.odd.wrapping_add(1), Ordering::Release);
    }

    /// Ends the step as an unwind would. Models use this instead of a real panic.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn unwind(self) {
        self.finish(true);
        std::mem::forget(self);
    }
}

impl<'a> Step<'a, true> {
    #[inline]
    pub(crate) fn open(version: &'a AtomicU64, flags: &'a AtomicU32) -> Self {
        Self::open_with::<true>(version, flags)
    }

    #[inline]
    pub(crate) fn open_excluding_claims(
        version: &'a AtomicU64,
        flags: &'a AtomicU32,
        inflight: &AtomicU32,
    ) -> Self {
        Self::open_excluding_claims_with::<true>(version, flags, inflight)
    }
}

impl<const POISON_ON_UNWIND: bool> Drop for Step<'_, POISON_ON_UNWIND> {
    fn drop(&mut self) {
        self.finish(std::thread::panicking());
    }
}

/// R1: the version a read attempt starts from, or `None` while a step is open.
#[inline]
pub(crate) fn read_begin(version: &AtomicU64) -> Option<u64> {
    let v = version.load(Ordering::Acquire);
    (v & 1 == 0).then_some(v)
}

/// R5: whether nothing stepped since `begun`. Everything the attempt loaded before this
/// call is consistent when it returns true.
#[inline]
pub(crate) fn read_validate_with<const FENCE: bool>(version: &AtomicU64, begun: u64) -> bool {
    if FENCE {
        fence(Ordering::Acquire);
    }
    version.load(Ordering::Relaxed) == begun
}

#[inline]
pub(crate) fn read_validate(version: &AtomicU64, begun: u64) -> bool {
    read_validate_with::<true>(version, begun)
}

// ----------------------------------------------------------------------------
// Child claims
// ----------------------------------------------------------------------------

/// Claimer side of the handshake: counts the claim in `inflight`, then checks that the run
/// has not stepped since the snapshot the claimer planned from. On `false` the claim is not
/// counted and the caller re-plans or takes the locked path.
#[inline]
pub(crate) fn claim_enter_with<const FENCE: bool>(
    inflight: &AtomicU32,
    version: &AtomicU64,
    expected: u64,
) -> bool {
    inflight.fetch_add(1, Ordering::SeqCst);
    if FENCE {
        fence(Ordering::SeqCst);
    }
    if version.load(Ordering::Acquire) == expected {
        return true;
    }
    inflight.fetch_sub(1, Ordering::Release);
    false
}

#[inline]
pub(crate) fn claim_enter(inflight: &AtomicU32, version: &AtomicU64, expected: u64) -> bool {
    claim_enter_with::<true>(inflight, version, expected)
}

/// Ends a counted claim; its table writes happen before a writer's wait sees zero.
#[inline]
pub(crate) fn claim_exit(inflight: &AtomicU32) {
    inflight.fetch_sub(1, Ordering::Release);
}

// ----------------------------------------------------------------------------
// Free-list reuse
// ----------------------------------------------------------------------------

/// A1 (fix 2): issued after popping an address or id from a free list and before the first
/// write to it, so a stale reader that sees the new owner's words also sees the old owner's
/// kill step and fails validation.
#[inline]
pub(crate) fn reuse_fence_with<const FENCE: bool>() {
    if FENCE {
        fence(Ordering::Release);
    }
}

#[inline]
pub(crate) fn reuse_fence() {
    reuse_fence_with::<true>();
}

// ----------------------------------------------------------------------------
// Coverage promotion
// ----------------------------------------------------------------------------

/// W5 promotion from a partial entry to a whole bit: the bit first, then the tombstone,
/// released so a reader whose acquire load sees the tombstone also sees the bit.
#[inline]
pub(crate) fn promote_with<const RELEASE: bool>(whole: &AtomicU64, bit: u64, entry: &AtomicU64) {
    whole.fetch_or(bit, Ordering::Release);
    let order = if RELEASE {
        Ordering::Release
    } else {
        Ordering::Relaxed
    };
    entry.store(TOMB, order);
}

#[inline]
pub(crate) fn promote(whole: &AtomicU64, bit: u64, entry: &AtomicU64) {
    promote_with::<true>(whole, bit, entry);
}

/// Loads a cutoff entry for the reader's coverage read (R4).
#[inline]
pub(crate) fn load_entry_with<const ACQUIRE: bool>(entry: &AtomicU64) -> u64 {
    entry.load(if ACQUIRE {
        Ordering::Acquire
    } else {
        Ordering::Relaxed
    })
}

#[inline]
pub(crate) fn load_entry(entry: &AtomicU64) -> u64 {
    load_entry_with::<true>(entry)
}

// ----------------------------------------------------------------------------
// Stepless append
// ----------------------------------------------------------------------------

/// W5 append: the caller has written the new positions; this publishes them.
#[inline]
pub(crate) fn publish_len_with<const RELEASE: bool>(len: &AtomicU32, new_len: u32) {
    len.store(
        new_len,
        if RELEASE {
            Ordering::Release
        } else {
            Ordering::Relaxed
        },
    );
}

#[inline]
pub(crate) fn publish_len(len: &AtomicU32, new_len: u32) {
    publish_len_with::<true>(len, new_len);
}

/// R3: `len` is the one window field loaded with `Acquire`, for the stepless append.
#[inline]
pub(crate) fn load_len_with<const ACQUIRE: bool>(len: &AtomicU32) -> u32 {
    len.load(if ACQUIRE {
        Ordering::Acquire
    } else {
        Ordering::Relaxed
    })
}

#[inline]
pub(crate) fn load_len(len: &AtomicU32) -> u32 {
    load_len_with::<true>(len)
}

// ----------------------------------------------------------------------------
// Rank mailboxes
// ----------------------------------------------------------------------------

pub(crate) const IDLE: u8 = 0;
pub(crate) const READY: u8 = 1;
pub(crate) const RUNNING: u8 = 2;

/// A rank's queued tasks and who may run them (spec B 8.1). A cell is on at most one ready
/// list exactly while it is `READY`, and only the lane that moved it to `RUNNING` applies
/// its tasks, so a rank's tasks run one at a time in queue order.
pub(crate) struct Mailbox<T> {
    state: AtomicU8,
    queue: Mutex<VecDeque<T>>,
}

impl<T> Default for Mailbox<T> {
    fn default() -> Self {
        Self {
            state: AtomicU8::new(IDLE),
            queue: Mutex::new(VecDeque::new()),
        }
    }
}

impl<T> Mailbox<T> {
    /// Producer: queues `task`. Returns true when the caller made the cell `READY` and must
    /// put it on a ready list.
    pub(crate) fn push(&self, task: T) -> bool {
        self.queue.lock().push_back(task);
        self.state
            .compare_exchange(IDLE, READY, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    /// Inline fast path: claims the cell for a task the caller applies without queueing it.
    /// Requires an empty queue, checked under the queue lock: `IDLE` alone does not imply
    /// an empty queue, because a producer queues before it marks the cell ready.
    pub(crate) fn try_inline(&self) -> bool {
        let queue = self.queue.lock();
        queue.is_empty()
            && self
                .state
                .compare_exchange(IDLE, RUNNING, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
    }

    /// Taker: `READY` to `RUNNING`. False is a scheduler-invariant violation (fix 8).
    pub(crate) fn take(&self) -> bool {
        self.state
            .compare_exchange(READY, RUNNING, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Owner: the next queued task.
    pub(crate) fn pop(&self) -> Option<T> {
        self.queue.lock().pop_front()
    }

    /// Owner: gives the cell up. Returns true when tasks arrived meanwhile and the caller
    /// made it `READY` again and must put it on a ready list. The re-check under the queue
    /// lock is what keeps a push that saw `RUNNING` from being stranded (`RECHECK = false`
    /// is the negative model).
    pub(crate) fn release_with<const RECHECK: bool>(&self) -> bool {
        self.state.store(IDLE, Ordering::Release);
        if !RECHECK {
            return false;
        }
        let queue = self.queue.lock();
        !queue.is_empty()
            && self
                .state
                .compare_exchange(IDLE, READY, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
    }

    pub(crate) fn release(&self) -> bool {
        self.release_with::<true>()
    }
}
