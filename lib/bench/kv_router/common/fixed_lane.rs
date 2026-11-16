// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed single-producer, single-consumer ID lane shared by the open-loop
//! KV-router benchmarks.
//!
//! The producer writes operation IDs into preallocated slots with `Relaxed`
//! stores, then makes them visible by storing the new count into `published`
//! with `Release`. The consumer reads `published` with `Acquire` before it
//! reads any slot below it. Each benchmark keeps its own payload loop and its
//! own publication cadence; this module only owns the slot protocol and the
//! close-aware wait.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use tokio::sync::Notify;

/// Slot value before the producer writes an ID. Operation IDs must stay below it.
pub const EMPTY_OPERATION_ID: u32 = u32::MAX;

/// The producer tried to write past the lane's fixed capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneOverflow;

/// Fixed-capacity ID lane with optional consumer-progress reporting.
pub struct FixedLane {
    slots: Box<[AtomicU32]>,
    published: AtomicUsize,
    consumed: AtomicUsize,
    closed: AtomicBool,
    notify: Notify,
}

impl FixedLane {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| AtomicU32::new(EMPTY_OPERATION_ID))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            published: AtomicUsize::new(0),
            consumed: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Write one ID at `cursor`. The ID is not visible until `publish`.
    #[inline]
    pub fn write(&self, cursor: usize, id: u32) -> Result<(), LaneOverflow> {
        let Some(slot) = self.slots.get(cursor) else {
            return Err(LaneOverflow);
        };
        slot.store(id, Ordering::Relaxed);
        Ok(())
    }

    /// Publish every slot below `count` and wake the consumer.
    #[inline]
    pub fn publish(&self, count: usize) {
        self.published.store(count, Ordering::Release);
        self.notify.notify_one();
    }

    #[inline]
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_one();
    }

    #[inline]
    pub fn published(&self) -> usize {
        self.published.load(Ordering::Acquire)
    }

    /// Read a slot. Callers must only read indices below a prior `published()` load.
    #[inline]
    pub fn id(&self, index: usize) -> u32 {
        self.slots[index].load(Ordering::Relaxed)
    }

    /// Wait until more IDs may be published after `consumed`.
    ///
    /// Returns `false` once the lane is closed and fully drained. Returns `true`
    /// after a wakeup or when new IDs are already visible; the caller then
    /// reloads `published()`. The `notified()` future is created before the
    /// recheck so a publish or close that lands in between is not lost.
    pub async fn wait_for_more(&self, consumed: usize) -> bool {
        if self.closed.load(Ordering::Acquire) && consumed == self.published.load(Ordering::Acquire)
        {
            return false;
        }

        let notified = self.notify.notified();
        if consumed == self.published.load(Ordering::Acquire)
            && !self.closed.load(Ordering::Acquire)
        {
            notified.await;
        }
        true
    }

    #[inline]
    pub fn record_consumed(&self, consumed: usize) {
        self.consumed.store(consumed, Ordering::Release);
    }

    #[inline]
    pub fn consumed(&self) -> usize {
        self.consumed.load(Ordering::Acquire)
    }

    pub fn depth(&self) -> usize {
        self.published().saturating_sub(self.consumed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::{EMPTY_OPERATION_ID, FixedLane, LaneOverflow};

    const WAKE_TIMEOUT: Duration = Duration::from_secs(5);

    #[test]
    fn write_past_capacity_overflows() {
        let lane = FixedLane::new(1);
        assert_eq!(lane.capacity(), 1);
        assert_eq!(lane.id(0), EMPTY_OPERATION_ID);
        assert_eq!(lane.write(0, 1), Ok(()));
        assert_eq!(lane.write(1, 2), Err(LaneOverflow));
        assert_eq!(lane.id(0), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn publish_wakes_a_waiting_consumer() {
        let lane = Arc::new(FixedLane::new(1));
        let waiter = tokio::spawn({
            let lane = Arc::clone(&lane);
            async move { lane.wait_for_more(0).await }
        });
        // Let the waiter park on `notified()` before publishing.
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());

        lane.write(0, 7).unwrap();
        lane.publish(1);
        let woke = tokio::time::timeout(WAKE_TIMEOUT, waiter)
            .await
            .expect("publish must wake the consumer")
            .unwrap();
        assert!(woke);
        assert_eq!(lane.published(), 1);
        assert_eq!(lane.id(0), 7);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_wakes_a_waiting_consumer_then_reports_drained() {
        let lane = Arc::new(FixedLane::new(1));
        let waiter = tokio::spawn({
            let lane = Arc::clone(&lane);
            async move { lane.wait_for_more(0).await }
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());

        lane.close();
        tokio::time::timeout(WAKE_TIMEOUT, waiter)
            .await
            .expect("close must wake the consumer")
            .unwrap();
        assert!(!lane.wait_for_more(0).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn publish_and_close_before_wait_is_not_lost() {
        let lane = FixedLane::new(1);
        lane.write(0, 7).unwrap();
        lane.publish(1);
        lane.close();

        // Not drained yet: must return without parking.
        let pending = tokio::time::timeout(WAKE_TIMEOUT, lane.wait_for_more(0))
            .await
            .expect("a published ID must not block the consumer");
        assert!(pending);
        // Drained and closed: must report the end of the lane.
        let drained = tokio::time::timeout(WAKE_TIMEOUT, lane.wait_for_more(1))
            .await
            .expect("a drained, closed lane must not block the consumer");
        assert!(!drained);
    }

    #[test]
    fn fixed_lane_reports_depth() {
        let lane = FixedLane::new(3);
        assert_eq!(lane.capacity(), 3);
        assert_eq!(lane.depth(), 0);
        lane.write(0, 0).unwrap();
        lane.write(1, 1).unwrap();
        lane.publish(2);
        assert_eq!(lane.depth(), 2);
        lane.record_consumed(1);
        assert_eq!(lane.consumed(), 1);
        assert_eq!(lane.depth(), 1);
        lane.record_consumed(2);
        assert_eq!(lane.depth(), 0);
    }
}
