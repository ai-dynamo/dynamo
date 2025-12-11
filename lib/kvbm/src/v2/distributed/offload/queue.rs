// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellable queue implementation using crossbeam SegQueue.
//!
//! Provides a lock-free queue wrapper that supports active cancellation via
//! a sweeper task that can iterate through queued items and remove those
//! belonging to cancelled transfers.

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_queue::SegQueue;
use dashmap::DashSet;

use super::handle::TransferId;

/// A queued item with its associated transfer ID.
pub struct QueueItem<T> {
    /// The transfer this item belongs to
    pub transfer_id: TransferId,
    /// The actual data
    pub data: T,
}

impl<T> QueueItem<T> {
    /// Create a new queue item.
    pub fn new(transfer_id: TransferId, data: T) -> Self {
        Self { transfer_id, data }
    }
}

/// A lock-free queue that supports active cancellation via sweeping.
///
/// Unlike mpsc channels where cancellation can only be checked at dequeue time,
/// this queue allows a dedicated sweeper task to iterate through queued items
/// and remove those belonging to cancelled transfers. This ensures that
/// `ImmutableBlock` guards are dropped promptly when a transfer is cancelled.
///
/// # Architecture
///
/// ```text
/// Producer ──► [SegQueue] ◄── Consumer
///                  ▲
///                  │
///             [Sweeper Task]
///                  │
///            (removes cancelled items)
/// ```
pub struct CancellableQueue<T> {
    /// The underlying lock-free queue
    inner: SegQueue<QueueItem<T>>,
    /// Set of cancelled transfer IDs
    cancelled: DashSet<TransferId>,
    /// Approximate length for monitoring (not exact due to concurrent access)
    len: AtomicUsize,
}

impl<T> CancellableQueue<T> {
    /// Create a new cancellable queue.
    pub fn new() -> Self {
        Self {
            inner: SegQueue::new(),
            cancelled: DashSet::new(),
            len: AtomicUsize::new(0),
        }
    }

    /// Push an item onto the queue.
    ///
    /// If the transfer has already been cancelled, the item is dropped immediately.
    /// Returns `true` if the item was queued, `false` if it was dropped due to cancellation.
    pub fn push(&self, transfer_id: TransferId, data: T) -> bool {
        // Fast path: check if already cancelled before queuing
        if self.cancelled.contains(&transfer_id) {
            return false;
        }

        self.inner.push(QueueItem::new(transfer_id, data));
        self.len.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Pop an item from the queue.
    ///
    /// Returns `None` if the queue is empty.
    /// Items from cancelled transfers may still be returned - use `pop_valid()`
    /// if you want to skip cancelled items automatically.
    pub fn pop(&self) -> Option<QueueItem<T>> {
        let item = self.inner.pop();
        if item.is_some() {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }
        item
    }

    /// Pop a valid (non-cancelled) item from the queue.
    ///
    /// Skips and drops items belonging to cancelled transfers.
    /// Returns `None` if no valid items are available.
    pub fn pop_valid(&self) -> Option<QueueItem<T>> {
        loop {
            match self.inner.pop() {
                Some(item) => {
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    if self.cancelled.contains(&item.transfer_id) {
                        // Drop cancelled item and try again
                        continue;
                    }
                    return Some(item);
                }
                None => return None,
            }
        }
    }

    /// Mark a transfer as cancelled.
    ///
    /// Items belonging to this transfer will be:
    /// - Dropped immediately if pushed after this call
    /// - Removed by the sweeper task if already in the queue
    /// - Skipped by `pop_valid()` if dequeued
    pub fn mark_cancelled(&self, transfer_id: TransferId) {
        self.cancelled.insert(transfer_id);
    }

    /// Check if a transfer has been cancelled.
    pub fn is_cancelled(&self, transfer_id: TransferId) -> bool {
        self.cancelled.contains(&transfer_id)
    }

    /// Remove cancelled items from the queue.
    ///
    /// This is called by the sweeper task to actively remove items from
    /// cancelled transfers, ensuring their resources (like `ImmutableBlock` guards)
    /// are released promptly.
    ///
    /// Returns the number of items removed.
    ///
    /// # Implementation Note
    ///
    /// This performs a full drain-and-requeue operation. While not ideal for
    /// very large queues, it ensures correctness with the lock-free SegQueue.
    /// For typical offload workloads (batches of 64-256 blocks), this is efficient.
    pub fn sweep(&self) -> usize {
        if self.cancelled.is_empty() {
            return 0;
        }

        // Drain all items and requeue non-cancelled ones
        let mut removed = 0;
        let mut kept = Vec::new();

        while let Some(item) = self.inner.pop() {
            if self.cancelled.contains(&item.transfer_id) {
                removed += 1;
                // Item is dropped here, releasing any held resources
            } else {
                kept.push(item);
            }
        }

        // Requeue kept items
        for item in kept {
            self.inner.push(item);
        }

        // Update length counter
        if removed > 0 {
            self.len.fetch_sub(removed, Ordering::Relaxed);
        }

        removed
    }

    /// Clear the cancelled set for a specific transfer.
    ///
    /// Called when a transfer is fully complete to clean up the cancelled set.
    pub fn clear_cancelled(&self, transfer_id: TransferId) {
        self.cancelled.remove(&transfer_id);
    }

    /// Get the approximate queue length.
    ///
    /// This is not exact due to concurrent modifications but useful for monitoring.
    pub fn len_approx(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Check if the queue is approximately empty.
    pub fn is_empty_approx(&self) -> bool {
        self.len_approx() == 0
    }

    /// Get the number of cancelled transfers being tracked.
    pub fn cancelled_count(&self) -> usize {
        self.cancelled.len()
    }
}

impl<T> Default for CancellableQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_push_pop() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        assert!(queue.push(id, 42));
        assert_eq!(queue.len_approx(), 1);

        let item = queue.pop().unwrap();
        assert_eq!(item.transfer_id, id);
        assert_eq!(item.data, 42);
        assert_eq!(queue.len_approx(), 0);
    }

    #[test]
    fn test_cancelled_push_rejected() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.mark_cancelled(id);
        assert!(!queue.push(id, 42));
        assert_eq!(queue.len_approx(), 0);
    }

    #[test]
    fn test_pop_valid_skips_cancelled() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id1 = TransferId::new();
        let id2 = TransferId::new();

        queue.push(id1, 1);
        queue.push(id2, 2);
        queue.push(id1, 3);

        queue.mark_cancelled(id1);

        // pop_valid should skip items from id1
        let item = queue.pop_valid().unwrap();
        assert_eq!(item.transfer_id, id2);
        assert_eq!(item.data, 2);

        // No more valid items
        assert!(queue.pop_valid().is_none());
    }

    #[test]
    fn test_sweep_removes_cancelled() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id1 = TransferId::new();
        let id2 = TransferId::new();

        queue.push(id1, 1);
        queue.push(id2, 2);
        queue.push(id1, 3);
        queue.push(id2, 4);

        assert_eq!(queue.len_approx(), 4);

        queue.mark_cancelled(id1);
        let removed = queue.sweep();

        assert_eq!(removed, 2);
        assert_eq!(queue.len_approx(), 2);

        // Remaining items should be from id2
        let item1 = queue.pop().unwrap();
        let item2 = queue.pop().unwrap();
        assert_eq!(item1.transfer_id, id2);
        assert_eq!(item2.transfer_id, id2);
    }

    #[test]
    fn test_sweep_empty_cancelled_set() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.push(id, 1);
        queue.push(id, 2);

        // Sweep with no cancelled transfers should be a no-op
        let removed = queue.sweep();
        assert_eq!(removed, 0);
        assert_eq!(queue.len_approx(), 2);
    }

    #[test]
    fn test_clear_cancelled() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.mark_cancelled(id);
        assert!(queue.is_cancelled(id));
        assert_eq!(queue.cancelled_count(), 1);

        queue.clear_cancelled(id);
        assert!(!queue.is_cancelled(id));
        assert_eq!(queue.cancelled_count(), 0);
    }
}
