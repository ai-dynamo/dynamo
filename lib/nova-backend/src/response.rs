// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, zero-allocation slot arena for coordinating async completions.
//! This implemented is specifically a 1:1 completeter/awaiter pattern.

use bytes::Bytes;
use dashmap::DashSet;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;
use tracing::{debug, warn};
use uuid::Uuid;

type WorkerId = u64;

const RESPONSE_SLOT_CAPACITY: usize = 32 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResponseId(Uuid);

impl ResponseId {
    fn from_u128(val: u128) -> Self {
        Self(Uuid::from_u128(val))
    }

    fn as_u128(&self) -> u128 {
        self.0.as_u128()
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for ResponseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Awaiter returned to callers for a registered response.
pub struct ResponseAwaiter {
    response_id: ResponseId,
    manager: Arc<ResponseManagerInner>,
    slot: Arc<Slot<Option<Bytes>, String>>,
    index: usize,
    consumed: bool,
}

impl ResponseAwaiter {
    fn new(
        manager: Arc<ResponseManagerInner>,
        slot: Arc<Slot<Option<Bytes>, String>>,
        index: usize,
    ) -> Self {
        let response_id = manager.encode_key(index as u64);
        Self {
            response_id,
            manager,
            slot,
            index,
            consumed: false,
        }
    }

    /// Identifier to include in the outbound request (acts as message + response key).
    pub fn response_id(&self) -> ResponseId {
        self.response_id
    }

    /// Wait for the response payload, returning the outcome supplied by the responder.
    ///
    /// This method can be called multiple times (e.g., in a tokio::select! loop) until
    /// it successfully receives a value. After successful receipt, subsequent calls will
    /// return an error.
    pub async fn recv(&mut self) -> Result<Option<Bytes>, String> {
        if self.consumed {
            return Err("response awaiter already consumed".to_string());
        }
        self.consumed = true;

        let result = self.slot.wait_and_take().await;
        self.recycle();

        match result {
            Some(outcome) => outcome,
            None => Err("response awaiter dropped before completion".to_string()),
        }
    }

    fn recycle(&self) {
        self.manager.recycle_slot(self.index);
    }
}

impl Drop for ResponseAwaiter {
    fn drop(&mut self) {
        if !self.consumed {
            self.recycle();
        }
    }
}

impl fmt::Debug for ResponseAwaiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResponseAwaiter")
            .field("response_id", &self.response_id)
            .field("consumed", &self.consumed)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResponseRegistrationError {
    #[error("no free response slots")]
    Exhausted,
}

/// Correlates response outcomes (ACK/NACK/payload) using a fixed-capacity slot arena.
pub struct ResponseManager {
    inner: Arc<ResponseManagerInner>,
}

impl ResponseManager {
    pub fn new(worker_id: WorkerId) -> Self {
        Self {
            inner: Arc::new(ResponseManagerInner::new(worker_id)),
        }
    }

    pub fn register_outcome(&self) -> Result<ResponseAwaiter, ResponseRegistrationError> {
        let (index, slot) = self
            .inner
            .try_allocate_slot()
            .ok_or(ResponseRegistrationError::Exhausted)?;

        self.inner.mark_pending();

        Ok(ResponseAwaiter::new(Arc::clone(&self.inner), slot, index))
    }

    pub fn complete_outcome(
        &self,
        response_id: ResponseId,
        outcome: Result<Option<Bytes>, String>,
    ) -> bool {
        self.inner.complete_outcome(response_id, outcome)
    }

    pub fn pending_outcome_count(&self) -> usize {
        self.inner.pending_outcome_count()
    }
}

struct ResponseManagerInner {
    worker_id: WorkerId,
    arena: Arc<SlotArena<Option<Bytes>, String>>,
    pending: AtomicUsize,
    capacity: usize,
}

impl ResponseManagerInner {
    fn new(worker_id: WorkerId) -> Self {
        let arena = SlotArena::with_capacity(RESPONSE_SLOT_CAPACITY);
        Self {
            worker_id,
            arena,
            pending: AtomicUsize::new(0),
            capacity: RESPONSE_SLOT_CAPACITY,
        }
    }

    #[allow(clippy::type_complexity)]
    fn try_allocate_slot(&self) -> Option<(usize, Arc<Slot<Option<Bytes>, String>>)> {
        self.arena.allocate()
    }

    fn recycle_slot(&self, index: usize) {
        self.arena.recycle(index);
        self.pending.fetch_sub(1, Ordering::Relaxed);
    }

    fn mark_pending(&self) {
        self.pending.fetch_add(1, Ordering::Relaxed);
    }

    fn encode_key(&self, slot_index: u64) -> ResponseId {
        let worker_bits = (self.worker_id as u128) << 64;
        let slot_bits = slot_index as u128;

        ResponseId::from_u128(worker_bits | slot_bits)
    }

    fn decode_key(&self, response_id: ResponseId) -> Option<(u64, u64)> {
        let raw = response_id.as_u128();
        let worker_id = (raw >> 64) as u64;
        let slot_index = (raw & 0xFFFF_FFFF_FFFF_FFFF) as u64;

        Some((worker_id, slot_index))
    }

    fn complete_outcome(
        &self,
        response_id: ResponseId,
        outcome: Result<Option<Bytes>, String>,
    ) -> bool {
        debug!(
            response_id = %response_id,
            "ResponseManager.complete_outcome() called - decoding response_id"
        );

        let (worker_id, slot_index_u64) = match self.decode_key(response_id) {
            Some(parts) => parts,
            None => {
                warn!(response_id = %response_id, "invalid response identifier");
                return false;
            }
        };

        debug!(
            response_id = %response_id,
            worker_id,
            slot_index = slot_index_u64,
            "ResponseManager decoded response_id successfully"
        );

        if worker_id != self.worker_id {
            warn!(
                response_id = %response_id,
                expected_worker = self.worker_id,
                received_worker = worker_id,
                "response targeted wrong worker"
            );
            return false;
        }

        let slot_index = slot_index_u64 as usize;
        if slot_index >= self.capacity {
            warn!(
                response_id = %response_id,
                slot_index,
                capacity = self.capacity,
                "response slot index out of bounds"
            );
            return false;
        }

        // Check if slot is currently allocated (not recycled)
        if !self.arena.is_allocated(slot_index) {
            warn!(
                response_id = %response_id,
                slot_index,
                "response slot has been recycled - discarding stale response"
            );
            return false;
        }

        let slot = match self.arena.slot(slot_index) {
            Some(slot) => {
                debug!(
                    response_id = %response_id,
                    slot_index,
                    "ResponseManager found slot in arena"
                );
                slot
            }
            None => {
                warn!(
                    response_id = %response_id,
                    slot_index,
                    "response slot not found (likely freed)"
                );
                return false;
            }
        };

        debug!(
            response_id = %response_id,
            slot_index,
            "ResponseManager completing slot outcome"
        );

        let completed = match outcome {
            Ok(payload) => {
                debug!(
                    response_id = %response_id,
                    slot_index,
                    payload_present = payload.is_some(),
                    "ResponseManager calling slot.complete_ok()"
                );
                slot.complete_ok(payload)
            }
            Err(err) => {
                debug!(
                    response_id = %response_id,
                    slot_index,
                    error = %err,
                    "ResponseManager calling slot.complete_err()"
                );
                slot.complete_err(err)
            }
        };

        if completed {
            debug!(response_id = %response_id, slot_index, "ResponseManager: slot.complete_ok/err RETURNED TRUE - awaiter should wake");
        } else {
            warn!(
                response_id = %response_id,
                slot_index,
                "ResponseManager: slot.complete_ok/err RETURNED FALSE - response outcome already completed or cancelled"
            );
        }
        completed
    }

    fn pending_outcome_count(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }
}

impl Clone for ResponseManager {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl fmt::Debug for ResponseManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResponseManager")
            .field("worker_id", &self.inner.worker_id)
            .field("pending", &self.pending_outcome_count())
            .field("capacity", &self.inner.capacity)
            .finish()
    }
}

struct Slot<T, E> {
    notify: Notify,
    value: Mutex<Option<Result<T, E>>>,
}

impl<T, E> Slot<T, E> {
    pub fn new() -> Self {
        Self {
            notify: Notify::new(),
            value: Mutex::new(None),
        }
    }

    /// Completes the slot with a success payload.
    pub fn complete_ok(&self, val: T) -> bool {
        self.finish(Ok(val))
    }

    /// Completes the slot with an error payload.
    pub fn complete_err(&self, err: E) -> bool {
        self.finish(Err(err))
    }

    fn finish(&self, res: Result<T, E>) -> bool {
        use tracing::debug;
        debug!("Slot.finish() called - locking value");
        let mut guard = self.value.lock();
        if guard.is_some() {
            debug!("Slot.finish() - value already present, returning false");
            return false; // already completed or taken
        }
        *guard = Some(res);
        debug!("Slot.finish() - value set, dropping lock");
        drop(guard);
        debug!("Slot.finish() - calling notify_one()");
        self.notify.notify_one();
        debug!("Slot.finish() - notify_one() called, returning true");
        true
    }

    /// Waits for completion; consumes and returns the result.
    ///
    /// IMPORTANT: Creates the notified future BEFORE checking the value to avoid lost wakeups.
    /// See: https://docs.rs/tokio/latest/tokio/sync/struct.Notify.html#avoiding-lost-wakeups
    pub async fn wait_and_take(&self) -> Option<Result<T, E>> {
        use tracing::debug;
        debug!("Slot.wait_and_take() called - entering loop");
        loop {
            // Create listener FIRST to avoid lost wakeup race condition
            debug!("Slot.wait_and_take() - creating notified future");
            let notified = self.notify.notified();

            // Then check value
            debug!("Slot.wait_and_take() - checking value");
            if let Some(val) = self.value.lock().take() {
                debug!("Slot.wait_and_take() - value present, returning");
                return Some(val);
            }

            debug!("Slot.wait_and_take() - value not present, awaiting notification");
            // Wait for notification (will catch notifications sent between check and wait)
            notified.await;
            debug!("Slot.wait_and_take() - notification received, looping again");
        }
    }

    /// Non-blocking poll version.
    #[allow(dead_code)]
    pub fn try_take(&self) -> Option<Result<T, E>> {
        self.value.lock().take()
    }

    /// Resets the slot (for reuse).
    pub fn recycle(&self) {
        *self.value.lock() = None;
    }
}

struct SlotArena<T, E> {
    slots: Vec<Arc<Slot<T, E>>>,
    free: parking_lot::Mutex<VecDeque<usize>>,
    allocated: DashSet<usize>,
}

impl<T, E> SlotArena<T, E> {
    pub fn with_capacity(cap: usize) -> Arc<Self> {
        let slots = (0..cap).map(|_| Arc::new(Slot::new())).collect();
        Arc::new(Self {
            slots,
            free: parking_lot::Mutex::new((0..cap).collect()),
            allocated: DashSet::new(),
        })
    }

    pub fn allocate(&self) -> Option<(usize, Arc<Slot<T, E>>)> {
        let mut free = self.free.lock();
        free.pop_front().map(|i| {
            self.allocated.insert(i);
            (i, self.slots[i].clone())
        })
    }

    pub fn slot(&self, index: usize) -> Option<Arc<Slot<T, E>>> {
        self.slots.get(index).cloned()
    }

    #[allow(dead_code)]
    pub fn complete(&self, index: usize, val: Result<T, E>) -> bool {
        let slot = &self.slots[index];
        match val {
            Ok(v) => slot.complete_ok(v),
            Err(e) => slot.complete_err(e),
        }
    }

    pub fn recycle(&self, index: usize) {
        self.slots[index].recycle();
        self.allocated.remove(&index);
        self.free.lock().push_back(index);
    }

    pub fn is_allocated(&self, index: usize) -> bool {
        self.allocated.contains(&index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn outcome_registration_and_completion() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        assert!(manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"ping")))));

        let bytes = awaiter.recv().await.unwrap().unwrap();

        // hash the bytes to compare
        let hash = xxhash_rust::xxh3::xxh3_64(&bytes);
        assert_eq!(hash, xxhash_rust::xxh3::xxh3_64(b"ping"));
    }

    #[tokio::test]
    async fn drop_recycles_slot() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);

        let awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        drop(awaiter);

        // Late response should be discarded
        assert!(!manager.complete_outcome(response_id, Ok(None)));
        assert_eq!(manager.pending_outcome_count(), 0);
    }

    #[tokio::test]
    async fn allocation_exhaustion() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);

        let mut awaiters = Vec::with_capacity(RESPONSE_SLOT_CAPACITY);
        for _ in 0..RESPONSE_SLOT_CAPACITY {
            let awaiter = manager.register_outcome().expect("allocate slot");
            awaiters.push(awaiter);
        }

        assert!(matches!(
            manager.register_outcome(),
            Err(ResponseRegistrationError::Exhausted)
        ));

        // Recycle one slot and ensure allocation succeeds again.
        let awaiter = awaiters.pop().expect("awaiter");
        drop(awaiter);

        let awaiter = manager.register_outcome().expect("allocate after recycle");
        drop(awaiter);
    }

    #[tokio::test]
    async fn recv_works_with_tokio_select() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Complete the response in a background task
        let manager_clone = manager.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            manager_clone.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"delayed"))));
        });

        // Use awaiter in a select loop (this can be dropped and recreated by select!)
        let result = tokio::select! {
            res = awaiter.recv() => res,
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                Err("timeout".to_string())
            }
        };

        assert!(result.is_ok());
        assert_eq!(result.unwrap().unwrap(), Bytes::from_static(b"delayed"));
    }

    #[tokio::test]
    async fn recv_prevents_double_consumption() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"data"))));

        // First recv should succeed
        let first = awaiter.recv().await;
        assert!(first.is_ok());

        // Second recv should fail
        let second = awaiter.recv().await;
        assert!(second.is_err());
        assert_eq!(second.unwrap_err(), "response awaiter already consumed");
    }

    // ============================================================================
    // Category 1: Error Path Testing
    // ============================================================================

    #[tokio::test]
    async fn complete_with_wrong_worker_id() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Create a manager with a different worker_id
        let other_manager = ResponseManager::new(999);

        // Attempt to complete with wrong worker should fail
        assert!(
            !other_manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"data"))))
        );

        // Verify the original awaiter is still waiting
        // Complete with the correct manager
        assert!(manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"correct")))));
        let result = awaiter.recv().await.unwrap().unwrap();
        assert_eq!(result, Bytes::from_static(b"correct"));
    }

    #[tokio::test]
    async fn complete_with_out_of_bounds_slot() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);

        // Create a fake response_id with out-of-bounds slot index
        let fake_slot_index = (RESPONSE_SLOT_CAPACITY + 1000) as u64;
        let worker_bits = (worker_id as u128) << 64;
        let fake_id = ResponseId::from_u128(worker_bits | fake_slot_index as u128);

        // Should reject out-of-bounds slot
        assert!(!manager.complete_outcome(fake_id, Ok(None)));
    }

    #[tokio::test]
    async fn complete_after_recycle_is_rejected() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Drop to recycle the slot
        drop(awaiter);

        // Allocate a new slot (might reuse the same index)
        let _new_awaiter = manager.register_outcome().expect("allocate new slot");

        // Old response_id should be rejected (slot was recycled)
        assert!(!manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"stale")))));
    }

    #[tokio::test]
    async fn double_completion_fails() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // First completion succeeds
        assert!(manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"first")))));

        // Second completion should fail
        assert!(!manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"second")))));

        // Awaiter should receive the first value
        let result = awaiter.recv().await.unwrap().unwrap();
        assert_eq!(result, Bytes::from_static(b"first"));
    }

    #[tokio::test]
    async fn complete_with_error_outcome() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Complete with error
        assert!(manager.complete_outcome(response_id, Err("operation failed".to_string())));

        // Awaiter should receive the error
        let result = awaiter.recv().await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "operation failed");
    }

    // ============================================================================
    // Category 2: State Management
    // ============================================================================

    #[tokio::test]
    async fn pending_count_tracking() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);

        assert_eq!(manager.pending_outcome_count(), 0);

        // Register 3 outcomes
        let mut awaiter1 = manager.register_outcome().expect("allocate 1");
        assert_eq!(manager.pending_outcome_count(), 1);

        let awaiter2 = manager.register_outcome().expect("allocate 2");
        assert_eq!(manager.pending_outcome_count(), 2);

        let mut awaiter3 = manager.register_outcome().expect("allocate 3");
        assert_eq!(manager.pending_outcome_count(), 3);

        // Complete and recv one
        manager.complete_outcome(awaiter1.response_id(), Ok(None));
        awaiter1.recv().await.unwrap();
        assert_eq!(manager.pending_outcome_count(), 2);

        // Drop one unconsumed
        drop(awaiter2);
        assert_eq!(manager.pending_outcome_count(), 1);

        // Complete and recv the last one
        manager.complete_outcome(awaiter3.response_id(), Ok(None));
        awaiter3.recv().await.unwrap();
        assert_eq!(manager.pending_outcome_count(), 0);
    }

    #[tokio::test]
    async fn slot_reuse_after_recycling() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);

        // Allocate and track response IDs
        let mut first_awaiter = manager.register_outcome().expect("allocate first");
        let first_id = first_awaiter.response_id();

        // Complete and consume
        manager.complete_outcome(first_id, Ok(Some(Bytes::from_static(b"first"))));
        first_awaiter.recv().await.unwrap();

        // Allocate again - may reuse the same slot
        let mut second_awaiter = manager.register_outcome().expect("allocate second");
        let second_id = second_awaiter.response_id();

        // IDs should be different (different generation/allocation)
        assert_ne!(first_id, second_id);

        // Old ID should not affect new slot
        assert!(!manager.complete_outcome(first_id, Ok(Some(Bytes::from_static(b"stale")))));

        // New ID should work correctly
        assert!(manager.complete_outcome(second_id, Ok(Some(Bytes::from_static(b"second")))));
        let result = second_awaiter.recv().await.unwrap().unwrap();
        assert_eq!(result, Bytes::from_static(b"second"));
    }

    #[tokio::test]
    async fn allocated_set_accuracy() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);

        // Allocate multiple slots
        let mut awaiters = vec![];
        for _ in 0..10 {
            awaiters.push(manager.register_outcome().expect("allocate"));
        }

        // All should be tracked as allocated
        assert_eq!(manager.pending_outcome_count(), 10);

        // Drop half
        awaiters.truncate(5);
        assert_eq!(manager.pending_outcome_count(), 5);

        // Allocate more - should reuse freed slots
        for _ in 0..5 {
            awaiters.push(manager.register_outcome().expect("allocate"));
        }
        assert_eq!(manager.pending_outcome_count(), 10);
    }

    #[tokio::test]
    async fn none_payload_handling() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Complete with Ok(None)
        assert!(manager.complete_outcome(response_id, Ok(None)));

        // Should successfully receive None
        let result = awaiter.recv().await.unwrap();
        assert!(result.is_none());
    }

    // ============================================================================
    // Category 3: Concurrency
    // ============================================================================

    #[tokio::test]
    async fn concurrent_allocation() {
        let worker_id = 42;
        let manager = Arc::new(ResponseManager::new(worker_id));

        let mut handles = vec![];
        let allocation_count = 100;

        // Spawn multiple tasks allocating concurrently
        for _ in 0..allocation_count {
            let mgr = Arc::clone(&manager);
            let handle = tokio::spawn(async move { mgr.register_outcome().expect("allocate") });
            handles.push(handle);
        }

        // Collect all awaiters
        let mut awaiters = vec![];
        for handle in handles {
            awaiters.push(handle.await.unwrap());
        }

        // All allocations should succeed
        assert_eq!(awaiters.len(), allocation_count);

        // All response IDs should be unique
        let mut ids = std::collections::HashSet::new();
        for awaiter in &awaiters {
            assert!(ids.insert(awaiter.response_id()));
        }
        assert_eq!(ids.len(), allocation_count);

        // Pending count should be accurate
        assert_eq!(manager.pending_outcome_count(), allocation_count);
    }

    #[tokio::test]
    async fn concurrent_completion() {
        let worker_id = 42;
        let manager = Arc::new(ResponseManager::new(worker_id));

        // Allocate multiple slots
        let mut awaiters = vec![];
        for _ in 0..50 {
            awaiters.push(manager.register_outcome().expect("allocate"));
        }

        // Spawn tasks to complete them concurrently
        let mut handles = vec![];
        for (i, awaiter) in awaiters.iter().enumerate() {
            let mgr = Arc::clone(&manager);
            let response_id = awaiter.response_id();
            let handle = tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_micros(i as u64 * 10)).await;
                mgr.complete_outcome(response_id, Ok(Some(Bytes::from(format!("data-{}", i)))))
            });
            handles.push(handle);
        }

        // Wait for all completions
        for handle in handles {
            assert!(handle.await.unwrap());
        }

        // All awaiters should be able to receive their values
        for (i, mut awaiter) in awaiters.into_iter().enumerate() {
            let result = awaiter.recv().await.unwrap().unwrap();
            assert_eq!(result, Bytes::from(format!("data-{}", i)));
        }

        assert_eq!(manager.pending_outcome_count(), 0);
    }

    #[tokio::test]
    async fn race_drop_and_complete() {
        let worker_id = 42;
        let manager = Arc::new(ResponseManager::new(worker_id));

        for iteration in 0..100 {
            let awaiter = manager.register_outcome().expect("allocate");
            let response_id = awaiter.response_id();

            let mgr = Arc::clone(&manager);
            let complete_handle = tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_micros(iteration % 3)).await;
                mgr.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"data"))))
            });

            let drop_handle = tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_micros((iteration + 1) % 3)).await;
                drop(awaiter);
            });

            // Wait for both - one should win
            let complete_result = complete_handle.await.unwrap();
            drop_handle.await.unwrap();

            // Completion may succeed or fail depending on timing
            // Either way, no panic should occur and state should be consistent
            let _ = complete_result;
        }

        // All slots should be recycled by now
        assert_eq!(manager.pending_outcome_count(), 0);
    }

    // ============================================================================
    // Category 4: UUID Operations
    // ============================================================================

    #[tokio::test]
    async fn encode_decode_boundary_values() {
        let max_worker_id = u64::MAX;
        let manager = ResponseManager::new(max_worker_id);

        // Allocate and get response_id with maximum worker_id
        let awaiter = manager.register_outcome().expect("allocate");
        let response_id = awaiter.response_id();

        // Decode manually to verify
        let raw = response_id.as_u128();
        let decoded_worker = (raw >> 64) as u64;
        let decoded_slot = (raw & 0xFFFF_FFFF_FFFF_FFFF) as u64;

        assert_eq!(decoded_worker, max_worker_id);
        assert_eq!(decoded_slot, 0); // First allocation
    }

    #[tokio::test]
    async fn uuid_round_trip_correctness() {
        let worker_id = 0x1234_5678_9ABC_DEF0u64;
        let manager = ResponseManager::new(worker_id);

        // Allocate multiple slots
        for expected_slot in 0..10 {
            let awaiter = manager.register_outcome().expect("allocate");
            let response_id = awaiter.response_id();

            // Manually decode and verify
            let raw = response_id.as_u128();
            let decoded_worker = (raw >> 64) as u64;
            let decoded_slot = (raw & 0xFFFF_FFFF_FFFF_FFFF) as u64;

            assert_eq!(decoded_worker, worker_id);
            assert_eq!(decoded_slot as usize, expected_slot);

            // Verify it can be completed with decoded ID
            assert!(manager.complete_outcome(response_id, Ok(None)));
        }
    }

    // ============================================================================
    // Category 5: Integration
    // ============================================================================

    #[tokio::test]
    async fn manager_clone_shares_state() {
        let worker_id = 42;
        let manager1 = ResponseManager::new(worker_id);
        let manager2 = manager1.clone();

        // Allocate with first manager
        let mut awaiter1 = manager1.register_outcome().expect("allocate with manager1");
        let response_id1 = awaiter1.response_id();

        // Complete with second manager (cloned)
        assert!(manager2.complete_outcome(response_id1, Ok(Some(Bytes::from_static(b"shared")))));

        // Receive with first manager's awaiter
        let result = awaiter1.recv().await.unwrap().unwrap();
        assert_eq!(result, Bytes::from_static(b"shared"));

        // Both managers should see the same pending count
        assert_eq!(manager1.pending_outcome_count(), 0);
        assert_eq!(manager2.pending_outcome_count(), 0);

        // Allocate with manager2, complete with manager1
        let mut awaiter2 = manager2.register_outcome().expect("allocate with manager2");
        let response_id2 = awaiter2.response_id();
        assert!(manager1.complete_outcome(response_id2, Ok(Some(Bytes::from_static(b"reverse")))));
        let result = awaiter2.recv().await.unwrap().unwrap();
        assert_eq!(result, Bytes::from_static(b"reverse"));
    }
}
