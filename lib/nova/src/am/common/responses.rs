// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, zero-allocation slot arena for coordinating async request/response completions.
//!
//! This module provides a thread-safe, bounded slot arena for coordinating asynchronous
//! request/response patterns. Each slot can be allocated to await a response, and external
//! entities can complete slots using a `ResponseId` that encodes the slot location.
//!
//! # Usage Pattern
//!
//! ```ignore
//! use dynamo_nova::am::common::responses::ResponseManager;
//!
//! let manager = ResponseManager::new(0); // worker_id = 0
//!
//! // Register an outcome to await
//! let mut awaiter = manager.register_outcome()?;
//!
//! // Send the response_id in your request (it serves as both message and response identifier)
//! let response_id = awaiter.response_id();
//!
//! // Later, receive the response (must be in async context)
//! // let result = awaiter.recv().await?;
//! ```
//!
//! # Generation-Based ABA Protection
//!
//! Each slot maintains a generation counter that increments when the slot is recycled.
//! The `ResponseId` encodes both the slot index and generation, preventing stale responses
//! from affecting new awaiters. If an awaiter is dropped and a new one acquires the same
//! slot index, the generation mismatch will cause stale completion attempts to be rejected.
//!
//! # Thread Safety
//!
//! All operations are thread-safe and lock-free where possible. The slot arena uses
//! atomic operations for allocation tracking and mutex-protected state for slot values.
//!
//! # Slot Retirement
//!
//! When a slot's generation counter reaches the maximum value (u48), the slot is
//! permanently retired and will not be returned to the free list. This prevents
//! generation counter wraparound issues.
//!
//! # Known Issues
//!
//! TODO(nova-rpc): Investigate `am_sync` completion path reliability.
//!
//! There is observed behavior where `am_sync` (client) paired with `am_handler_async`
//! (handler) does not reliably complete. The client awaits indefinitely even after
//! the handler finishes processing. This was discovered during KVBM RDMA transfer
//! testing where transfers would timeout.
//!
//! Switching to `unary` (client) with `unary_handler_async` (handler) resolved
//! the issue. The key difference is that:
//! - `am_handler_async` returns `Result<()>` - return value NOT sent to client
//! - `unary_handler_async` returns `Result<Option<Bytes>>` - return value IS sent
//!
//! Areas to investigate:
//! 1. Ack/Event response path in `complete_outcome()` vs Response path
//! 2. Potential race condition in `poll_wait` / `poll_recv` when completion
//!    arrives before waker registration
//! 3. Whether `am_sync` properly registers a response awaiter, or if it uses
//!    a different completion mechanism (Event channel vs Response channel)
//!
//! Current recommendation: Use only `am_send` (fire-and-forget) or `unary`
//! (request-response). Avoid `am_sync` until this is resolved.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use dashmap::DashSet;
use futures::task::AtomicWaker;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::mem::size_of;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use thiserror::Error;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::events::Outcome;

type WorkerId = u64;
type SlotAllocation<T, E> = (usize, u64, Arc<Slot<T, E>>);

const RESPONSE_SLOT_CAPACITY: usize = u16::MAX as usize;
const MAX_GENERATION: u64 = (1u64 << 48) - 1;

#[derive(Debug, Error)]
pub(crate) enum DecodeError {
    #[error("Response header too short: expected at least 18 bytes, got {0}")]
    HeaderTooShort(usize),

    #[error("Invalid headers length")]
    InvalidHeadersLength,

    #[error("Failed to deserialize headers: {0}")]
    HeaderDeserializationError(#[from] rmp_serde::decode::Error),
}

/// Decodes a response header into (ResponseId, Outcome, headers).
///
/// Wire format (19+ bytes):
/// - 16 bytes: response_id (u128, little-endian)
/// - 1 byte: outcome (0 = Ok, 1 = Error)
/// - 2 bytes: headers_len (u16, little-endian)
/// - N bytes: headers (MessagePack encoded HashMap, if headers_len > 0)
#[allow(clippy::type_complexity)]
pub(crate) fn decode_response_header(
    header: Bytes,
) -> Result<(ResponseId, Outcome, Option<HashMap<String, String>>), DecodeError> {
    let mut header = header;

    // Validate minimum header length (16 bytes response_id + 1 byte outcome + 2 bytes headers_len)
    if header.len() < 19 {
        return Err(DecodeError::HeaderTooShort(header.len()));
    }

    // Read response_id (16 bytes / u128)
    let response_id_value = header.get_u128_le();
    let response_id = ResponseId::from_u128(response_id_value);

    // Read outcome (1 byte)
    let outcome_byte = header.get_u8();
    let outcome = if outcome_byte == 0 {
        Outcome::Ok
    } else {
        Outcome::Error
    };

    // Read headers
    let headers_len = header.get_u16_le() as usize;
    let headers = if headers_len > 0 {
        // Validate headers length
        if header.remaining() < headers_len {
            return Err(DecodeError::InvalidHeadersLength);
        }

        let headers_bytes = header.copy_to_bytes(headers_len);
        let headers_map: HashMap<String, String> = rmp_serde::from_slice(&headers_bytes)?;
        Some(headers_map)
    } else {
        None
    };

    Ok((response_id, outcome, headers))
}

/// Encodes a response header with outcome.
///
/// Wire format (19+ bytes):
/// - 16 bytes: response_id (u128, little-endian)
/// - 1 byte: outcome (0 = Ok, 1 = Error)
/// - 2 bytes: headers_len (u16, little-endian)
/// - N bytes: headers (MessagePack encoded HashMap, if headers_len > 0)
#[inline]
pub(crate) fn encode_response_header(
    response_id: ResponseId,
    outcome: Outcome,
    headers: Option<HashMap<String, String>>,
) -> Result<Bytes, rmp_serde::encode::Error> {
    // Encode headers to MessagePack if present
    let headers_bytes = if let Some(ref h) = headers {
        let msgpack_bytes = rmp_serde::to_vec(h)?;
        Some(msgpack_bytes)
    } else {
        None
    };

    let headers_len = headers_bytes.as_ref().map(|b| b.len()).unwrap_or(0);
    // response_id (16) + outcome (1) + headers_len (2) + msgpack
    let capacity = size_of::<u128>() + 1 + 2 + headers_len;
    let mut bytes = BytesMut::with_capacity(capacity);

    // Encode response_id
    bytes.extend_from_slice(&response_id.as_u128().to_le_bytes());

    // Encode outcome
    let outcome_byte: u8 = match outcome {
        Outcome::Ok => 0,
        Outcome::Error => 1,
    };
    bytes.put_u8(outcome_byte);

    // Encode headers_len and headers
    bytes.extend_from_slice(&(headers_len as u16).to_le_bytes());
    if let Some(hbytes) = headers_bytes {
        bytes.extend_from_slice(&hbytes);
    }

    Ok(bytes.freeze())
}

/// Encodes a response key from worker_id, slot_index, and generation.
///
/// Layout (u128):
/// - Bits 0-63: worker_id (u64)
/// - Bits 64-79: slot_index (u16)
/// - Bits 80-127: generation (u64, capped at 48 bits)
#[inline]
fn encode_response_key(worker_id: WorkerId, slot_index: usize, generation: u64) -> u128 {
    let worker_bits = worker_id as u128;
    let slot_bits = ((slot_index as u16) as u128) << 64;
    let gen_bits = (generation as u128) << 80;
    worker_bits | slot_bits | gen_bits
}

/// Decodes a response key into (worker_id, slot_index, generation).
///
/// Layout (u128):
/// - Bits 0-63: worker_id (u64)
/// - Bits 64-79: slot_index (u16)
/// - Bits 80-127: generation (u64, capped at 48 bits)
#[inline]
fn decode_response_key(raw: u128) -> (WorkerId, usize, u64) {
    let worker_id = (raw & 0xFFFF_FFFF_FFFF_FFFF) as u64;
    let slot_index = ((raw >> 64) & 0xFFFF) as u16;
    let generation = ((raw >> 80) & 0xFFFF_FFFF_FFFF) as u64;
    (worker_id, slot_index as usize, generation)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ResponseId(Uuid);

impl ResponseId {
    pub(crate) fn from_u128(val: u128) -> Self {
        Self(Uuid::from_u128(val))
    }

    pub(crate) fn as_u128(&self) -> u128 {
        self.0.as_u128()
    }

    pub(crate) fn worker_id(&self) -> WorkerId {
        let (worker_id, _, _) = decode_response_key(self.as_u128());
        worker_id
    }

    pub(crate) fn slot_index(&self) -> usize {
        let (_, slot_index, _) = decode_response_key(self.as_u128());
        slot_index
    }

    pub(crate) fn generation(&self) -> u64 {
        let (_, _, generation) = decode_response_key(self.as_u128());
        generation
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
        generation: u64,
    ) -> Self {
        let response_id = manager.encode_key(index, generation);
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

        let result = self.slot.wait_and_take().await;
        self.consumed = true;
        self.recycle();

        match result {
            Some(outcome) => outcome,
            None => Err("response awaiter dropped before completion".to_string()),
        }
    }

    /// Poll-based version of recv that can be used in manual Future implementations.
    ///
    /// This avoids heap allocation by using stack-pinning internally.
    /// Returns `Poll::Pending` if the response isn't ready yet, registering the waker
    /// to be notified when it becomes available.
    pub fn poll_recv(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<Option<Bytes>, String>> {
        use std::task::Poll;

        if self.consumed {
            return Poll::Ready(Err("response awaiter already consumed".to_string()));
        }

        match self.slot.poll_wait(cx) {
            Poll::Ready(result) => {
                self.consumed = true;
                self.recycle();
                Poll::Ready(match result {
                    Some(outcome) => outcome,
                    None => Err("response awaiter dropped before completion".to_string()),
                })
            }
            Poll::Pending => Poll::Pending,
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
pub(crate) enum ResponseRegistrationError {
    #[error("no free response slots")]
    Exhausted,
}

/// Correlates response outcomes (ACK/NACK/payload) using a fixed-capacity slot arena.
pub(crate) struct ResponseManager {
    inner: Arc<ResponseManagerInner>,
}

impl ResponseManager {
    pub fn new(worker_id: WorkerId) -> Self {
        Self {
            inner: Arc::new(ResponseManagerInner::new(worker_id)),
        }
    }

    pub fn register_outcome(&self) -> Result<ResponseAwaiter, ResponseRegistrationError> {
        let (index, generation, slot) = self
            .inner
            .try_allocate_slot()
            .ok_or(ResponseRegistrationError::Exhausted)?;

        self.inner.mark_pending();

        Ok(ResponseAwaiter::new(
            Arc::clone(&self.inner),
            slot,
            index,
            generation,
        ))
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

    fn try_allocate_slot(&self) -> Option<SlotAllocation<Option<Bytes>, String>> {
        self.arena.allocate()
    }

    fn recycle_slot(&self, index: usize) {
        self.arena.recycle(index);
        self.pending.fetch_sub(1, Ordering::Release);
    }

    fn mark_pending(&self) {
        self.pending.fetch_add(1, Ordering::AcqRel);
    }

    fn encode_key(&self, slot_index: usize, generation: u64) -> ResponseId {
        ResponseId::from_u128(encode_response_key(self.worker_id, slot_index, generation))
    }

    fn decode_key(&self, response_id: ResponseId) -> Option<(u64, usize, u64)> {
        Some(decode_response_key(response_id.as_u128()))
    }

    fn complete_outcome(
        &self,
        response_id: ResponseId,
        outcome: Result<Option<Bytes>, String>,
    ) -> bool {
        trace!(
            response_id = %response_id,
            "ResponseManager.complete_outcome() called - decoding response_id"
        );

        let (worker_id, slot_index, expected_generation) = match self.decode_key(response_id) {
            Some(parts) => parts,
            None => {
                warn!(response_id = %response_id, "invalid response identifier");
                return false;
            }
        };

        trace!(
            response_id = %response_id,
            worker_id,
            slot_index,
            expected_generation,
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
                trace!(
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

        trace!(
            response_id = %response_id,
            slot_index,
            expected_generation,
            "ResponseManager completing slot outcome"
        );

        let completed = match outcome {
            Ok(payload) => {
                trace!(
                    response_id = %response_id,
                    slot_index,
                    payload_present = payload.is_some(),
                    expected_generation,
                    "ResponseManager calling slot.complete_ok()"
                );
                slot.complete_ok(payload, expected_generation)
            }
            Err(err) => {
                trace!(
                    response_id = %response_id,
                    slot_index,
                    error = %err,
                    expected_generation,
                    "ResponseManager calling slot.complete_err()"
                );
                slot.complete_err(err, expected_generation)
            }
        };

        if completed {
            debug!(response_id = %response_id, slot_index, "ResponseManager: slot.complete_ok/err RETURNED TRUE - awaiter should wake");
        } else {
            warn!(
                response_id = %response_id,
                slot_index,
                expected_generation,
                "ResponseManager: slot.complete_ok/err RETURNED FALSE - response outcome already completed, cancelled, or generation mismatch"
            );
        }
        completed
    }

    fn pending_outcome_count(&self) -> usize {
        self.pending.load(Ordering::Acquire)
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

struct SlotState<T, E> {
    value: Option<Result<T, E>>,
    generation: u64,
    did_finish: bool,
}

impl<T, E> SlotState<T, E> {
    fn new(generation: u64) -> Self {
        Self {
            value: None,
            generation,
            did_finish: false,
        }
    }

    fn can_complete(&self, expected_gen: u64) -> bool {
        self.generation == expected_gen && self.value.is_none()
    }

    fn complete(&mut self, value: Result<T, E>, expected_gen: u64) -> bool {
        if !self.can_complete(expected_gen) {
            return false;
        }
        self.value = Some(value);
        self.generation = self.generation.wrapping_add(1);
        self.did_finish = true;
        true
    }

    fn take_value(&mut self) -> Option<Result<T, E>> {
        self.value.take()
    }

    fn recycle(&mut self) -> u64 {
        if !self.did_finish {
            self.generation = self.generation.wrapping_add(1);
        }
        self.value = None;
        self.did_finish = false;
        self.generation
    }

    fn generation(&self) -> u64 {
        self.generation
    }
}

struct Slot<T, E> {
    /// Waker for async waiting (supports both poll-based and async APIs)
    waker: AtomicWaker,
    state: Mutex<SlotState<T, E>>,
}

impl<T, E> Slot<T, E> {
    pub fn new() -> Self {
        Self {
            waker: AtomicWaker::new(),
            state: Mutex::new(SlotState::new(0)),
        }
    }

    /// Completes the slot with a success payload.
    pub fn complete_ok(&self, val: T, expected_generation: u64) -> bool {
        self.finish(Ok(val), expected_generation)
    }

    /// Completes the slot with an error payload.
    pub fn complete_err(&self, err: E, expected_generation: u64) -> bool {
        self.finish(Err(err), expected_generation)
    }

    fn finish(&self, res: Result<T, E>, expected_generation: u64) -> bool {
        use tracing::{debug, trace};
        trace!("Slot.finish() called - locking state");
        let mut guard = self.state.lock();
        let success = guard.complete(res, expected_generation);
        if success {
            trace!("Slot.finish() - value set, dropping lock");
            drop(guard);
            trace!("Slot.finish() - waking waiter");
            self.waker.wake();
            debug!("Slot.finish() - waiter woken, returning true");
        } else {
            debug!("Slot.finish() - generation mismatch or already completed, returning false");
        }
        success
    }

    /// Waits for completion; consumes and returns the result.
    ///
    /// Uses `poll_wait` internally via `poll_fn` for a unified implementation.
    pub async fn wait_and_take(&self) -> Option<Result<T, E>> {
        std::future::poll_fn(|cx| self.poll_wait(cx)).await
    }

    /// Poll-based waiting that registers the provided waker.
    ///
    /// Uses `AtomicWaker` to properly persist the waker registration across poll calls,
    /// avoiding heap allocation while ensuring no lost wakeups.
    ///
    /// The caller's waker will be woken when `finish()` is called on this slot.
    pub fn poll_wait(
        &self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<T, E>>> {
        use std::task::Poll;

        // Register waker FIRST to avoid lost wakeup race condition
        // AtomicWaker persists the registration across poll calls
        self.waker.register(cx.waker());

        // Then check if value is ready
        if let Some(val) = self.state.lock().take_value() {
            return Poll::Ready(Some(val));
        }

        // Value not ready, waker is registered, return Pending
        Poll::Pending
    }

    // TODO: add polling/query to public api waiters
    /// Non-blocking poll version.
    #[expect(dead_code)]
    #[doc(hidden)]
    pub fn try_take(&self) -> Option<Result<T, E>> {
        self.state.lock().take_value()
    }

    /// Resets the slot (for reuse). Returns the new generation after recycling.
    pub fn recycle(&self) -> u64 {
        self.state.lock().recycle()
    }

    /// Gets the current generation of the slot.
    pub fn current_generation(&self) -> u64 {
        self.state.lock().generation()
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

    pub fn allocate(&self) -> Option<SlotAllocation<T, E>> {
        let mut free = self.free.lock();
        free.pop_front().map(|i| {
            self.allocated.insert(i);
            let generation = self.slots[i].current_generation();
            (i, generation, self.slots[i].clone())
        })
    }

    pub fn slot(&self, index: usize) -> Option<Arc<Slot<T, E>>> {
        self.slots.get(index).cloned()
    }

    #[expect(dead_code)]
    pub fn complete(&self, index: usize, val: Result<T, E>, expected_generation: u64) -> bool {
        let slot = &self.slots[index];
        match val {
            Ok(v) => slot.complete_ok(v, expected_generation),
            Err(e) => slot.complete_err(e, expected_generation),
        }
    }

    pub fn recycle(&self, index: usize) {
        let new_generation = self.slots[index].recycle();
        self.allocated.remove(&index);

        // Only return to free list if generation hasn't exceeded maximum
        if new_generation <= MAX_GENERATION {
            self.free.lock().push_back(index);
        }
        // Otherwise, slot is permanently retired
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
        let decoded_worker = (raw & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        let decoded_slot = ((raw >> 64) & 0xFFFF) as u16;

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
            let decoded_worker = (raw & 0xFFFF_FFFF_FFFF_FFFF) as u64;
            let decoded_slot = ((raw >> 64) & 0xFFFF) as u16;

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

    // ============================================================================
    // Category 6: Generation Counter Tests
    // ============================================================================

    #[tokio::test]
    async fn generation_mismatch_rejection() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Drop awaiter to recycle and increment generation
        drop(awaiter);

        // Try to complete with old response_id (should fail due to generation mismatch)
        assert!(!manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"stale")))));
    }

    #[tokio::test]
    async fn generation_validation_on_complete() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Complete with correct generation should succeed
        assert!(manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"data")))));

        // Try to complete again with same response_id (should fail - already completed)
        assert!(!manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"second")))));

        // Awaiter should receive the first value
        let result = awaiter.recv().await.unwrap().unwrap();
        assert_eq!(result, Bytes::from_static(b"data"));
    }

    #[tokio::test]
    async fn did_finish_prevents_double_increment() {
        let worker_id = 42;
        let manager = ResponseManager::new(worker_id);
        let mut awaiter = manager.register_outcome().expect("allocate slot");
        let response_id = awaiter.response_id();

        // Complete the slot (this increments generation and sets did_finish=true)
        assert!(manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"data")))));

        // Receive the value (this calls recycle, but did_finish=true so no increment)
        awaiter.recv().await.unwrap();

        // Allocate again - should get the incremented generation from finish, not recycle
        let awaiter2 = manager.register_outcome().expect("allocate again");
        let response_id2 = awaiter2.response_id();

        // Old response_id should still be rejected (generation mismatch)
        assert!(!manager.complete_outcome(response_id, Ok(Some(Bytes::from_static(b"stale")))));

        // New response_id should work
        assert!(manager.complete_outcome(response_id2, Ok(Some(Bytes::from_static(b"new")))));
    }

    #[tokio::test]
    async fn concurrent_complete_with_mismatched_generations() {
        let worker_id = 42;
        let manager = Arc::new(ResponseManager::new(worker_id));

        // Allocate and get response_id
        let awaiter = manager.register_outcome().expect("allocate");
        let response_id = awaiter.response_id();

        // Drop to recycle and increment generation
        drop(awaiter);

        // Spawn multiple tasks trying to complete with stale response_id
        let mut handles = vec![];
        for _ in 0..10 {
            let mgr = Arc::clone(&manager);
            let rid = response_id;
            let handle = tokio::spawn(async move {
                mgr.complete_outcome(rid, Ok(Some(Bytes::from_static(b"stale"))))
            });
            handles.push(handle);
        }

        // All should fail due to generation mismatch
        for handle in handles {
            assert!(!handle.await.unwrap());
        }
    }

    // ============================================================================
    // Category 7: Header Decode Error Handling
    // ============================================================================

    #[test]
    fn test_decode_response_header_too_short() {
        // Test with header shorter than 16 bytes
        let short_header = Bytes::from_static(&[1, 2, 3, 4, 5]);
        let result = decode_response_header(short_header);

        assert!(result.is_err(), "Should error on short header");
        match result {
            Err(DecodeError::HeaderTooShort(len)) => {
                assert_eq!(len, 5);
            }
            _ => panic!("Expected HeaderTooShort error"),
        }
    }

    #[test]
    fn test_decode_response_header_empty() {
        // Test with empty header
        let empty_header = Bytes::new();
        let result = decode_response_header(empty_header);

        assert!(result.is_err(), "Should error on empty header");
        match result {
            Err(DecodeError::HeaderTooShort(len)) => {
                assert_eq!(len, 0);
            }
            _ => panic!("Expected HeaderTooShort error"),
        }
    }

    #[test]
    fn test_decode_response_header_exactly_19_bytes() {
        // Test with exactly 19 bytes (16 response_id + 1 outcome + 2 headers_len with no headers)
        let valid_header = Bytes::from(vec![0u8; 19]);
        let result = decode_response_header(valid_header);

        assert!(result.is_ok(), "Should succeed with exactly 19 bytes");
        let (_response_id, outcome, headers) = result.unwrap();
        assert!(matches!(outcome, Outcome::Ok));
        assert!(headers.is_none(), "Should have no headers");
    }

    #[test]
    fn test_decode_response_header_more_than_19_bytes() {
        // Test with more than 19 bytes (extra bytes should be headers)
        let mut data = vec![0u8; 19];
        data.extend_from_slice(&[1, 2, 3, 4]); // Extra bytes
        let long_header = Bytes::from(data);
        let result = decode_response_header(long_header);

        // This should succeed - headers_len is 0, extra bytes are ignored
        assert!(result.is_ok(), "Should handle extra bytes");
    }

    #[test]
    fn test_decode_response_header_18_bytes() {
        // Test with 18 bytes (one byte short - missing headers_len bytes)
        let short_header = Bytes::from(vec![0u8; 18]);
        let result = decode_response_header(short_header);

        assert!(result.is_err(), "Should error with 18 bytes");
        match result {
            Err(DecodeError::HeaderTooShort(len)) => {
                assert_eq!(len, 18);
            }
            _ => panic!("Expected HeaderTooShort error"),
        }
    }

    #[test]
    fn test_decode_response_header_round_trip() {
        // Test encoding and decoding a response ID (without headers)
        let response_id = ResponseId::from_u128(0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0);
        let encoded = encode_response_header(response_id, Outcome::Ok, None).unwrap();

        assert_eq!(
            encoded.len(),
            19,
            "Encoded header should be 19 bytes (16 response_id + 1 outcome + 2 headers_len)"
        );

        let (decoded_id, decoded_outcome, decoded_headers) =
            decode_response_header(encoded).unwrap();
        assert_eq!(decoded_id.as_u128(), response_id.as_u128());
        assert!(matches!(decoded_outcome, Outcome::Ok));
        assert!(decoded_headers.is_none(), "Headers should be None");
    }

    #[test]
    fn test_decode_response_header_error_outcome_round_trip() {
        // Test encoding and decoding a response ID with Error outcome
        let response_id = ResponseId::from_u128(0xABCD_EF01_2345_6789_ABCD_EF01_2345_6789);
        let encoded = encode_response_header(response_id, Outcome::Error, None).unwrap();

        let (decoded_id, decoded_outcome, decoded_headers) =
            decode_response_header(encoded).unwrap();
        assert_eq!(decoded_id.as_u128(), response_id.as_u128());
        assert!(matches!(decoded_outcome, Outcome::Error));
        assert!(decoded_headers.is_none());
    }

    // ============================================================================
    // Response Headers Tests
    // ============================================================================

    #[test]
    fn test_response_headers_encode_decode_round_trip() {
        // Test encoding and decoding response with headers
        let response_id = ResponseId::from_u128(0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0);
        let mut headers = HashMap::new();
        headers.insert("trace-id".to_string(), "abc123".to_string());
        headers.insert("span-id".to_string(), "def456".to_string());

        let encoded =
            encode_response_header(response_id, Outcome::Ok, Some(headers.clone())).unwrap();

        // Should be larger than 19 bytes due to headers
        assert!(encoded.len() > 19, "Should be larger with headers");

        let (decoded_id, decoded_outcome, decoded_headers) =
            decode_response_header(encoded).unwrap();

        assert_eq!(decoded_id.as_u128(), response_id.as_u128());
        assert!(matches!(decoded_outcome, Outcome::Ok));
        assert!(decoded_headers.is_some());

        let decoded_headers = decoded_headers.unwrap();
        assert_eq!(decoded_headers.len(), 2);
        assert_eq!(decoded_headers.get("trace-id").unwrap(), "abc123");
        assert_eq!(decoded_headers.get("span-id").unwrap(), "def456");
    }

    #[test]
    fn test_response_headers_empty_map() {
        // Test with empty headers map
        let response_id = ResponseId::from_u128(12345);
        let headers = HashMap::new();

        let encoded = encode_response_header(response_id, Outcome::Ok, Some(headers)).unwrap();
        let (decoded_id, decoded_outcome, decoded_headers) =
            decode_response_header(encoded).unwrap();

        assert_eq!(decoded_id.as_u128(), response_id.as_u128());
        assert!(matches!(decoded_outcome, Outcome::Ok));
        assert!(decoded_headers.is_some());
        assert_eq!(decoded_headers.unwrap().len(), 0);
    }

    #[test]
    fn test_response_headers_with_unicode() {
        // Test headers with unicode characters
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();
        headers.insert("emoji".to_string(), "🚀".to_string());
        headers.insert("chinese".to_string(), "你好".to_string());

        let encoded =
            encode_response_header(response_id, Outcome::Ok, Some(headers.clone())).unwrap();
        let (decoded_id, decoded_outcome, decoded_headers) =
            decode_response_header(encoded).unwrap();

        assert_eq!(decoded_id.as_u128(), response_id.as_u128());
        assert!(matches!(decoded_outcome, Outcome::Ok));
        let decoded_headers = decoded_headers.unwrap();
        assert_eq!(decoded_headers.get("emoji").unwrap(), "🚀");
        assert_eq!(decoded_headers.get("chinese").unwrap(), "你好");
    }

    #[test]
    fn test_response_headers_many_entries() {
        // Test with many header entries
        let response_id = ResponseId::from_u128(12345);
        let mut headers = HashMap::new();

        for i in 0..50 {
            headers.insert(format!("key-{}", i), format!("value-{}", i));
        }

        let encoded =
            encode_response_header(response_id, Outcome::Ok, Some(headers.clone())).unwrap();
        let (decoded_id, decoded_outcome, decoded_headers) =
            decode_response_header(encoded).unwrap();

        assert_eq!(decoded_id.as_u128(), response_id.as_u128());
        assert!(matches!(decoded_outcome, Outcome::Ok));
        let decoded_headers = decoded_headers.unwrap();
        assert_eq!(decoded_headers.len(), 50);
        assert_eq!(decoded_headers.get("key-25").unwrap(), "value-25");
    }

    #[test]
    fn test_response_headers_none_vs_empty() {
        // Test that None and Some(empty) are different
        let response_id = ResponseId::from_u128(12345);

        // None case
        let encoded_none = encode_response_header(response_id, Outcome::Ok, None).unwrap();
        let none_len = encoded_none.len();
        let (_, outcome_none, headers_none) = decode_response_header(encoded_none).unwrap();
        assert!(matches!(outcome_none, Outcome::Ok));
        assert!(headers_none.is_none());

        // Empty map case
        let empty_map = HashMap::new();
        let encoded_empty =
            encode_response_header(response_id, Outcome::Ok, Some(empty_map)).unwrap();
        let empty_len = encoded_empty.len();
        let (_, outcome_empty, headers_empty) = decode_response_header(encoded_empty).unwrap();
        assert!(matches!(outcome_empty, Outcome::Ok));
        assert!(headers_empty.is_some());
        assert_eq!(headers_empty.unwrap().len(), 0);

        // Sizes should be different
        assert!(empty_len > none_len, "Empty map should be larger than None");
    }
}
