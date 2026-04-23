// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `MockWorker`: simulates G1↔G2 transfer timing without moving real memory.
//!
//! Flow per transfer:
//! 1. Caller sets the current simulation time via [`MockWorker::set_now_ms`].
//! 2. `OffloadEngine`'s pipeline drain task invokes
//!    [`WorkerTransfers::execute_local_transfer`]; the worker reserves a PS
//!    slot on the appropriate accountant, registers a [`velo::Event`], and
//!    returns a [`TransferCompleteNotification`] wired to the event.
//! 3. [`MockWorker::drain_completions`] advances both accountants under PS
//!    and triggers the event for each drained `TransferId`, unblocking the
//!    pipeline.
//!
//! Non-G1↔G2 methods (remote NIXL, cross-instance, G4 object storage) return
//! `bail!` / all-Err futures — simulation is deliberately restricted to G2.
//!
//! [`MockWorker::cancel_transfer`] pairs with `OffloadEngine`'s cancel
//! machinery: without it, a cancelled transfer would leave a phantom entry
//! in the PS active set and inflate the denominator for everything after.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use futures::future::BoxFuture;

use kvbm_common::LogicalLayoutHandle;
use kvbm_engine::object::ObjectBlockOps;
use kvbm_engine::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, RemoteDescriptor, SerializedLayoutResponse,
    Worker, WorkerTransfers,
};
use kvbm_engine::{BlockId, InstanceId, SequenceHash};
use kvbm_physical::manager::{LayoutHandle, SerializedLayout};
use kvbm_physical::transfer::{PhysicalLayout, TransferCompleteNotification, TransferOptions};
use velo::{Event, EventManager};

use super::accountant::{BandwidthAccountant, TransferId};

/// Direction of a G1↔G2 transfer. Picked from the `src` / `dst`
/// `LogicalLayoutHandle` values passed to `execute_local_transfer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    G1ToG2,
    G2ToG1,
}

/// Shared state between `MockWorker` and `MockOffloadEngine`. Both hold an
/// `Arc` clone. The single mutex keeps the two PS accountants and the
/// awaiter map consistent under concurrent access between the scheduler
/// thread (which calls `tick` + `set_now_ms` + any engine-level `enqueue_*`
/// helpers) and the pipeline drain worker thread (which calls
/// `execute_local_transfer`).
pub(crate) struct TransferState {
    offload_bw: BandwidthAccountant,
    onboard_bw: BandwidthAccountant,
    /// Pending `velo::Event`s keyed by the `TransferId` the accountant
    /// issued. When the accountant drains an id on `advance_to`, we
    /// `remove` the `Event` from this map and `trigger()` it.
    awaiters: HashMap<TransferId, Event>,
    /// Completion flags for swap-in reservations. Polled synchronously
    /// by the scheduler via `SwapInHandle::is_complete()` — kept
    /// separate from `awaiters` because swap-in does not feed a velo
    /// notification back into a kvbm-engine pipeline; the scheduler
    /// owns lifecycle directly.
    swap_in_flags: HashMap<TransferId, Arc<std::sync::atomic::AtomicBool>>,
}

impl TransferState {
    fn new(offload_gbps: f64, onboard_gbps: f64) -> Self {
        // One shared `TransferId` counter across both accountants so ids
        // are globally unique. `awaiters` and `swap_in_flags` below are
        // single maps keyed by TransferId; per-accountant counters would
        // hand out overlapping ids and cause completion signals to
        // cross-fire between unrelated transfers.
        let id_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        Self {
            offload_bw: BandwidthAccountant::with_gbps_and_counter(
                offload_gbps,
                id_counter.clone(),
            ),
            onboard_bw: BandwidthAccountant::with_gbps_and_counter(onboard_gbps, id_counter),
            awaiters: HashMap::new(),
            swap_in_flags: HashMap::new(),
        }
    }
}

// Store `now_ms` as integer microseconds so it can round-trip through
// `AtomicU64`. Microsecond precision is more than sufficient for the
// mocker's ms+ tick cadence.
fn ms_to_us(ms: f64) -> u64 {
    (ms.max(0.0) * 1000.0) as u64
}
fn us_to_ms(us: u64) -> f64 {
    (us as f64) / 1000.0
}

/// Mock implementation of kvbm-engine's `Worker`, `WorkerTransfers`, and
/// `ObjectBlockOps` traits. Simulates G1↔G2 transfer timing via a pair of
/// processor-sharing bandwidth accountants; never touches real memory.
///
/// The mode (live / offline) is encoded purely in how the caller sets
/// `now_ms` — wall-clock `elapsed()` for live, virtual `Runtime.now_ms`
/// for offline — so this struct carries no mode marker.
pub struct MockWorker {
    /// Current simulation time, in integer microseconds. Engine stores via
    /// `set_now_ms`; worker reads via `now_ms` from inside the pipeline
    /// drain task.
    now_us: Arc<AtomicU64>,
    /// Shared transfer state. Cloned into `MockOffloadEngine` so its
    /// `tick(now_ms)` can drive completions.
    pub(crate) state: Arc<Mutex<TransferState>>,
    /// Event system for creating completion awaiters.
    event_manager: EventManager,
    /// Bytes per block — used to derive transfer size from block-id counts.
    block_bytes: usize,
    g1_handle: Option<LayoutHandle>,
    g2_handle: Option<LayoutHandle>,
}

impl MockWorker {
    /// Build a new `MockWorker`.
    ///
    /// `offload_gbps` and `onboard_gbps` are throughput caps for the G1→G2
    /// and G2→G1 links respectively; zero or negative values mean
    /// "infinite bandwidth" (transfers complete instantly on next tick).
    /// `block_bytes` is how many bytes each simulation block represents —
    /// typically `block_size * kv_bytes_per_token`.
    pub fn new(
        block_bytes: usize,
        offload_gbps: f64,
        onboard_gbps: f64,
        g1_handle: Option<LayoutHandle>,
        g2_handle: Option<LayoutHandle>,
    ) -> Self {
        Self {
            now_us: Arc::new(AtomicU64::new(0)),
            state: Arc::new(Mutex::new(TransferState::new(offload_gbps, onboard_gbps))),
            event_manager: EventManager::local(),
            block_bytes,
            g1_handle,
            g2_handle,
        }
    }

    /// Update the worker's notion of current simulation time. Engine calls
    /// this at the start of `tick(now_ms)` and before every `enqueue_*` /
    /// `start_swap_in` so the pipeline drain task reads a fresh `now_ms`
    /// inside `execute_local_transfer`.
    pub fn set_now_ms(&self, now_ms: f64) {
        self.now_us.store(ms_to_us(now_ms), Ordering::Release);
    }

    pub fn now_ms(&self) -> f64 {
        us_to_ms(self.now_us.load(Ordering::Acquire))
    }

    /// Advance both accountants to `now_ms` under PS and notify any
    /// completion sinks registered for drained `TransferId`s: `velo::Event`
    /// awaiters (for kvbm-engine pipeline transfers) and `AtomicBool`
    /// flags (for swap-in reservations polled by the scheduler). Called
    /// from `MockOffloadEngine::tick` and implicitly before every new
    /// reservation — both uses need the accountant's active set to
    /// reflect completed transfers at the queried time.
    pub fn drain_completions(&self, now_ms: f64) {
        let mut state = self.state.lock().expect("TransferState mutex poisoned");
        Self::drain_locked(&mut state, now_ms);
    }

    /// Shared drain body used by `drain_completions`, `cancel_transfer`,
    /// `reserve_transfer`, and `reserve_swap_in`. Caller holds the lock.
    fn drain_locked(state: &mut TransferState, now_ms: f64) {
        let drained: Vec<TransferId> = state
            .offload_bw
            .advance_to(now_ms)
            .into_iter()
            .chain(state.onboard_bw.advance_to(now_ms))
            .collect();
        for id in drained {
            if let Some(event) = state.awaiters.remove(&id) {
                // Ignore trigger errors — the velo event system may be
                // shut down during cleanup.
                let _ = event.trigger();
            }
            if let Some(flag) = state.swap_in_flags.remove(&id) {
                flag.store(true, Ordering::Release);
            }
        }
    }

    /// Cancel an in-flight transfer identified by `id`. Correctly updates
    /// both the PS accountant (so remaining transfers see the right `N-1`
    /// share from this moment forward) and the awaiter map (so the pipeline
    /// task doesn't hang on a leaked notification). Swap-in flags are
    /// removed without flipping to `true` — a cancelled swap-in must not
    /// look "complete" to the scheduler; the caller is expected to drop
    /// the `SwapInHandle`.
    pub fn cancel_transfer(&self, id: TransferId) -> bool {
        let now_ms = self.now_ms();
        let mut state = self.state.lock().expect("TransferState mutex poisoned");

        // Bring accountants current before removing the cancelled id, so
        // other active transfers get credit for the pre-cancel interval
        // under the old N rate.
        Self::drain_locked(&mut state, now_ms);

        // Remove the cancelled id (no-op on the accountant that doesn't
        // hold it, or if the drain above already consumed it).
        state.offload_bw.complete(id);
        state.onboard_bw.complete(id);

        // Fire the awaiter so the pipeline task doesn't hang.
        let fired = match state.awaiters.remove(&id) {
            Some(event) => {
                let _ = event.trigger();
                true
            }
            None => false,
        };

        // Drop a swap-in flag without flipping it — cancel is not
        // completion. Return value reflects only awaiter firing to
        // preserve `cancel_transfer`'s original semantics.
        state.swap_in_flags.remove(&id);

        fired
    }

    /// Reserve an onboard (G2→G1) transfer whose completion is observed
    /// via `complete` — `MockOffloadEngine::tick` (or any drain path)
    /// flips this bool when the PS accountant drains the reservation.
    /// Returned `TransferId` is opaque to the caller; it travels only as
    /// far as `MockWorker::cancel_transfer` if cancellation is needed.
    pub fn reserve_swap_in(
        &self,
        now_ms: f64,
        num_blocks: usize,
        complete: Arc<std::sync::atomic::AtomicBool>,
    ) -> TransferId {
        let bytes = num_blocks.saturating_mul(self.block_bytes);
        let mut state = self.state.lock().expect("TransferState mutex poisoned");
        Self::drain_locked(&mut state, now_ms);
        let id = state.onboard_bw.reserve(now_ms, bytes);
        state.swap_in_flags.insert(id, complete);
        id
    }

    /// Earliest pending deadline across both link accountants. `None` if
    /// both are idle. Used by the scheduler's stall-advance.
    pub fn earliest_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("TransferState mutex poisoned");
        state
            .offload_bw
            .earliest_finish()
            .into_iter()
            .chain(state.onboard_bw.earliest_finish())
            .reduce(f64::min)
    }

    /// Reserve a new transfer on the requested link, register a velo
    /// `Event` in the shared awaiter map, and return a
    /// `TransferCompleteNotification` backed by that event's awaiter.
    fn reserve_transfer(
        &self,
        direction: TransferDirection,
        now_ms: f64,
        num_blocks: usize,
    ) -> Result<TransferCompleteNotification> {
        let bytes = num_blocks.saturating_mul(self.block_bytes);
        let mut state = self.state.lock().expect("TransferState mutex poisoned");
        // Bring accountants current before reserving — PS needs an accurate
        // active-set count when picking a rate for the new arrival.
        Self::drain_locked(&mut state, now_ms);

        let id = match direction {
            TransferDirection::G1ToG2 => state.offload_bw.reserve(now_ms, bytes),
            TransferDirection::G2ToG1 => state.onboard_bw.reserve(now_ms, bytes),
        };

        // Allocate a velo event + awaiter. Store the `Event` so we can
        // `trigger()` it later (triggering consumes `self`).
        let event = self
            .event_manager
            .new_event()
            .map_err(|e| anyhow!("MockWorker: failed to allocate velo event: {e}"))?;
        let awaiter = event
            .awaiter()
            .map_err(|e| anyhow!("MockWorker: failed to build event awaiter: {e}"))?;
        state.awaiters.insert(id, event);
        drop(state);

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

/// Map `(src, dst)` logical layout handles to a mocker-supported direction.
fn infer_direction(
    src: LogicalLayoutHandle,
    dst: LogicalLayoutHandle,
) -> Result<TransferDirection> {
    match (src, dst) {
        (LogicalLayoutHandle::G1, LogicalLayoutHandle::G2) => Ok(TransferDirection::G1ToG2),
        (LogicalLayoutHandle::G2, LogicalLayoutHandle::G1) => Ok(TransferDirection::G2ToG1),
        (s, d) => bail!(
            "MockWorker only simulates G1↔G2 transfers; got src={:?} dst={:?}",
            s,
            d
        ),
    }
}

impl WorkerTransfers for MockWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let direction = infer_direction(src, dst)?;
        let now_ms = self.now_ms();
        self.reserve_transfer(direction, now_ms, src_block_ids.len())
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!(
            "MockWorker: execute_remote_onboard not supported (mocker simulates G1↔G2 only)"
        )
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("MockWorker: execute_remote_offload not supported")
    }

    fn connect_remote(
        &self,
        _instance_id: InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        bail!("MockWorker: connect_remote not supported")
    }

    fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("MockWorker: execute_remote_onboard_for_instance not supported")
    }
}

impl Worker for MockWorker {
    fn g1_handle(&self) -> Option<LayoutHandle> {
        self.g1_handle
    }

    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle
    }

    fn g3_handle(&self) -> Option<LayoutHandle> {
        None
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        bail!("MockWorker: export_metadata not supported (mocker is single-instance)")
    }

    fn import_metadata(&self, _metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        bail!("MockWorker: import_metadata not supported (mocker is single-instance)")
    }
}

impl ObjectBlockOps for MockWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        // G4 object storage is not simulated; pretend nothing exists.
        Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn put_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        _layout: PhysicalLayout,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        _layout: PhysicalLayout,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn make_worker() -> MockWorker {
        // 1 GB/s bandwidth on both links, 1 MB per block.
        MockWorker::new(1_000_000, 1.0, 1.0, None, None)
    }

    #[tokio::test]
    async fn mock_worker_single_transfer_completes_on_tick() {
        // A single G1→G2 transfer reserved at t=0 should complete after
        // drain_completions is called with now_ms >= bytes/bandwidth.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let src_ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        let dst_ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        let notification = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                src_ids,
                dst_ids,
                TransferOptions::default(),
            )
            .expect("reservation should succeed");
        // Before drain, the notification is pending (would yield).
        assert!(notification.could_yield());

        // Advance virtual clock past the transfer's finish time and drain.
        worker.drain_completions(1.0);

        // The notification should now resolve immediately.
        notification
            .await
            .expect("transfer notification should resolve Ok after drain");
    }

    #[tokio::test]
    async fn mock_worker_two_concurrent_transfers_complete_at_2x() {
        // PS regression at the Worker layer: two G1→G2 transfers at t=0
        // should both complete at t = 2 * single_transfer_duration.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let mk_ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };

        let n1 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                mk_ids(),
                mk_ids(),
                TransferOptions::default(),
            )
            .unwrap();
        let n2 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                mk_ids(),
                mk_ids(),
                TransferOptions::default(),
            )
            .unwrap();

        // At t = 1 ms (single-transfer duration), both must still be
        // pending under PS — each got 0.5 MB/ms so both have 0.5 MB left.
        worker.drain_completions(1.0);
        assert!(n1.could_yield());
        assert!(n2.could_yield());

        // At t = 2 ms, both finish simultaneously.
        worker.drain_completions(2.0);
        n1.await.expect("n1 should resolve Ok");
        n2.await.expect("n2 should resolve Ok");
    }

    #[tokio::test]
    async fn mock_worker_rejects_unsupported_directions() {
        // Mocker does not simulate G3/G4 — those directions must fail at
        // the Worker layer (not silently succeed as no-ops).
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);

        // G2 → G3 is out of scope.
        let result = worker.execute_local_transfer(
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G3,
            ids.clone(),
            ids,
            TransferOptions::default(),
        );
        let err = match result {
            Ok(_) => panic!("G2→G3 must be rejected"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("G1↔G2"), "unexpected error: {msg}");
    }

    #[tokio::test]
    async fn mock_worker_cancel_removes_phantom_from_active_set() {
        // Regression for the `OffloadEngine` cancel path: cancelling an
        // in-flight transfer must (a) remove it from the PS accountant's
        // active set so later transfers see the correct `N-1` denominator,
        // (b) resolve its notification so the pipeline task doesn't hang,
        // and (c) credit other transfers for the pre-cancel interval at
        // the old `N` rate.
        //
        // Setup: two G1→G2 transfers at t=0, bandwidth 1 GB/s, 1 MB each.
        // Under PS with N=2, each progresses at 0.5 MB/ms. At t=0.5 we
        // cancel T1 without having first drained — `cancel_transfer` must
        // advance_to(0.5) internally before removing T1, so T2 gets credit
        // for 0.25 MB of work (0.5 ms × 0.5 MB/ms). Post-cancel state:
        // T2 remaining = 0.75 MB, N=1, rate 1 MB/ms, so T2 finishes at
        // t = 0.5 + 0.75 = 1.25 ms.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let mk_ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };

        let n1 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                mk_ids(),
                mk_ids(),
                TransferOptions::default(),
            )
            .unwrap();
        let n2 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                mk_ids(),
                mk_ids(),
                TransferOptions::default(),
            )
            .unwrap();

        // Jump the simulation clock to 0.5 ms; do NOT drain beforehand so
        // the advance inside cancel_transfer is the only state update.
        worker.set_now_ms(0.5);

        // Retrieve the first-assigned TransferId via the shared map. Ids
        // are monotonic from 0, so `min` returns the older transfer.
        let first_id = {
            let state = worker.state.lock().unwrap();
            *state.awaiters.keys().min().unwrap()
        };
        assert!(
            worker.cancel_transfer(first_id),
            "cancel_transfer should report the id was active"
        );
        // Double-cancel is a no-op.
        assert!(!worker.cancel_transfer(first_id));

        // Remaining transfer (T2) should now be scheduled to finish at 1.25 ms.
        let earliest = worker.earliest_finish().unwrap();
        assert!(
            (earliest - 1.25).abs() < EPS,
            "expected 1.25 ms, got {earliest}"
        );

        // Cancelled notification resolves as Ok (semantics: `trigger`).
        n1.await.expect("cancelled notification resolves Ok");

        // T2 finishes at 1.25 ms as computed above.
        worker.drain_completions(1.25);
        n2.await.expect("n2 should resolve at 1.25 ms");
    }

    #[tokio::test]
    async fn mock_worker_concurrent_offload_and_swap_in_do_not_collide() {
        // Regression for the transfer-id collision bug: with per-accountant
        // counters, both offload_bw and onboard_bw would hand out id=0 on
        // first reserve, and draining the offload at t=1ms would then also
        // flip the swap-in flag because `awaiters` and `swap_in_flags`
        // share a TransferId keyspace.
        //
        // Setup: 1 GB/s on both links, 1 MB per block.
        // - Reserve swap-in first (→ onboard id, sit in swap_in_flags).
        // - Execute an offload (→ offload id, sit in awaiters). Under PS
        //   with N=2 on DIFFERENT links, each gets the full link, so the
        //   offload finishes at 1 ms, swap-in at 1 ms as well.
        // At 0.99 ms (before either finishes) the swap-in flag must still
        // be false; at 1.0 ms both fire, and the flag may flip (correct)
        // — but we want to verify that a drain BEFORE finish time doesn't
        // cross-fire the flag due to id aliasing.
        use std::sync::atomic::{AtomicBool, Ordering};
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let complete = Arc::new(AtomicBool::new(false));
        let swap_id = worker.reserve_swap_in(0.0, 1, complete.clone());
        let mk_ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };
        let _offload = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                mk_ids(),
                mk_ids(),
                TransferOptions::default(),
            )
            .unwrap();
        // Pre-fix (per-accountant counter): swap_id and offload_id would
        // both be 0 — identical. With shared counter they must differ.
        let offload_id_is_different = {
            let state = worker.state.lock().unwrap();
            state.awaiters.keys().all(|k| *k != swap_id)
                && state.swap_in_flags.keys().all(|k| *k == swap_id)
        };
        assert!(
            offload_id_is_different,
            "awaiter and swap-in must have distinct TransferIds"
        );
        // Draining at 0.5 ms must not flip the swap-in flag (neither
        // finishes yet), and must NOT collide on id=0.
        worker.drain_completions(0.5);
        assert!(
            !complete.load(Ordering::Acquire),
            "swap-in must not complete mid-flight; id collision would falsely flip"
        );
    }

    #[tokio::test]
    async fn mock_worker_swap_in_flag_flips_on_drain() {
        // Reserve a G2→G1 swap-in for 1 block (1 MB at 1 GB/s → 1 ms).
        // Before drain the flag must be false; after advancing past the
        // finish time the same drain must flip it to true.
        use std::sync::atomic::{AtomicBool, Ordering};
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let complete = Arc::new(AtomicBool::new(false));
        let _id = worker.reserve_swap_in(0.0, 1, complete.clone());
        assert!(!complete.load(Ordering::Acquire));
        worker.drain_completions(0.5);
        assert!(
            !complete.load(Ordering::Acquire),
            "swap-in must not complete before its finish time"
        );
        worker.drain_completions(1.0);
        assert!(
            complete.load(Ordering::Acquire),
            "swap-in flag must flip after drain past finish time"
        );
    }

    #[test]
    fn mock_worker_cancel_unknown_id_is_noop() {
        // Defensive: cancelling an id that was never reserved (or was
        // already drained/cancelled) must not panic and must return false.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        assert!(!worker.cancel_transfer(9999));
    }

    #[tokio::test]
    async fn mock_worker_earliest_finish_min_of_both_links() {
        // With both directions active, `earliest_finish` returns the
        // minimum of the two accountants' next-completion times.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);

        // G1→G2 at t=0, finishes at 1.0 ms under PS with N=1.
        let _n1 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                ids.clone(),
                ids.clone(),
                TransferOptions::default(),
            )
            .unwrap();

        // G2→G1 at t=0 on the OTHER link, finishes at 1.0 ms with N=1 too.
        let _n2 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G1,
                ids.clone(),
                ids,
                TransferOptions::default(),
            )
            .unwrap();

        let earliest = worker.earliest_finish().unwrap();
        assert!(
            (earliest - 1.0).abs() < EPS,
            "expected 1.0 ms, got {earliest}"
        );
    }
}
