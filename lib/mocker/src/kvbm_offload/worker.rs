// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `MockWorker`: simulates G1↔G2 transfer timing without moving real memory.
//!
//! Flow per transfer:
//! 1. Caller sets the current simulation time via [`MockWorker::set_now_ms`].
//! 2. `OffloadEngine`'s pipeline drain task invokes
//!    [`WorkerTransfers::execute_local_transfer`]; the worker reserves a PS
//!    slot on the appropriate model, registers a [`velo::Event`], and
//!    returns a [`TransferCompleteNotification`] wired to the event.
//! 3. [`MockWorker::drain_completions`] advances both models under PS
//!    and triggers the event for each drained `TransferId`, unblocking the
//!    pipeline.
//!
//! Non-G1↔G2 methods (remote NIXL, cross-instance, G4 object storage) return
//! `bail!` / all-Err futures — simulation is deliberately restricted to G2.

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
use tokio::sync::Notify;
use velo::{Event, EventManager};

use super::bandwidth_sharing_model::{BandwidthSharingModel, TransferId};

/// Direction of a G1↔G2 transfer. Picked from the `src` / `dst`
/// `LogicalLayoutHandle` values passed to `execute_local_transfer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    G1ToG2,
    G2ToG1,
}

struct PipelineAwaiter {
    event: Event,
    direction: TransferDirection,
    num_blocks: usize,
}

/// Shared state between `MockWorker` and `MockOffloadEngine`. Both hold an
/// `Arc` clone. The single mutex keeps the two PS models and the
/// awaiter map consistent under concurrent access between the scheduler
/// thread (which calls `tick` + `set_now_ms` + any engine-level `enqueue_*`
/// helpers) and the pipeline drain worker thread (which calls
/// `execute_local_transfer`).
pub(crate) struct TransferState {
    offload_bw: BandwidthSharingModel,
    onboard_bw: BandwidthSharingModel,
    /// Pending `velo::Event`s keyed by the `TransferId` the model
    /// issued. When the model drains an id on `advance_to`, we
    /// `remove` the `Event` from this map and `trigger()` it.
    awaiters: HashMap<TransferId, PipelineAwaiter>,
    /// Completion flags for swap-in reservations. Polled synchronously
    /// by the scheduler via `SwapInHandle::is_complete()` — kept
    /// separate from `awaiters` because swap-in does not feed a velo
    /// notification back into a kvbm-engine pipeline; the scheduler
    /// owns lifecycle directly.
    swap_in_flags: HashMap<TransferId, Arc<std::sync::atomic::AtomicBool>>,
}

impl TransferState {
    fn new(offload_gbps: f64, onboard_gbps: f64) -> Self {
        // One shared `TransferId` counter across both models so ids
        // are globally unique. `awaiters` and `swap_in_flags` below are
        // single maps keyed by TransferId; per-model counters would
        // hand out overlapping ids and cause completion signals to
        // cross-fire between unrelated transfers.
        let id_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        Self {
            offload_bw: BandwidthSharingModel::new(offload_gbps, id_counter.clone()),
            onboard_bw: BandwidthSharingModel::new(onboard_gbps, id_counter),
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
/// processor-sharing bandwidth models; never touches real memory.
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
    /// Monotonic count of pipeline transfer reservations accepted by
    /// `reserve_transfer`. Offline replay uses this as a concrete barrier:
    /// enqueue should not let virtual time jump until the worker has
    /// actually reserved the simulated bandwidth slot.
    reservation_count: AtomicU64,
    /// Wakes offline barriers waiting for the kvbm-engine pipeline to
    /// reserve simulated bandwidth after an enqueue.
    reservation_notify: Arc<Notify>,
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
            reservation_count: AtomicU64::new(0),
            reservation_notify: Arc::new(Notify::new()),
            block_bytes,
            g1_handle,
            g2_handle,
        }
    }

    /// Update the worker's notion of current simulation time. Engine calls
    /// this at the start of `tick(now_ms)` and before every `enqueue_*` /
    /// `try_onboard_prefix` so transfer reservation reads a fresh `now_ms`
    /// inside `execute_local_transfer`.
    pub fn set_now_ms(&self, now_ms: f64) {
        self.now_us.store(ms_to_us(now_ms), Ordering::Release);
    }

    pub fn now_ms(&self) -> f64 {
        us_to_ms(self.now_us.load(Ordering::Acquire))
    }

    pub fn reservation_count(&self) -> u64 {
        self.reservation_count.load(Ordering::Acquire)
    }

    pub(crate) fn reservation_notifier(&self) -> Arc<Notify> {
        self.reservation_notify.clone()
    }

    /// Advance both models to `now_ms` under PS and notify any
    /// completion sinks registered for drained `TransferId`s: `velo::Event`
    /// awaiters (for kvbm-engine pipeline transfers) and `AtomicBool`
    /// flags (for swap-in reservations polled by the scheduler). Called
    /// from `MockOffloadEngine::tick` and implicitly before every new
    /// reservation — both uses need the model's active set to
    /// reflect completed transfers at the queried time.
    pub fn drain_completions(&self, now_ms: f64) -> (usize, usize, usize, usize) {
        let mut state = self.state.lock().expect("TransferState mutex poisoned");
        Self::drain_locked(&mut state, now_ms)
    }

    /// Shared drain body used by `drain_completions`, `reserve_transfer`, and
    /// `reserve_swap_in`. Caller holds the lock.
    fn drain_locked(state: &mut TransferState, now_ms: f64) -> (usize, usize, usize, usize) {
        let offload_before = state.offload_bw.active_count();
        let onboard_before = state.onboard_bw.active_count();
        let offload_drained = state.offload_bw.advance_to(now_ms);
        let onboard_drained = state.onboard_bw.advance_to(now_ms);
        let offload_drained_count = offload_drained.len();
        let onboard_drained_count = onboard_drained.len();
        let drained: Vec<TransferId> = offload_drained.into_iter().chain(onboard_drained).collect();
        tracing::debug!(
            now_ms,
            offload_active_before = offload_before,
            onboard_active_before = onboard_before,
            drained_count = drained.len(),
            offload_drained_count,
            onboard_drained_count,
            awaiter_map_size = state.awaiters.len(),
            "kvbm-offload: drain transfer completions"
        );

        let mut awaiter_fired = 0usize;
        let mut offload_awaiter_blocks = 0usize;
        let mut onboard_awaiter_blocks = 0usize;
        let mut swap_in_flipped = 0usize;
        for id in drained {
            if let Some(awaiter) = state.awaiters.remove(&id) {
                match awaiter.direction {
                    TransferDirection::G1ToG2 => offload_awaiter_blocks += awaiter.num_blocks,
                    TransferDirection::G2ToG1 => onboard_awaiter_blocks += awaiter.num_blocks,
                }
                // Ignore trigger errors — the velo event system may be
                // shut down during cleanup.
                let _ = awaiter.event.trigger();
                awaiter_fired += 1;
            }
            if let Some(flag) = state.swap_in_flags.remove(&id) {
                flag.store(true, Ordering::Release);
                swap_in_flipped += 1;
            }
        }
        tracing::debug!(
            awaiter_fired,
            offload_awaiter_blocks,
            onboard_awaiter_blocks,
            swap_in_flipped,
            "kvbm-offload: fired completed transfer waiters"
        );
        (
            offload_drained_count,
            onboard_drained_count,
            offload_awaiter_blocks,
            onboard_awaiter_blocks,
        )
    }

    /// Reserve an onboard (G2→G1) transfer whose completion is observed
    /// via `complete` — `MockOffloadEngine::tick` (or any drain path)
    /// flips this bool when the PS model drains the reservation.
    pub fn reserve_swap_in(
        &self,
        now_ms: f64,
        num_blocks: usize,
        complete: Arc<std::sync::atomic::AtomicBool>,
    ) -> TransferId {
        let bytes = num_blocks.saturating_mul(self.block_bytes);
        let mut state = self.state.lock().expect("TransferState mutex poisoned");
        Self::drain_locked(&mut state, now_ms);
        let id = state.onboard_bw.start_transfer(now_ms, bytes);
        state.swap_in_flags.insert(id, complete);
        id
    }

    /// Earliest pending deadline across both link models. `None` if
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

    /// Reserve a pipeline transfer on the requested link, register a
    /// velo `Event` in the shared awaiter map, and return a
    /// `TransferCompleteNotification` backed by that event's awaiter.
    fn reserve_transfer(
        &self,
        direction: TransferDirection,
        now_ms: f64,
        num_blocks: usize,
    ) -> Result<TransferCompleteNotification> {
        let bytes = num_blocks.saturating_mul(self.block_bytes);
        let mut state = self.state.lock().expect("TransferState mutex poisoned");
        Self::drain_locked(&mut state, now_ms);

        let id = match direction {
            TransferDirection::G1ToG2 => state.offload_bw.start_transfer(now_ms, bytes),
            TransferDirection::G2ToG1 => state.onboard_bw.start_transfer(now_ms, bytes),
        };
        self.reservation_count.fetch_add(1, Ordering::AcqRel);
        self.reservation_notify.notify_waiters();

        // Allocate a velo event + awaiter. Store the `Event` so we can
        // `trigger()` it later (triggering consumes `self`).
        let event = self
            .event_manager
            .new_event()
            .map_err(|e| anyhow!("MockWorker: failed to allocate velo event: {e}"))?;
        let awaiter = event
            .awaiter()
            .map_err(|e| anyhow!("MockWorker: failed to build event awaiter: {e}"))?;
        state.awaiters.insert(
            id,
            PipelineAwaiter {
                event,
                direction,
                num_blocks,
            },
        );
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
        bail!("MockWorker: execute_remote_onboard not supported (mocker simulates G1↔G2 only)")
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

// G4 is not supported yet. Mock implementation of the ObjectBlockOps trait.
impl ObjectBlockOps for MockWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
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
    async fn mock_worker_offload_and_swap_in_share_id_keyspace() {
        // Invariant: pipeline transfers (`awaiters`) and G2→G1 swap-ins
        // (`swap_in_flags`) live in two HashMaps but share one TransferId
        // keyspace, because `drain_locked` looks up every drained id in
        // both maps. `TransferState::new` enforces this by handing the
        // same Arc<AtomicU64> counter to `offload_bw` and `onboard_bw`.
        //
        // If a future refactor gives each BandwidthSharingModel its own
        // counter, both would start at 0 and the first offload + first
        // swap-in would alias on id=0 — causing a completing offload to
        // falsely flip the swap-in flag (and vice versa). This test pins
        // that invariant: ids drawn across the two models must be disjoint.
        use std::sync::atomic::AtomicBool;
        let worker = make_worker();
        worker.set_now_ms(0.0);

        // Reserve one swap-in (onboard model) and one offload (offload model)
        // at the same virtual time, so both counters are at their initial value.
        let swap_id = worker.reserve_swap_in(0.0, 1, Arc::new(AtomicBool::new(false)));
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        let _offload = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                ids.clone(),
                ids,
                TransferOptions::default(),
            )
            .unwrap();

        let state = worker.state.lock().unwrap();
        let awaiter_id = *state
            .awaiters
            .keys()
            .next()
            .expect("offload must register an awaiter");
        assert_ne!(
            awaiter_id, swap_id,
            "offload and swap-in must draw distinct TransferIds"
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

    #[tokio::test]
    async fn mock_worker_earliest_finish_min_of_both_links() {
        // With both directions active, `earliest_finish` returns the
        // minimum of the two models' next-completion times.
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
