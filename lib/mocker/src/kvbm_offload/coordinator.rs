// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mocker-owned offload lease state and settlement checkpoints.

use std::sync::{Arc, Mutex};

use kvbm_engine::leader::FindMatchesResult;
use kvbm_engine::offload::{
    PipelineLane, SettlementTarget, SettlementToken, TransferHandle, TransferProgressCursor,
    TransferStatus,
};
use kvbm_engine::{BlockId, G2, SequenceHash};
use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};
use tokio::sync::watch;

use crate::common::protocols::G1;

use super::capacity_reservation::CapacityReservationGuard;

new_key_type! {
    /// Generational identity for one coordinator-owned lifecycle.
    pub(crate) struct OffloadId;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaseState {
    Evaluating,
    Active,
    CancelRequested,
    Prepared,
    Finished,
    Cancelled,
    Failed,
}

impl LeaseState {
    fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SwapInTerminal {
    Pending,
    Completed,
    Cancelled,
    Failed(Arc<str>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SwapInStatus {
    pub(crate) terminal: SwapInTerminal,
    pub(crate) block_count: usize,
}

impl SwapInStatus {
    pub(crate) fn pending(block_count: usize) -> Self {
        Self {
            terminal: SwapInTerminal::Pending,
            block_count,
        }
    }

    pub(crate) fn completed(block_count: usize) -> Self {
        Self {
            terminal: SwapInTerminal::Completed,
            block_count,
        }
    }
}

/// Public compatibility handle. Resource ownership stays in the coordinator.
pub struct SwapInHandle {
    pub(crate) id: OffloadId,
    status: watch::Receiver<SwapInStatus>,
}

impl SwapInHandle {
    pub fn is_complete(&self) -> bool {
        !matches!(self.status.borrow().terminal, SwapInTerminal::Pending)
    }

    pub fn block_count(&self) -> usize {
        self.status.borrow().block_count
    }

    pub(crate) fn id(&self) -> OffloadId {
        self.id
    }

    pub(crate) fn terminal(&self) -> SwapInTerminal {
        self.status.borrow().terminal.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LeaseError {
    pub(crate) id: OffloadId,
    pub(crate) message: Arc<str>,
}

struct LaneCheckpoint {
    token: SettlementToken,
    observed_batches: u64,
}

pub(crate) struct SharedSettlement {
    pub(crate) token: SettlementToken,
    pub(crate) target: SettlementTarget,
}

#[derive(Default)]
pub(crate) struct PreparedProgress {
    pub(crate) chains: Vec<PreparedChain>,
    pub(crate) failures: Vec<LeaseError>,
}

pub(crate) struct PreparedChain {
    pub(crate) parent: OffloadId,
    pub(crate) lane: PipelineLane,
    pub(crate) hashes: Vec<SequenceHash>,
}

#[derive(Default)]
pub(crate) struct AcknowledgedProgress {
    pub(crate) released_g1_slots: usize,
    pub(crate) released_g2_reservations: usize,
    pub(crate) released_g3_reservations: usize,
    pub(crate) abandoned_visibility: Vec<SequenceHash>,
}

pub(crate) struct G1ToG2Lease {
    handle: TransferHandle,
    cursor: TransferProgressCursor,
    observed_completed: usize,
    source_slots: FxHashMap<BlockId, MutableBlock<G1>>,
    lower_chain: FxHashMap<BlockId, SequenceHash>,
    pending_visibility: FxHashMap<BlockId, SequenceHash>,
    passed: FxHashSet<BlockId>,
    prepared_source_slots: Vec<MutableBlock<G1>>,
    prepared_g2_reservations: usize,
    prepared_chain: Vec<SequenceHash>,
    prepared_abandoned_visibility: Vec<SequenceHash>,
    chain_to_g3: bool,
    chain_to_g4: bool,
    registered_g3_chain: usize,
    registered_g4_chain: usize,
    unreleased_g2_reservations: usize,
}

impl G1ToG2Lease {
    pub(crate) fn new(
        handle: TransferHandle,
        source_slots: Vec<MutableBlock<G1>>,
        lower_chain: FxHashMap<BlockId, SequenceHash>,
        pending_visibility: FxHashMap<BlockId, SequenceHash>,
        chain_to_g3: bool,
        chain_to_g4: bool,
    ) -> Self {
        Self {
            cursor: handle.new_progress_cursor(),
            handle,
            observed_completed: 0,
            source_slots: source_slots
                .into_iter()
                .map(|slot| (slot.block_id(), slot))
                .collect(),
            lower_chain,
            pending_visibility,
            passed: FxHashSet::default(),
            prepared_source_slots: Vec::new(),
            prepared_g2_reservations: 0,
            prepared_chain: Vec::new(),
            prepared_abandoned_visibility: Vec::new(),
            chain_to_g3,
            chain_to_g4,
            registered_g3_chain: 0,
            registered_g4_chain: 0,
            unreleased_g2_reservations: 0,
        }
    }

    fn prepare(&mut self, allow_completed: bool) -> Result<bool, Arc<str>> {
        let counts = self.handle.progress_counts();
        if counts.completed > self.observed_completed && !allow_completed {
            return Ok(false);
        }
        let delta = self.handle.consume_progress(&mut self.cursor);
        self.observed_completed = counts.completed;
        self.unreleased_g2_reservations = self
            .unreleased_g2_reservations
            .checked_add(delta.passed_blocks.len())
            .ok_or_else(|| Arc::from("G1→G2 reservation count overflow"))?;
        self.passed.extend(delta.passed_blocks);

        for block_id in delta.completed_blocks {
            if let Some(slot) = self.source_slots.remove(&block_id) {
                self.prepared_source_slots.push(slot);
            }
            self.prepare_g2_reservation_release()?;
            if let Some(hash) = self.lower_chain.remove(&block_id) {
                self.prepared_chain.push(hash);
            }
            self.pending_visibility.remove(&block_id);
        }
        for block_id in delta.failed_blocks {
            if let Some(slot) = self.source_slots.remove(&block_id) {
                self.prepared_source_slots.push(slot);
            }
            self.prepare_g2_reservation_release()?;
            self.lower_chain.remove(&block_id);
            if let Some(hash) = self.pending_visibility.remove(&block_id) {
                self.prepared_abandoned_visibility.push(hash);
            }
        }

        if !matches!(self.handle.status(), TransferStatus::Evaluating) {
            self.lower_chain
                .retain(|block_id, _| self.passed.contains(block_id));
        }
        if self.handle.status().is_terminal() {
            self.prepared_source_slots
                .extend(self.source_slots.drain().map(|(_, slot)| slot));
            self.lower_chain.clear();
            self.passed.clear();
            self.prepared_abandoned_visibility
                .extend(self.pending_visibility.drain().map(|(_, hash)| hash));
            self.prepared_g2_reservations = self
                .prepared_g2_reservations
                .checked_add(std::mem::take(&mut self.unreleased_g2_reservations))
                .ok_or_else(|| Arc::from("G1→G2 prepared reservation overflow"))?;
        }
        Ok(!self.prepared_source_slots.is_empty()
            || self.prepared_g2_reservations > 0
            || !self.prepared_chain.is_empty()
            || !self.prepared_abandoned_visibility.is_empty()
            || self.handle.status().is_terminal())
    }

    fn prepare_g2_reservation_release(&mut self) -> Result<(), Arc<str>> {
        self.unreleased_g2_reservations = self
            .unreleased_g2_reservations
            .checked_sub(1)
            .ok_or_else(|| Arc::from("G1→G2 reservation released more than once"))?;
        self.prepared_g2_reservations = self
            .prepared_g2_reservations
            .checked_add(1)
            .ok_or_else(|| Arc::from("G1→G2 prepared reservation overflow"))?;
        Ok(())
    }

    fn prepared_chains(&self, parent: OffloadId) -> Vec<PreparedChain> {
        let mut actions = Vec::with_capacity(2);
        if self.chain_to_g3 && self.registered_g3_chain < self.prepared_chain.len() {
            actions.push(PreparedChain {
                parent,
                lane: PipelineLane::G2ToG3,
                hashes: self.prepared_chain[self.registered_g3_chain..].to_vec(),
            });
        }
        if self.chain_to_g4 && self.registered_g4_chain < self.prepared_chain.len() {
            actions.push(PreparedChain {
                parent,
                lane: PipelineLane::G2ToG4,
                hashes: self.prepared_chain[self.registered_g4_chain..].to_vec(),
            });
        }
        actions
    }

    fn mark_prepared_chain_registered(&mut self, lane: PipelineLane, count: usize) {
        let registered = match lane {
            PipelineLane::G2ToG3 => &mut self.registered_g3_chain,
            PipelineLane::G2ToG4 => &mut self.registered_g4_chain,
            PipelineLane::G1ToG2 => panic!("G1→G2 is not a lower-tier child lane"),
        };
        *registered = registered
            .checked_add(count)
            .expect("registered lower-tier chain count overflow");
        assert!(
            *registered <= self.prepared_chain.len(),
            "registered more lower-tier chain actions than were prepared"
        );
    }

    fn acknowledge(&mut self) -> AcknowledgedProgress {
        assert!(
            !self.chain_to_g3 || self.registered_g3_chain == self.prepared_chain.len(),
            "G1→G2 parent acknowledged before every G2→G3 child was registered"
        );
        assert!(
            !self.chain_to_g4 || self.registered_g4_chain == self.prepared_chain.len(),
            "G1→G2 parent acknowledged before every G2→G4 child was registered"
        );
        let released_g1_slots = self.prepared_source_slots.len();
        self.prepared_source_slots.clear();
        self.prepared_chain.clear();
        self.registered_g3_chain = 0;
        self.registered_g4_chain = 0;
        AcknowledgedProgress {
            released_g1_slots,
            released_g2_reservations: std::mem::take(&mut self.prepared_g2_reservations),
            abandoned_visibility: std::mem::take(&mut self.prepared_abandoned_visibility),
            ..Default::default()
        }
    }

    fn can_finish(&self) -> bool {
        self.handle.status().is_terminal()
            && self.source_slots.is_empty()
            && self.lower_chain.is_empty()
            && self.pending_visibility.is_empty()
            && self.prepared_source_slots.is_empty()
            && self.prepared_chain.is_empty()
            && self.prepared_abandoned_visibility.is_empty()
            && self.prepared_g2_reservations == 0
            && self.unreleased_g2_reservations == 0
    }
}

pub(crate) struct G2ToG3Lease {
    handle: TransferHandle,
    cursor: TransferProgressCursor,
    observed_completed: usize,
    unreleased_reservations: usize,
    prepared_reservations: usize,
}

impl G2ToG3Lease {
    pub(crate) fn new(handle: TransferHandle) -> Self {
        Self {
            cursor: handle.new_progress_cursor(),
            handle,
            observed_completed: 0,
            unreleased_reservations: 0,
            prepared_reservations: 0,
        }
    }

    fn prepare(&mut self, allow_completed: bool) -> Result<bool, Arc<str>> {
        let counts = self.handle.progress_counts();
        if counts.completed > self.observed_completed && !allow_completed {
            return Ok(false);
        }
        let delta = self.handle.consume_progress(&mut self.cursor);
        self.observed_completed = counts.completed;
        self.unreleased_reservations = self
            .unreleased_reservations
            .checked_add(delta.passed_blocks.len())
            .ok_or_else(|| Arc::from("G2→G3 reservation count overflow"))?;
        let settled = delta.completed_blocks.len() + delta.failed_blocks.len();
        self.unreleased_reservations = self
            .unreleased_reservations
            .checked_sub(settled)
            .ok_or_else(|| Arc::from("G2→G3 reservation released more than once"))?;
        self.prepared_reservations = self
            .prepared_reservations
            .checked_add(settled)
            .ok_or_else(|| Arc::from("G2→G3 prepared reservation overflow"))?;
        if self.handle.status().is_terminal() {
            self.prepared_reservations = self
                .prepared_reservations
                .checked_add(std::mem::take(&mut self.unreleased_reservations))
                .ok_or_else(|| Arc::from("G2→G3 prepared reservation overflow"))?;
        }
        Ok(self.prepared_reservations > 0 || self.handle.status().is_terminal())
    }

    fn acknowledge(&mut self) -> AcknowledgedProgress {
        AcknowledgedProgress {
            released_g3_reservations: std::mem::take(&mut self.prepared_reservations),
            ..Default::default()
        }
    }

    fn can_finish(&self) -> bool {
        self.handle.status().is_terminal()
            && self.unreleased_reservations == 0
            && self.prepared_reservations == 0
    }
}

pub(crate) struct G2ToG4Lease {
    handle: TransferHandle,
    cursor: TransferProgressCursor,
    observed_completed: usize,
}

impl G2ToG4Lease {
    pub(crate) fn new(handle: TransferHandle) -> Self {
        Self {
            cursor: handle.new_progress_cursor(),
            handle,
            observed_completed: 0,
        }
    }

    fn prepare(&mut self, allow_completed: bool) -> bool {
        let counts = self.handle.progress_counts();
        if counts.completed > self.observed_completed && !allow_completed {
            return false;
        }
        self.handle.consume_progress(&mut self.cursor);
        self.observed_completed = counts.completed;
        self.handle.status().is_terminal()
    }
}

pub(crate) struct SwapInResources {
    pub(crate) status_tx: watch::Sender<SwapInStatus>,
    pub(crate) g2_blocks: Option<Vec<ImmutableBlock<G2>>>,
    pub(crate) destination_slots: Option<Vec<MutableBlock<G1>>>,
    pub(crate) prefix_pins: Option<Vec<ImmutableBlock<G1>>>,
}

pub(crate) struct DirectSwapInLease {
    resources: SwapInResources,
}

impl DirectSwapInLease {
    pub(crate) fn new(resources: SwapInResources) -> Self {
        Self { resources }
    }
}

pub(crate) struct StagedSwapInLease {
    pub(crate) result: FindMatchesResult,
    pub(crate) reservation_blocks: usize,
    pub(crate) g2_capacity_reservation: Option<CapacityReservationGuard>,
    pub(crate) resources: SwapInResources,
    pub(crate) g2_to_g1_started: bool,
}

enum SwapInLease {
    Direct(DirectSwapInLease),
    Staged(StagedSwapInLease),
}

impl SwapInLease {
    fn resources(&self) -> &SwapInResources {
        match self {
            Self::Direct(lease) => &lease.resources,
            Self::Staged(lease) => &lease.resources,
        }
    }

    fn resources_mut(&mut self) -> &mut SwapInResources {
        match self {
            Self::Direct(lease) => &mut lease.resources,
            Self::Staged(lease) => &mut lease.resources,
        }
    }

    fn into_resources(self) -> SwapInResources {
        match self {
            Self::Direct(lease) => lease.resources,
            Self::Staged(lease) => lease.resources,
        }
    }

    pub(crate) fn is_terminal(&self) -> bool {
        !matches!(
            self.resources().status_tx.borrow().terminal,
            SwapInTerminal::Pending
        )
    }
}

enum LeaseKind {
    G1ToG2(G1ToG2Lease),
    G2ToG3(G2ToG3Lease),
    G2ToG4(G2ToG4Lease),
    SwapIn(SwapInLease),
}

struct PendingOffloadLease {
    state: LeaseState,
    kind: LeaseKind,
}

struct CoordinatorState {
    leases: SlotMap<OffloadId, PendingOffloadLease>,
    g3_checkpoint: Option<LaneCheckpoint>,
    g4_checkpoint: Option<LaneCheckpoint>,
}

pub(crate) struct OffloadCoordinator {
    state: Mutex<CoordinatorState>,
}

impl OffloadCoordinator {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(CoordinatorState {
                leases: SlotMap::with_key(),
                g3_checkpoint: None,
                g4_checkpoint: None,
            }),
        }
    }

    pub(crate) fn insert_g1_to_g2(&self, lease: G1ToG2Lease) -> OffloadId {
        self.insert(LeaseKind::G1ToG2(lease), LeaseState::Evaluating)
    }

    pub(crate) fn begin_shared_lane(&self, lane: PipelineLane, token: SettlementToken) {
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let checkpoint = match lane {
            PipelineLane::G2ToG3 => &mut state.g3_checkpoint,
            PipelineLane::G2ToG4 => &mut state.g4_checkpoint,
            PipelineLane::G1ToG2 => panic!("G1→G2 uses a per-round checkpoint"),
        };
        checkpoint.get_or_insert(LaneCheckpoint {
            token,
            observed_batches: 0,
        });
    }

    pub(crate) fn insert_g2_to_g3(&self, lease: G2ToG3Lease) -> OffloadId {
        self.insert(LeaseKind::G2ToG3(lease), LeaseState::Evaluating)
    }

    pub(crate) fn insert_g2_to_g4(&self, lease: G2ToG4Lease) -> OffloadId {
        self.insert(LeaseKind::G2ToG4(lease), LeaseState::Evaluating)
    }

    pub(crate) fn insert_direct_swap_in(&self, lease: DirectSwapInLease) -> SwapInHandle {
        self.insert_swap_in(SwapInLease::Direct(lease))
    }

    pub(crate) fn insert_staged_swap_in(&self, lease: StagedSwapInLease) -> SwapInHandle {
        self.insert_swap_in(SwapInLease::Staged(lease))
    }

    fn insert_swap_in(&self, lease: SwapInLease) -> SwapInHandle {
        let status = lease.resources().status_tx.subscribe();
        let id = self.insert(LeaseKind::SwapIn(lease), LeaseState::Active);
        SwapInHandle { id, status }
    }

    fn insert(&self, kind: LeaseKind, state: LeaseState) -> OffloadId {
        self.state
            .lock()
            .expect("offload coordinator mutex poisoned")
            .leases
            .insert(PendingOffloadLease { state, kind })
    }

    pub(crate) fn shared_settlement(
        &self,
        lane: PipelineLane,
        newly_observed_batches: usize,
    ) -> Option<SharedSettlement> {
        if newly_observed_batches == 0 {
            return None;
        }
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let checkpoint = match lane {
            PipelineLane::G2ToG3 => state.g3_checkpoint.as_mut(),
            PipelineLane::G2ToG4 => state.g4_checkpoint.as_mut(),
            PipelineLane::G1ToG2 => None,
        }
        .expect("shared completion observed without a pre-drain checkpoint");
        checkpoint.observed_batches = checkpoint
            .observed_batches
            .checked_add(
                u64::try_from(newly_observed_batches).expect("shared completion count exceeds u64"),
            )
            .expect("shared settlement target overflow");
        let mut target = SettlementTarget::new();
        target
            .add_completed_batches(lane, checkpoint.observed_batches)
            .expect("shared settlement target overflow");
        Some(SharedSettlement {
            token: checkpoint.token.clone(),
            target,
        })
    }

    pub(crate) fn prepare_progress(
        &self,
        settled_g1: bool,
        settled_g3: bool,
        settled_g4: bool,
    ) -> PreparedProgress {
        let mut output = PreparedProgress::default();
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        for (id, pending) in state.leases.iter_mut() {
            let prepared = match &mut pending.kind {
                LeaseKind::G1ToG2(lease) => match lease.prepare(settled_g1) {
                    Ok(prepared) => {
                        output.chains.extend(lease.prepared_chains(id));
                        prepared
                    }
                    Err(message) => {
                        pending.state = LeaseState::Failed;
                        output.failures.push(LeaseError { id, message });
                        false
                    }
                },
                LeaseKind::G2ToG3(lease) => match lease.prepare(settled_g3) {
                    Ok(prepared) => prepared,
                    Err(message) => {
                        pending.state = LeaseState::Failed;
                        output.failures.push(LeaseError { id, message });
                        false
                    }
                },
                LeaseKind::G2ToG4(lease) => lease.prepare(settled_g4),
                LeaseKind::SwapIn(_) => false,
            };
            if prepared && !pending.state.is_terminal() {
                pending.state = LeaseState::Prepared;
            }
        }
        output
    }

    pub(crate) fn acknowledge_prepared(&self) -> AcknowledgedProgress {
        let mut output = AcknowledgedProgress::default();
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        for pending in state.leases.values_mut() {
            if pending.state != LeaseState::Prepared {
                continue;
            }
            let progress = match &mut pending.kind {
                LeaseKind::G1ToG2(lease) => lease.acknowledge(),
                LeaseKind::G2ToG3(lease) => lease.acknowledge(),
                LeaseKind::G2ToG4(_) | LeaseKind::SwapIn(_) => AcknowledgedProgress::default(),
            };
            output.released_g1_slots += progress.released_g1_slots;
            output.released_g2_reservations += progress.released_g2_reservations;
            output.released_g3_reservations += progress.released_g3_reservations;
            output
                .abandoned_visibility
                .extend(progress.abandoned_visibility);
            pending.state = LeaseState::Active;
        }

        for pending in state.leases.values_mut() {
            pending.state = match &pending.kind {
                LeaseKind::G1ToG2(lease) if lease.can_finish() => {
                    terminal_lease_state(lease.handle.status())
                }
                LeaseKind::G2ToG3(lease) if lease.can_finish() => {
                    terminal_lease_state(lease.handle.status())
                }
                LeaseKind::G2ToG4(lease) if lease.handle.status().is_terminal() => {
                    terminal_lease_state(lease.handle.status())
                }
                _ => pending.state,
            };
        }

        state
            .leases
            .retain(|_, pending| !pending.state.is_terminal());
        Self::retire_empty_checkpoints(&mut state);
        output
    }

    pub(crate) fn mark_chain_registered(&self, id: OffloadId, lane: PipelineLane, count: usize) {
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let pending = state
            .leases
            .get_mut(id)
            .expect("prepared parent lease disappeared before child registration");
        let LeaseKind::G1ToG2(lease) = &mut pending.kind else {
            panic!("lower-tier chain parent is not a G1→G2 lease");
        };
        lease.mark_prepared_chain_registered(lane, count);
    }

    fn retire_empty_checkpoints(state: &mut CoordinatorState) {
        let has_g3 = state
            .leases
            .values()
            .any(|lease| matches!(lease.kind, LeaseKind::G2ToG3(_)));
        let has_g4 = state
            .leases
            .values()
            .any(|lease| matches!(lease.kind, LeaseKind::G2ToG4(_)));
        if !has_g3 {
            state.g3_checkpoint = None;
        }
        if !has_g4 {
            state.g4_checkpoint = None;
        }
    }

    pub(crate) fn has_live_g1(&self, id: OffloadId) -> bool {
        self.state
            .lock()
            .expect("offload coordinator mutex poisoned")
            .leases
            .get(id)
            .is_some_and(|lease| matches!(lease.kind, LeaseKind::G1ToG2(_)))
    }

    pub(crate) fn cancel_swap_in(&self, id: OffloadId) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let Some(pending) = state.leases.get_mut(id) else {
            return false;
        };
        if !matches!(pending.kind, LeaseKind::SwapIn(_)) {
            return false;
        }
        pending.state = LeaseState::CancelRequested;
        true
    }

    pub(crate) fn with_staged_swap_in_mut<R>(
        &self,
        id: OffloadId,
        f: impl FnOnce(&mut StagedSwapInLease, LeaseState) -> R,
    ) -> Option<R> {
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let pending = state.leases.get_mut(id)?;
        let lease_state = pending.state;
        let LeaseKind::SwapIn(SwapInLease::Staged(lease)) = &mut pending.kind else {
            return None;
        };
        Some(f(lease, lease_state))
    }

    pub(crate) fn staged_swap_in_ids(&self) -> Vec<OffloadId> {
        self.state
            .lock()
            .expect("offload coordinator mutex poisoned")
            .leases
            .iter()
            .filter_map(|(id, lease)| {
                matches!(lease.kind, LeaseKind::SwapIn(SwapInLease::Staged(_))).then_some(id)
            })
            .collect()
    }

    pub(crate) fn take_completed_swap_in(
        &self,
        id: OffloadId,
    ) -> Option<(Vec<MutableBlock<G1>>, Vec<ImmutableBlock<G1>>)> {
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let pending = state.leases.get(id)?;
        let LeaseKind::SwapIn(lease) = &pending.kind else {
            return None;
        };
        if !matches!(
            lease.resources().status_tx.borrow().terminal,
            SwapInTerminal::Completed
        ) {
            return None;
        }
        let pending = state.leases.remove(id)?;
        let LeaseKind::SwapIn(lease) = pending.kind else {
            unreachable!("validated swap-in lease changed kind")
        };
        let mut resources = lease.into_resources();
        Some((
            resources.destination_slots.take().unwrap_or_default(),
            resources.prefix_pins.take().unwrap_or_default(),
        ))
    }

    pub(crate) fn fail_swap_in(&self, id: OffloadId, context: Arc<str>) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let Some(pending) = state.leases.get_mut(id) else {
            return false;
        };
        let LeaseKind::SwapIn(lease) = &mut pending.kind else {
            return false;
        };
        pending.state = LeaseState::Failed;
        if let SwapInLease::Staged(staged) = lease {
            drop(staged.g2_capacity_reservation.take());
            staged.g2_to_g1_started = true;
        }
        let resources = lease.resources_mut();
        let block_count = resources.status_tx.borrow().block_count;
        resources.status_tx.send_replace(SwapInStatus {
            terminal: SwapInTerminal::Failed(context),
            block_count,
        });
        true
    }

    pub(crate) fn reap_terminal_swap_ins(&self) -> usize {
        let mut state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let ids: Vec<_> = state
            .leases
            .iter()
            .filter_map(|(id, pending)| {
                if !matches!(
                    pending.state,
                    LeaseState::CancelRequested | LeaseState::Failed
                ) {
                    return None;
                }
                let LeaseKind::SwapIn(lease) = &pending.kind else {
                    return None;
                };
                lease.is_terminal().then_some(id)
            })
            .collect();
        let mut released = 0usize;
        for id in ids {
            if let Some(PendingOffloadLease {
                kind: LeaseKind::SwapIn(mut lease),
                ..
            }) = state.leases.remove(id)
            {
                let resources = lease.resources_mut();
                if pending_terminal_is_cancel_requested(resources) {
                    let block_count = resources.status_tx.borrow().block_count;
                    resources.status_tx.send_replace(SwapInStatus {
                        terminal: SwapInTerminal::Cancelled,
                        block_count,
                    });
                }
                released += resources
                    .destination_slots
                    .take()
                    .map_or(0, |slots| slots.len());
                released += resources.prefix_pins.take().map_or(0, |pins| pins.len());
            }
        }
        released
    }

    pub(crate) fn live_lease_count(&self) -> usize {
        self.state
            .lock()
            .expect("offload coordinator mutex poisoned")
            .leases
            .len()
    }

    #[cfg(test)]
    pub(crate) fn lane_lease_count(&self, lane: PipelineLane) -> usize {
        self.state
            .lock()
            .expect("offload coordinator mutex poisoned")
            .leases
            .values()
            .filter(|pending| {
                matches!(
                    (&pending.kind, lane),
                    (LeaseKind::G1ToG2(_), PipelineLane::G1ToG2)
                        | (LeaseKind::G2ToG3(_), PipelineLane::G2ToG3)
                        | (LeaseKind::G2ToG4(_), PipelineLane::G2ToG4)
                )
            })
            .count()
    }

    #[cfg(test)]
    pub(crate) fn pending_g1_ownership(&self) -> (Vec<BlockId>, Vec<BlockId>) {
        let state = self
            .state
            .lock()
            .expect("offload coordinator mutex poisoned");
        let mut source_slots = Vec::new();
        let mut passed = Vec::new();
        for pending in state.leases.values() {
            if let LeaseKind::G1ToG2(lease) = &pending.kind {
                source_slots.extend(lease.source_slots.keys().copied());
                passed.extend(lease.handle.passed_blocks());
            }
        }
        source_slots.sort_unstable();
        passed.sort_unstable();
        (source_slots, passed)
    }
}

fn pending_terminal_is_cancel_requested(resources: &SwapInResources) -> bool {
    matches!(
        resources.status_tx.borrow().terminal,
        SwapInTerminal::Completed | SwapInTerminal::Cancelled
    )
}

fn terminal_lease_state(status: TransferStatus) -> LeaseState {
    match status {
        TransferStatus::Complete => LeaseState::Finished,
        TransferStatus::Cancelled => LeaseState::Cancelled,
        TransferStatus::Failed => LeaseState::Failed,
        TransferStatus::Evaluating | TransferStatus::Queued | TransferStatus::Transferring => {
            LeaseState::Active
        }
    }
}
