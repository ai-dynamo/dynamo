// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared connector slot state management and transfer planning utilities.

use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{G1, G2, G3};

use dynamo_nova::events::EventHandle;
use dynamo_tokens::{TokenBlock, TokenBlockSequence};
use tracing::error;
use uuid::Uuid;

use crate::{
    integrations::connector::leader::data::BlocksView,
    physical::{manager::LayoutHandle, transfer::TransferOptions},
    v2::logical::blocks::{BlockId, ImmutableBlock},
};

#[derive(Debug)]
pub struct SlotCore {
    request_id: String,
    sequence: TokenBlockSequence,
    block_size: usize,
    cached_device_tokens: usize,
    cached_host_tokens: usize,
    cached_disk_tokens: usize,
}

impl SlotCore {
    pub fn new(request_id: String, sequence: TokenBlockSequence, block_size: usize) -> Self {
        Self {
            request_id,
            sequence,
            block_size,
            cached_device_tokens: 0,
            cached_host_tokens: 0,
            cached_disk_tokens: 0,
        }
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn sequence(&self) -> &TokenBlockSequence {
        &self.sequence
    }

    pub fn sequence_mut(&mut self) -> &mut TokenBlockSequence {
        &mut self.sequence
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn set_cached_device_tokens(&mut self, num_tokens: usize) {
        self.cached_device_tokens = num_tokens;
    }

    pub fn set_cached_host_tokens(&mut self, num_tokens: usize) {
        self.cached_host_tokens = num_tokens;
    }

    pub fn set_cached_disk_tokens(&mut self, num_tokens: usize) {
        self.cached_disk_tokens = num_tokens;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Initialized,
    OnboardStaged { tokens: usize },
    Onboarding { tokens: usize },
    Prefilling,
    Decoding,
    SkippedPrefill,
    SkippedDecode,
    Preempted,
    Finishing,
    Finished,
}

#[derive(Debug, Clone)]
pub struct OperationInfo {
    pub direction: TransferDirection,
    pub num_blocks: usize,
}

#[derive(Debug)]
pub struct OperationRecord {
    pub id: Uuid,
    pub event: Option<EventHandle>,
    pub info: OperationInfo,
    pub completed: bool,
}

#[derive(Default)]
pub struct InFlightTransfers {
    pub pending: Vec<WorkerTransferRequest>,
    pub active_operations: Vec<TransferOperation>,
    pub candidates: HashMap<Uuid, TransferCandidateRecord>,
    pub candidate_metadata: HashMap<Uuid, TransferCandidateMetadata>,
    pub candidate_artifacts: HashMap<Uuid, Arc<dyn OperationArtifacts>>,
    pub operation_entries: HashMap<Uuid, OperationEntry>,
}

pub struct Slot {
    core: SlotCore,
    state: SlotState,
    inflight: InFlightTransfers,
    operations: Vec<OperationRecord>,
    allow_new_ops: bool,
    pending_event: Option<EventHandle>,
    finish_event: Option<EventHandle>,
    finished_sending: bool,
    finished_receiving: bool,
    computed_tokens: usize,
    evaluated_blocks: usize,

    // Matched blocks from cache tiers
    device_blocks: BlocksView<G1>,
    matched_g2_blocks: Vec<ImmutableBlock<G2>>,
    matched_g3_blocks: Vec<ImmutableBlock<G3>>,

    // Search tracking
    search_started: bool,

    // Onboarding state
    onboarding_in_progress: bool,
}

impl Slot {
    pub fn new(core: SlotCore) -> Self {
        Self {
            core,
            state: SlotState::Initialized,
            inflight: InFlightTransfers::default(),
            operations: Vec::new(),
            allow_new_ops: true,
            pending_event: None,
            finish_event: None,
            finished_sending: false,
            finished_receiving: false,
            computed_tokens: 0,
            evaluated_blocks: 0,
            device_blocks: BlocksView::<G1>::default(),
            matched_g2_blocks: Vec::new(),
            matched_g3_blocks: Vec::new(),
            search_started: false,
            onboarding_in_progress: false,
        }
    }

    pub fn advance_evaluated_blocks(&mut self, num_blocks: usize) {
        self.evaluated_blocks += num_blocks;
    }

    pub fn device_blocks(&self) -> &BlocksView<G1> {
        &self.device_blocks
    }

    pub fn device_blocks_mut(&mut self) -> &mut BlocksView<G1> {
        &mut self.device_blocks
    }

    pub fn core(&self) -> &SlotCore {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut SlotCore {
        &mut self.core
    }

    pub fn state(&self) -> &SlotState {
        &self.state
    }

    pub fn set_state(&mut self, state: SlotState) {
        self.state = state;
    }

    pub fn allow_new_operations(&self) -> bool {
        self.allow_new_ops
    }

    pub fn disallow_new_operations(&mut self) {
        self.allow_new_ops = false;
    }

    pub fn pending_event(&self) -> Option<EventHandle> {
        self.pending_event
    }

    pub fn set_pending_event(&mut self, event: Option<EventHandle>) {
        self.pending_event = event;
    }

    pub fn finish_event(&self) -> Option<EventHandle> {
        self.finish_event
    }

    pub fn set_finish_event(&mut self, event: Option<EventHandle>) {
        self.finish_event = event;
    }

    pub fn mark_finished_sending(&mut self) {
        self.finished_sending = true;
    }

    pub fn mark_finished_receiving(&mut self) {
        self.finished_receiving = true;
    }

    pub fn finished_sending(&self) -> bool {
        self.finished_sending
    }

    pub fn finished_receiving(&self) -> bool {
        self.finished_receiving
    }

    pub fn record_operation(&mut self, id: Uuid, info: OperationInfo, event: Option<EventHandle>) {
        self.operations.push(OperationRecord {
            id,
            event,
            info,
            completed: false,
        });
    }

    pub fn complete_operation(&mut self, id: Uuid) {
        if let Some(entry) = self.operations.iter_mut().find(|op| op.id == id) {
            entry.completed = true;
        }
    }

    pub fn outstanding_operations(&self) -> Vec<Uuid> {
        self.operations
            .iter()
            .filter(|op| !op.completed)
            .map(|op| op.id)
            .collect()
    }

    pub fn record_candidate(&mut self, record: TransferCandidateRecord) {
        self.inflight
            .candidates
            .insert(record.planned.transfer_id, record);
    }

    pub fn take_candidate_metadata(&mut self, uuid: Uuid) -> TransferCandidateMetadata {
        self.inflight
            .candidate_metadata
            .remove(&uuid)
            .unwrap_or_default()
    }

    pub fn store_candidate_metadata(&mut self, uuid: Uuid, metadata: TransferCandidateMetadata) {
        self.inflight.candidate_metadata.insert(uuid, metadata);
    }

    pub fn mark_candidate_promoted(&mut self, uuid: Uuid) -> Option<&mut TransferCandidateRecord> {
        self.inflight.candidates.get_mut(&uuid).map(|record| {
            record.status = TransferCandidateStatus::Promoted;
            record
        })
    }

    pub fn remove_candidate_with_status(
        &mut self,
        uuid: Uuid,
        status: TransferCandidateStatus,
    ) -> Option<TransferCandidateRecord> {
        self.inflight.candidates.remove(&uuid).map(|mut record| {
            record.status = status;
            record
        })
    }

    pub fn skip_candidate(
        &mut self,
        uuid: Uuid,
        reason: TransferSkipReason,
    ) -> Option<TransferCandidateRecord> {
        self.remove_candidate_with_status(uuid, TransferCandidateStatus::Skipped(reason))
    }

    pub fn acknowledge_transfer(&mut self, uuid: Uuid) {
        self.inflight.pending.retain(|req| req.transfer_id != uuid);
        self.inflight.active_operations.retain(|op| op.uuid != uuid);
        if let Some(artifacts) = self.inflight.candidate_artifacts.remove(&uuid) {
            if let Err(err) = artifacts.on_complete() {
                error!(uuid = %uuid, "operation artifact completion failed: {err:?}");
            }
        }
        self.inflight.operation_entries.remove(&uuid);
        self.complete_operation(uuid);
        let _ = self.remove_candidate_with_status(uuid, TransferCandidateStatus::Completed);
    }

    pub fn apply_evaluation_outcome(
        &mut self,
        outcome: TransferEvaluationOutcome,
    ) -> Option<PromotedTransfer> {
        match outcome {
            TransferEvaluationOutcome::Promote(mut promotion) => {
                if let Some(_) = self.mark_candidate_promoted(promotion.uuid) {
                    if let Some(op) = promotion.operation_artifacts.take() {
                        self.inflight.candidate_artifacts.insert(promotion.uuid, op);
                    }
                    if let Some(entry) = promotion.operation_entry.clone() {
                        self.inflight
                            .operation_entries
                            .insert(promotion.uuid, entry);
                    }
                    self.inflight.pending.push(promotion.worker_request.clone());
                    if let Some(meta) = promotion.operation_metadata.take() {
                        self.inflight.active_operations.push(meta);
                    }
                    self.complete_operation(promotion.uuid);
                    Some(PromotedTransfer {
                        uuid: promotion.uuid,
                        execution_plan: promotion.execution_plan,
                        operation_entry: promotion.operation_entry,
                    })
                } else {
                    None
                }
            }
            TransferEvaluationOutcome::Skip(skip) => {
                let _ = self.skip_candidate(skip.uuid, skip.reason);
                None
            }
        }
    }

    // === Matched Block Management ===

    /// Mark that block matching search has started for this slot.
    pub fn set_search_started(&mut self) {
        self.search_started = true;
    }

    /// Check if search has been started for this slot.
    pub fn search_started(&self) -> bool {
        self.search_started
    }

    /// Store matched G2 blocks from CPU tier.
    pub fn store_g2_matches(&mut self, blocks: Vec<ImmutableBlock<G2>>) {
        self.matched_g2_blocks = blocks;
    }

    /// Store matched G3 blocks from disk tier.
    pub fn store_g3_matches(&mut self, blocks: Vec<ImmutableBlock<G3>>) {
        self.matched_g3_blocks = blocks;
    }

    /// Clear all matched blocks.
    pub fn clear_matches(&mut self) {
        self.matched_g2_blocks.clear();
        self.matched_g3_blocks.clear();
    }

    /// Get matched G2 blocks.
    pub fn matched_g2_blocks(&self) -> &[ImmutableBlock<G2>] {
        &self.matched_g2_blocks
    }

    /// Get matched G3 blocks.
    pub fn matched_g3_blocks(&self) -> &[ImmutableBlock<G3>] {
        &self.matched_g3_blocks
    }

    // === Onboarding State Management ===

    /// Mark onboarding as in progress.
    /// Will mark the slot as onboarding and advance the evaluated blocks by the given number of blocks.
    pub fn onboarding_blocks(
        &mut self,
        g1_blocks: &BlocksView<G1>,
        g2_blocks: &BlocksView<G2>,
        g3_blocks: &BlocksView<G3>,
    ) {
        debug_assert!(g1_blocks.len() == g2_blocks.len() + g3_blocks.len());
        tracing::debug!(
            "Onboarding blocks: g1={}, g2={}, g3={}",
            g1_blocks.len(),
            g2_blocks.len(),
            g3_blocks.len()
        );
        self.onboarding_in_progress = true;
        self.evaluated_blocks += g1_blocks.len();
    }

    /// Check if onboarding is in progress.
    pub fn is_onboarding(&self) -> bool {
        self.onboarding_in_progress
    }
}

impl fmt::Debug for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slot")
            .field("state", &self.state)
            .field("request_id", &self.core.request_id)
            .finish()
    }
}

#[derive(Default, Debug, Clone)]
pub struct SlotActions {
    pub transfers: Vec<PlannedTransfer>,
    pub state_notifications: Vec<StateNotification>,
    pub num_cached_device_tokens: Option<usize>,
    pub num_cached_host_tokens: Option<usize>,
    pub num_cached_disk_tokens: Option<usize>,
}

impl SlotActions {
    pub fn with_transfer(mut self, transfer: PlannedTransfer) -> Self {
        self.transfers.push(transfer);
        self
    }

    pub fn with_notification(mut self, state: SlotState, iteration: Option<u64>) -> Self {
        self.state_notifications
            .push(StateNotification { state, iteration });
        self
    }
}

#[derive(Debug, Clone)]
pub struct PlannedTransfer {
    pub transfer_id: Uuid,
    pub direction: TransferDirection,
    pub block_ids: Vec<BlockId>,
    pub token_blocks: Vec<TokenBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    DeviceToHost,
    HostToDevice,
    DiskToDevice,
}

#[derive(Debug, Clone, Copy)]
pub struct StateNotification {
    pub state: SlotState,
    pub iteration: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferCandidateStatus {
    PendingEvaluation,
    Promoted,
    Skipped(TransferSkipReason),
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferSkipReason {
    AlreadyCachedHigherTier,
    SlotStopping,
    Cancelled,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct TransferCandidateRecord {
    pub planned: PlannedTransfer,
    pub metadata: TransferCandidateMetadata,
    pub status: TransferCandidateStatus,
    pub enqueued_iteration: Option<u64>,
}

impl TransferCandidateRecord {
    pub fn new(
        planned: PlannedTransfer,
        metadata: TransferCandidateMetadata,
        enqueued_iteration: Option<u64>,
    ) -> Self {
        Self {
            planned,
            metadata,
            status: TransferCandidateStatus::PendingEvaluation,
            enqueued_iteration,
        }
    }
}

#[derive(Clone)]
pub enum TransferCandidateMetadata {
    Offload {
        token_blocks: Vec<TokenBlock>,
    },
    OnboardHost {
        blocks: Vec<BlockId>,
        artifacts: Option<Arc<dyn OperationArtifacts>>,
    },
    OnboardDisk {
        blocks: Vec<BlockId>,
        artifacts: Option<Arc<dyn OperationArtifacts>>,
    },
    None,
}

impl Default for TransferCandidateMetadata {
    fn default() -> Self {
        TransferCandidateMetadata::None
    }
}

impl fmt::Debug for TransferCandidateMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferCandidateMetadata::Offload { token_blocks } => f
                .debug_struct("Offload")
                .field("num_blocks", &token_blocks.len())
                .finish(),
            TransferCandidateMetadata::OnboardHost { blocks, .. } => f
                .debug_struct("OnboardHost")
                .field("num_blocks", &blocks.len())
                .finish(),
            TransferCandidateMetadata::OnboardDisk { blocks, .. } => f
                .debug_struct("OnboardDisk")
                .field("num_blocks", &blocks.len())
                .finish(),
            TransferCandidateMetadata::None => f.write_str("None"),
        }
    }
}

#[derive(Clone)]
pub enum TransferEvaluationOutcome {
    Promote(TransferPromotion),
    Skip(TransferSkip),
}

#[derive(Clone)]
pub struct TransferPromotion {
    pub uuid: Uuid,
    pub worker_request: WorkerTransferRequest,
    pub execution_plan: TransferExecutionPlan,
    pub operation_metadata: Option<TransferOperation>,
    pub operation_artifacts: Option<Arc<dyn OperationArtifacts>>,
    pub operation_entry: Option<OperationEntry>,
}

#[derive(Clone)]
pub struct TransferSkip {
    pub uuid: Uuid,
    pub reason: TransferSkipReason,
}

#[derive(Clone)]
pub enum TransferExecutionPlan {
    Offload(OffloadPlan),
    Onboard(OnboardPlan),
    Noop,
}

#[derive(Debug, Clone)]
pub struct OffloadPlan {
    pub block_ids: Vec<BlockId>,
    pub token_blocks: Vec<TokenBlock>,
    pub dst_block_ids: Vec<BlockId>,
}

#[derive(Clone)]
pub struct OnboardPlan {
    pub source: TransferSource,
    pub dst_block_ids: Vec<BlockId>,
    pub src_blocks: Vec<BlockId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferSource {
    Host,
    Disk,
}

#[derive(Clone)]
pub struct PromotedTransfer {
    pub uuid: Uuid,
    pub execution_plan: TransferExecutionPlan,
    pub operation_entry: Option<OperationEntry>,
}

impl PromotedTransfer {
    pub fn into_engine_command(self) -> TransferEngineCommand {
        match self.execution_plan {
            TransferExecutionPlan::Offload(plan) => TransferEngineCommand::Offload {
                uuid: self.uuid,
                block_ids: plan.block_ids,
                token_blocks: plan.token_blocks,
                dst_block_ids: plan.dst_block_ids,
                operation_entry: self
                    .operation_entry
                    .unwrap_or_else(|| OperationEntry::offload_defaults(&[])),
            },
            TransferExecutionPlan::Onboard(plan) => TransferEngineCommand::Onboard {
                uuid: self.uuid,
                dst_block_ids: plan.dst_block_ids,
                src_blocks: plan.src_blocks,
                operation_entry: self
                    .operation_entry
                    .unwrap_or_else(|| OperationEntry::onboard_defaults(&[])),
            },
            TransferExecutionPlan::Noop => TransferEngineCommand::Noop { uuid: self.uuid },
        }
    }
}

#[derive(Debug, Clone)]
pub struct OperationEntry {
    pub src_handle: LayoutHandle,
    pub src_block_ids: Vec<BlockId>,
    pub dst_handle: LayoutHandle,
    pub dst_block_ids: Vec<BlockId>,
}

impl OperationEntry {
    pub fn offload_defaults(block_ids: &[BlockId]) -> Self {
        Self {
            src_handle: LayoutHandle::from_u128(0),
            src_block_ids: block_ids.to_vec(),
            dst_handle: LayoutHandle::from_u128(0),
            dst_block_ids: block_ids.to_vec(),
        }
    }

    pub fn onboard_defaults(dst_block_ids: &[BlockId]) -> Self {
        Self {
            src_handle: LayoutHandle::from_u128(0),
            src_block_ids: dst_block_ids.to_vec(),
            dst_handle: LayoutHandle::from_u128(0),
            dst_block_ids: dst_block_ids.to_vec(),
        }
    }
}

#[derive(Clone)]
pub struct WorkerTransferRequest {
    pub transfer_id: Uuid,
    pub src_layout: LayoutHandle,
    pub src_block_ids: Vec<BlockId>,
    pub dst_layout: LayoutHandle,
    pub dst_block_ids: Vec<BlockId>,
    pub options: TransferOptions,
}

impl fmt::Debug for WorkerTransferRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerTransferRequest")
            .field("transfer_id", &self.transfer_id)
            .field("src_layout", &self.src_layout)
            .field("dst_layout", &self.dst_layout)
            .field("num_blocks", &self.src_block_ids.len())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct TransferOperation {
    pub uuid: Uuid,
    pub origin: TransferOrigin,
}

#[derive(Debug, Clone)]
pub enum TransferOrigin {
    ApplyScheduler { num_blocks: usize },
    ExternalMatch { source: TransferSource },
    Reset,
}

pub trait OperationArtifacts: Send + Sync {
    fn operation_entry(&self) -> OperationEntry;
    fn on_complete(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

pub struct StubOperationArtifacts;

impl OperationArtifacts for StubOperationArtifacts {
    fn operation_entry(&self) -> OperationEntry {
        OperationEntry::offload_defaults(&[])
    }

    fn on_complete(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MockMutableBlock {
    layout_handle: LayoutHandle,
    block_id: BlockId,
}

impl MockMutableBlock {
    pub fn new(layout_handle: LayoutHandle, block_id: BlockId) -> Self {
        Self {
            layout_handle,
            block_id,
        }
    }

    pub fn layout_handle(&self) -> LayoutHandle {
        self.layout_handle
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }
}

#[derive(Debug, Clone)]
pub struct MockImmutableBlock {
    layout_handle: LayoutHandle,
    block_id: BlockId,
}

impl MockImmutableBlock {
    pub fn layout_handle(&self) -> LayoutHandle {
        self.layout_handle
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }
}

#[derive(Debug)]
pub struct MockBlockManager {
    next_layout: AtomicU64,
    next_block: AtomicU64,
}

impl MockBlockManager {
    pub fn new() -> Self {
        Self {
            next_layout: AtomicU64::new(1),
            next_block: AtomicU64::new(1),
        }
    }

    pub fn allocate_blocks(&self, count: usize) -> Vec<MockMutableBlock> {
        (0..count)
            .map(|_| {
                let layout = LayoutHandle::from_u128(
                    self.next_layout.fetch_add(1, Ordering::Relaxed) as u128,
                );
                let block_id = self.next_block.fetch_add(1, Ordering::Relaxed) as BlockId;
                MockMutableBlock::new(layout, block_id)
            })
            .collect()
    }

    pub fn register_blocks(&self, blocks: Vec<MockMutableBlock>) -> Vec<MockImmutableBlock> {
        blocks
            .into_iter()
            .map(|b| MockImmutableBlock {
                layout_handle: b.layout_handle(),
                block_id: b.block_id(),
            })
            .collect()
    }
}

pub struct OffloadOperationArtifacts {
    manager: Arc<MockBlockManager>,
    mutable_blocks: Mutex<Option<Vec<MockMutableBlock>>>,
    operation_entry: OperationEntry,
}

impl OffloadOperationArtifacts {
    pub fn new(
        manager: Arc<MockBlockManager>,
        blocks: Vec<MockMutableBlock>,
        src_block_ids: &[BlockId],
    ) -> Self {
        let dst_block_ids = blocks.iter().map(|b| b.block_id()).collect::<Vec<_>>();
        let layout_handle = blocks
            .first()
            .map(|b| b.layout_handle())
            .unwrap_or_else(|| LayoutHandle::from_u128(0));

        let entry = OperationEntry {
            src_handle: LayoutHandle::from_u128(0),
            src_block_ids: src_block_ids.to_vec(),
            dst_handle: layout_handle,
            dst_block_ids,
        };

        Self {
            manager,
            mutable_blocks: Mutex::new(Some(blocks)),
            operation_entry: entry,
        }
    }
}

impl OperationArtifacts for OffloadOperationArtifacts {
    fn operation_entry(&self) -> OperationEntry {
        self.operation_entry.clone()
    }

    fn on_complete(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(blocks) = self.mutable_blocks.lock().unwrap().take() {
            let _ = self.manager.register_blocks(blocks);
        }
        Ok(())
    }
}

pub trait TransferEvaluator: Send + Sync + 'static {
    fn evaluate(
        &self,
        request_id: &str,
        candidate: TransferCandidateRecord,
    ) -> TransferEvaluationOutcome;
}

pub struct DefaultTransferEvaluator {
    block_manager: Arc<MockBlockManager>,
}

impl DefaultTransferEvaluator {
    pub fn new(block_manager: Arc<MockBlockManager>) -> Self {
        Self { block_manager }
    }

    fn evaluate_offload(&self, candidate: TransferCandidateRecord) -> TransferEvaluationOutcome {
        let planned = candidate.planned.clone();
        let tokens = match candidate.metadata {
            TransferCandidateMetadata::Offload { token_blocks } => token_blocks,
            _ => planned.token_blocks.clone(),
        };

        if planned.block_ids.is_empty() || tokens.is_empty() {
            return TransferEvaluationOutcome::Skip(TransferSkip {
                uuid: planned.transfer_id,
                reason: TransferSkipReason::AlreadyCachedHigherTier,
            });
        }

        let dst_blocks = self.block_manager.allocate_blocks(planned.block_ids.len());
        let artifacts = Arc::new(OffloadOperationArtifacts::new(
            self.block_manager.clone(),
            dst_blocks,
            &planned.block_ids,
        ));
        let entry = artifacts.operation_entry();

        let plan = TransferExecutionPlan::Offload(OffloadPlan {
            block_ids: planned.block_ids.clone(),
            token_blocks: tokens,
            dst_block_ids: entry.dst_block_ids.clone(),
        });

        let worker_request = WorkerTransferRequest {
            transfer_id: planned.transfer_id,
            src_layout: LayoutHandle::from_u128(0),
            src_block_ids: planned.block_ids.clone(),
            dst_layout: entry.dst_handle,
            dst_block_ids: entry.dst_block_ids.clone(),
            options: TransferOptions::default(),
        };

        TransferEvaluationOutcome::Promote(TransferPromotion {
            uuid: planned.transfer_id,
            worker_request,
            execution_plan: plan,
            operation_metadata: None,
            operation_artifacts: Some(artifacts),
            operation_entry: Some(entry),
        })
    }
}

impl TransferEvaluator for DefaultTransferEvaluator {
    fn evaluate(
        &self,
        _request_id: &str,
        candidate: TransferCandidateRecord,
    ) -> TransferEvaluationOutcome {
        match candidate.planned.direction {
            TransferDirection::DeviceToHost => self.evaluate_offload(candidate),
            TransferDirection::HostToDevice | TransferDirection::DiskToDevice => {
                TransferEvaluationOutcome::Skip(TransferSkip {
                    uuid: candidate.planned.transfer_id,
                    reason: TransferSkipReason::Other(
                        "onboarding not supported in default evaluator".to_string(),
                    ),
                })
            }
        }
    }
}

pub trait TransferSlotHandle: Send + Sync {
    fn apply_evaluation(&self, outcome: TransferEvaluationOutcome) -> Option<PromotedTransfer>;
    fn acknowledge(&self, uuid: Uuid);
}

impl TransferSlotHandle for Mutex<Slot> {
    fn apply_evaluation(&self, outcome: TransferEvaluationOutcome) -> Option<PromotedTransfer> {
        let mut slot = self.lock().unwrap();
        slot.apply_evaluation_outcome(outcome)
    }

    fn acknowledge(&self, uuid: Uuid) {
        let mut slot = self.lock().unwrap();
        slot.acknowledge_transfer(uuid);
    }
}

#[derive(Debug)]
pub enum TransferEngineCommand {
    Offload {
        uuid: Uuid,
        block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
        dst_block_ids: Vec<BlockId>,
        operation_entry: OperationEntry,
    },
    Onboard {
        uuid: Uuid,
        dst_block_ids: Vec<BlockId>,
        src_blocks: Vec<BlockId>,
        operation_entry: OperationEntry,
    },
    Noop {
        uuid: Uuid,
    },
}
