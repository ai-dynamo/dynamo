// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Connector Protocol
//!
//! This module defines the messages used to communicate between the following components:
//! - Leader -> TransferEngine (block_manager::distributed)
//! - TransferEngine -> Scheduler
//! - Worker -> Scheduler
//!
//! ## Locality
//!
//! The TransferEngine, Scheduler and Worker are all guaranteed to be in the same process. `Scheduler`
//! is a per-worker scheduler and `TransferEngine` is also a per-worker component.
//!
//! ## Connector Operations
//!
//! There a two types of connector operations: load operations and store operations. The following must
//! be true:
//! - All loads must be initiated when the Slot is in the [`SlotState::Initialized`] state.
//! - While the slot is in the [`SlotState::OnboardStaged`] or the [`SlotState::Onboarding`] state,
//!   no active tokens can be scheduled, no stores can be issued.
//!   - Uknowns:
//!     - What happens on cancellation?
//! - To transition to the [`SlotState::Prefilling`] state, the slot must be in either the [`SlotState::Initialized`]
//!   [`SlotState::NotScheduled`], or [`SlotState::OnboardStaged`] state.
//!   - When in the [`SlotState::Prefilling`] state, store/save operations are allowed.
//!   - Store/Save operations are determined when processing the [`SchedulerOutput`].
//!   - If a store operation is issued, the following will happen:
//!     - Leader will trigger a message to the TransferEngine with the use StoreRequest and a ConnectorStoreRequest
//!     - The presence of the ConnectorStoreRequest will trigger the TransferEngine to request a SchedulerStoreRequest,
//!       this will block the transfer engine's store task from executing until released by the scheduler.
//!     - The Scheduler will not release the store task until the Worker has made sufficient progress, i.e. the data is
//!       to be stored has been computed and in device memory.
//!     - All leader slots are visited on each build metadata step, this allows for any leader initiated actions to be
//!       included in the metadata sent to the worker.
//!       - An operation must include: request_id, the iteration on which it was issued, the operation type, and a descriptor.
//!     - The Worker will pick up all operations from the leader's metadata and enqueue to the scheduler.
//!     - The Worker will issue notifications to the Scheduler at the start of each iteration and the completion of each
//!       layer in that iteration.
//!     - For an operation to be scheduled to run, the following must be true:
//!       - The TransferEngine must have registered the operation with the Scheduler.
//!       - The Worker must have registered the operation with the Scheduler.
//!       - Sufficient progress, either layer-wise or iteration-wise, must have been made.
//!     - For an operation to run, the following must be true:
//!       - The operation must be in the scheduled queue.
//!       - A concurrent token must be acquired.
//!     - A running operation will be monitored by a task awaiting a completion event.
//!       - When the completion event is received, the atomic completion counter will be incremented.
//!
//!
//! All transfer requests are triggered by the leader based on the details in the [`SchedulerOutput`].
//!
//! [`SchedulerOutput`] is transform

use super::scheduler::{DISCONNECTED_WARNING, SchedulingDecision};
use super::*;

use tokio::sync::oneshot;

pub type LayerName = String;
pub type LayerIndex = u32;
pub type Iteration = u64;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum RequestType {
    /// If Scheduled, then the [`super::scheduler::TransferSchedulerClient`] will commuicate with the scheudler
    /// to await a boxed [`ScheduledTransferCompletionHandle`].
    Scheduled,

    /// If Immediate, then the [`super::scheduler::TransferSchedulerClient`] will immediately return a
    /// [`ImmediateTransferCompletionHandle`].
    Immediate,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum TransferType {
    Load,
    Store,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum SchedulerRequirement {
    IterationComplete(Iteration),

    /// The layer with the provided name and iteration counter must be complete.
    LayerNameComplete(LayerName, Iteration),

    /// The layer index and iteration counter must be complete.
    LayerComplete(LayerIndex, Iteration),
}

/// Issued by the leader, received by the TransferEngine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderTransferRequest {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub requirement: Option<SchedulerRequirement>,
    pub request_type: RequestType,
}

pub enum TransferToSchedulerMessage {
    ScheduleRequest(TransferScheduleRequest),
    ImmediateResult(ImmediateTransferResult),
}

/// Issued by the TransferEngine, received by the Scheduler.
/// Note: In order to be considered for scheduling, the [`TransferScheduleRequest`] and the [`WorkerTransferRequest`]
/// for the same operation (uuid) must be present on the scheduler.
pub struct TransferScheduleRequest {
    pub leader_request: LeaderTransferRequest,
    pub response_tx: oneshot::Sender<ScheduledTaskHandle>,
}

pub struct ScheduledTaskHandle {
    pub decision_rx: oneshot::Receiver<(SchedulingDecision, oneshot::Sender<anyhow::Result<()>>)>,
    pub cancel_token: CancellationToken,
}

impl ScheduledTaskHandle {
    pub async fn wait_for_decision(self) -> Box<dyn TransferCompletionHandle> {
        tokio::select! {
            Ok((decision, completion_tx)) = self.decision_rx => {
                Box::new(ScheduledTransferCompletionHandle::new(decision, completion_tx))
            }
            _ = self.cancel_token.cancelled() => {
                Box::new(CancelledTransferCompletionHandle)
            }
        }
    }
}

/// Recived by the Worker, forward to the Scheduler.
///
/// In ordered to be considered for scheduling, both the [`TransferScheduleRequest`] and the [`WorkerTransferRequest`]
/// must be present on the scheduler.
///
/// Note: No response is required. The Worker holds an atomic counter for each oepration type. The expected count (local/non-atomic)
/// is incremented on receiving a request. The Worker knows all operations are complete when the shared atomic counter matches the
/// expected count.
///
/// Workers can not handle errors, they only deal with counters. All operations (which can be cancelled) must completed for a Worker
/// to mark the request_id as complete.
///
/// Scheduler requirements are only provided by the leader initiated transfer request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTransferRequest {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub transfer_type: TransferType,
    pub request_type: RequestType,
}

/// Sent by Worker to Scheduler.
/// Combines [`WorkerTransferRequest`] and [`WorkerRequestState`] and issues a [`WorkerSchedulerRequest`]
///
/// This object has all the links to the worker to track completion and observe any cancellation signals.
pub struct WorkerSchedulerRequest {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub transfer_type: TransferType,
    pub cancel_token: CancellationToken,
}

/// One-time use object returned from [`Scheduler::schedule_transfer`]
/// This object carries with it the [`SchedulingDecision`] and is used to mark the transfer as complete.
#[async_trait::async_trait]
pub trait TransferCompletionHandle: Send {
    fn scheduler_decision(&self) -> SchedulingDecision;
    async fn mark_complete(&self, result: anyhow::Result<()>);
}

pub struct ScheduledTransferCompletionHandle {
    scheduler_decision: SchedulingDecision,
    completion_tx: Mutex<Option<oneshot::Sender<anyhow::Result<()>>>>,
}

impl ScheduledTransferCompletionHandle {
    pub(crate) fn new(
        scheduler_decision: SchedulingDecision,
        completion_tx: oneshot::Sender<anyhow::Result<()>>,
    ) -> Self {
        Self {
            scheduler_decision,
            completion_tx: Mutex::new(Some(completion_tx)),
        }
    }
}

#[async_trait::async_trait]
impl TransferCompletionHandle for ScheduledTransferCompletionHandle {
    fn scheduler_decision(&self) -> SchedulingDecision {
        self.scheduler_decision
    }

    async fn mark_complete(&self, result: anyhow::Result<()>) {
        if let Some(completion_tx) = self.completion_tx.lock().unwrap().take()
            && completion_tx.send(result).is_err()
        {
            tracing::error!(
                "failed to send completion status; this could lead to silent data corruption"
            );
        }
    }
}

impl Drop for ScheduledTransferCompletionHandle {
    fn drop(&mut self) {
        if self.completion_tx.lock().unwrap().is_some() {
            // This is a fundamental logic error. The results of the application are undefined.
            // We must abort.
            panic!(concat!(
                "logic error: implementation failed to respect the [TransferCompletionHandle] policy; ",
                "handle dropped without being explicitly marked; this may lead to data corruption if ",
                "the handle was dropped while a transfer was still in progress; please report immediately.",
            ));
        }
    }
}

pub struct ImmediateTransferResult {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub status: anyhow::Result<()>,
}

pub struct ImmediateTransferCompletionHandle {
    request_id: String,
    uuid: uuid::Uuid,
    completion_tx: Mutex<Option<tokio::sync::mpsc::Sender<TransferToSchedulerMessage>>>,
}

impl ImmediateTransferCompletionHandle {
    pub(crate) fn new(
        request_id: String,
        uuid: uuid::Uuid,
        completion_tx: tokio::sync::mpsc::Sender<TransferToSchedulerMessage>,
    ) -> Self {
        Self {
            request_id,
            uuid,
            completion_tx: Mutex::new(Some(completion_tx)),
        }
    }
}

#[async_trait::async_trait]
impl TransferCompletionHandle for ImmediateTransferCompletionHandle {
    fn scheduler_decision(&self) -> SchedulingDecision {
        SchedulingDecision::Execute
    }

    async fn mark_complete(&self, result: anyhow::Result<()>) {
        // To ensure the future is Send, avoid holding the MutexGuard across .await.
        let completion_tx = {
            let mut guard = self.completion_tx.lock().unwrap();
            guard.take()
        };
        if let Some(completion_tx) = completion_tx
            && completion_tx
                .send(TransferToSchedulerMessage::ImmediateResult(
                    ImmediateTransferResult {
                        request_id: self.request_id.clone(),
                        uuid: self.uuid,
                        status: result,
                    },
                ))
                .await
                .is_err()
        {
            tracing::error!(DISCONNECTED_WARNING);
        }
    }
}

impl Drop for ImmediateTransferCompletionHandle {
    fn drop(&mut self) {
        if self.completion_tx.lock().unwrap().is_some() {
            // This is a fundamental logic error. The results of the application are undefined.
            // We must abort.
            panic!(concat!(
                "logic error: implementation failed to respect the [TransferCompletionHandle] policy; ",
                "handle dropped without being explicitly marked; this may lead to data corruption if ",
                "the handle was dropped while a transfer was still in progress; please report immediately.",
            ));
        }
    }
}

pub struct CancelledTransferCompletionHandle;

#[async_trait::async_trait]
impl TransferCompletionHandle for CancelledTransferCompletionHandle {
    fn scheduler_decision(&self) -> SchedulingDecision {
        SchedulingDecision::Cancel
    }

    async fn mark_complete(&self, _result: anyhow::Result<()>) {
        // Do nothing
    }
}
