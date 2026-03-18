// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use super::protocol::*;
use super::*;

use tokio::sync::mpsc;

pub const DISCONNECTED_WARNING: &str =
    "runtime error: connections between components were lost; likely tearing down";

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("runtime error: connections between components were lost; likely tearing down")]
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulingDecision {
    Execute,
    Cancel,
}

/// A client for the scheduler. One-time use. Capture a clone per task.
#[derive(Clone)]
pub struct TransferSchedulerClient {
    scheduler_tx: mpsc::Sender<TransferToSchedulerMessage>,
}

impl TransferSchedulerClient {
    pub fn new(scheduler_tx: mpsc::Sender<TransferToSchedulerMessage>) -> Self {
        Self { scheduler_tx }
    }

    /// If the [SchedulingDecision::Execute] is returned, the caller receives a completion handle.
    /// The completion handle be marked as completed after the
    ///
    /// If the [SchedulingDecision::Cancel] is returned, the transfer is cancelled and the completion handle
    /// must not be dropped.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.request_id, operation_id = %request.uuid))]
    pub async fn schedule_transfer(
        self,
        request: LeaderTransferRequest,
    ) -> anyhow::Result<Box<dyn TransferCompletionHandle>> {
        let scheduler_tx = self.scheduler_tx.clone();
        match request.request_type {
            RequestType::Immediate => {
                let handle = ImmediateTransferCompletionHandle::new(
                    request.request_id,
                    request.uuid,
                    scheduler_tx.clone(),
                );
                Ok(Box::new(handle))
            }
            RequestType::Scheduled => {
                let (response_tx, response_rx) = oneshot::channel();
                let request = TransferScheduleRequest {
                    leader_request: request,
                    response_tx,
                };

                tracing::debug!("sending schedule request to scheduler");
                scheduler_tx
                    .send(TransferToSchedulerMessage::ScheduleRequest(request))
                    .await?;

                tracing::debug!("awaiting response from scheduler");
                let handle = response_rx.await?.wait_for_decision().await;

                tracing::debug!(
                    "received scheduler decision: {:?}",
                    handle.scheduler_decision()
                );
                Ok(handle)
            }
        }
    }
}

pub struct WorkerSchedulerClient {
    slots: HashMap<String, WorkerSchedulerClientSlot>,
    scheduler_tx: mpsc::UnboundedSender<SchedulerMessage>,
    iteration: u64,
    iteration_complete: bool,
    layers_complete: u32,
}

impl WorkerSchedulerClient {
    pub fn new(
        scheduler_tx: mpsc::UnboundedSender<SchedulerMessage>,
        _cancel_token: CancellationToken,
    ) -> Self {
        Self {
            slots: HashMap::new(),
            scheduler_tx,
            iteration: 0,
            iteration_complete: true,
            layers_complete: 0,
        }
    }

    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    pub fn start_next_iteration(&mut self) -> Result<(), SchedulerError> {
        // debug_assert!(
        //     self.iteration_complete,
        //     "previous iteration must be complete before starting a new iteration"
        // );
        self.iteration += 1;
        self.iteration_complete = false;
        self.layers_complete = 0;
        self.scheduler_tx
            .send(SchedulerMessage::StartIteration(self.iteration))
            .map_err(|_| SchedulerError::Disconnected)
    }

    pub fn mark_layer_complete(&mut self, layer_name: String) -> Result<(), SchedulerError> {
        debug_assert!(
            !self.iteration_complete,
            "iteration must be complete before marking a layer as complete"
        );
        self.layers_complete += 1;
        self.scheduler_tx
            .send(SchedulerMessage::UpdateLayersCompleted(
                layer_name,
                self.layers_complete,
            ))
            .map_err(|_| SchedulerError::Disconnected)
    }

    pub fn mark_iteration_complete(&mut self) -> Result<(), SchedulerError> {
        debug_assert!(
            !self.iteration_complete,
            "iteration must be complete before marking it as complete"
        );
        self.iteration_complete = true;
        self.scheduler_tx
            .send(SchedulerMessage::EndIteration(self.iteration))
            .map_err(|_| SchedulerError::Disconnected)
    }
}

#[derive(Debug, Default)]
pub struct WorkerSchedulerClientSlot {
    operations: Vec<uuid::Uuid>,
    completed: Arc<AtomicU64>,
}

impl WorkerSchedulerClientSlot {
    fn new() -> Self {
        Self {
            operations: Vec::new(),
            completed: Arc::new(AtomicU64::new(0)),
        }
    }

    fn make_scheduler_slot_request(
        &self,
        request_id: String,
        expected_immediate_ops: u64,
    ) -> SchedulerCreateSlotDetails {
        SchedulerCreateSlotDetails {
            request_id,
            completed: self.completed.clone(),
            expected_immediate_ops,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Relaxed) == self.operations.len() as u64
    }
}

impl WorkerSchedulerClient {
    /// Create a slot with the expected number of immediate (onboard) operations.
    /// This count is used to properly track completion and must match the number of
    /// ImmediateTransferResult messages that will be received.
    pub fn create_slot_with_immediate_ops(
        &mut self,
        request_id: String,
        expected_immediate_ops: u64,
    ) -> Result<(), SchedulerError> {
        // create a request slot
        let slot = WorkerSchedulerClientSlot::new();
        let request = slot.make_scheduler_slot_request(request_id.clone(), expected_immediate_ops);

        // insert the slot into the local worker slots map
        self.slots.insert(request_id.clone(), slot);

        // send a request to insert the slot into the engine state
        self.scheduler_tx
            .send(SchedulerMessage::CreateSlot(request))
            .map_err(|_| SchedulerError::Disconnected)?;
        Ok(())
    }

    /// Create a slot with no expected immediate operations (backward compatibility).
    pub fn create_slot(&mut self, request_id: String) -> Result<(), SchedulerError> {
        self.create_slot_with_immediate_ops(request_id, 0)
    }

    pub fn remove_slot(&mut self, request_id: &String) {
        let slot = self.slots.remove(request_id).expect("slot does not exist");
        assert!(slot.is_complete());
        self.scheduler_tx
            .send(SchedulerMessage::RequestFinished(request_id.clone()))
            .expect("failed to send request finished message; disconnected");
    }

    /// Enqueues a request to the scheduler.
    ///
    /// Both the worker client and the scheduler keep track of outstanding requests.
    /// The atomic counter to mark completion is shared, but only incremented by the scheduler.
    pub fn enqueue_request(&mut self, request: WorkerTransferRequest) {
        debug_assert!(
            self.slots.contains_key(&request.request_id),
            "slot does not exist"
        );

        let slot = self
            .slots
            .get_mut(&request.request_id)
            .expect("slot does not exist");

        slot.operations.push(request.uuid);

        match request.request_type {
            RequestType::Immediate => {}
            RequestType::Scheduled => {
                self.scheduler_tx
                    .send(SchedulerMessage::EnqueueRequest(request))
                    .expect("failed to enqueue request; disconnected");
            }
        }
    }

    pub fn has_slot(&self, request_id: &str) -> bool {
        self.slots.contains_key(request_id)
    }

    pub fn is_complete(&self, request_id: &str) -> bool {
        match self.slots.get(request_id) {
            Some(slot) => slot.is_complete(),
            None => true,
        }
    }

    /// Clone the scheduler channel for async use.
    pub fn get_scheduler_tx(&self) -> mpsc::UnboundedSender<SchedulerMessage> {
        self.scheduler_tx.clone()
    }

    /// Record operation in slot (bookkeeping only, no send).
    /// This updates the slot's expected operation count so is_complete() works correctly.
    pub fn record_operation(&mut self, request_id: &str, uuid: uuid::Uuid) {
        let slot = self.slots.get_mut(request_id).expect("slot does not exist");
        slot.operations.push(uuid);
    }
}

pub type Iteration = u64;
pub type LayerName = String;
pub type LayerIndex = u32;

pub enum SchedulerMessage {
    /// Issued by worker to create a shared request state between worker and scheduler
    CreateSlot(SchedulerCreateSlotDetails),

    /// Enqueue a worker requested operation to the scheduler, this is one-half of the necessary
    /// bits to enqueu the operation. The other half is leader driven and propagated to the scheduler
    /// via the [TransferScheduleRequest]
    EnqueueRequest(WorkerTransferRequest),

    /// Issued at the start of a forward pass iteration
    StartIteration(Iteration),

    /// Issued at the end of a forward pass iteration, with the iteration number
    EndIteration(Iteration),

    /// Issued by the leader to update the number of layers completed
    UpdateLayersCompleted(LayerName, LayerIndex),

    /// Worker received a notification that the given request id has been completed.
    RequestFinished(String),
}

pub struct Scheduler {
    // Created by Worker
    slots: HashMap<String, SchedulerSlot>,

    // Created during the responses to a scheduled transfer request
    // Note: this does not require a slot to exist yet
    cancel_tokens: HashMap<String, CancellationToken>,

    // Created by immediately scheduled transfers completing and returning their completion
    // signals to the scheduler.
    // Note: this does not require a slot to exist yet
    unprocessed_immediate_results: HashMap<String, HashSet<uuid::Uuid>>,

    // This object coordinates the two-stage execution of a scheduled transfer request.
    // If the scheduled request arrives first, the controller object will be Some; otherwise,
    // the worker-side request arrived first and it will be None.
    enqueued_requests: HashMap<String, HashMap<uuid::Uuid, TransferRequestSource>>,

    // Messages from the worker arrive on this channel
    worker_rx: mpsc::UnboundedReceiver<SchedulerMessage>,

    // Messages from the transfer client arrive on this channel
    transfer_rx: mpsc::Receiver<TransferToSchedulerMessage>,
    iteration: u64,
    layers_complete: u32,
    iteration_complete: bool,
}

impl Scheduler {
    pub fn new(
        cancel_token: CancellationToken,
    ) -> (Self, WorkerSchedulerClient, TransferSchedulerClient) {
        let (scheduler_tx, scheduler_rx) = mpsc::unbounded_channel();
        let (transfer_tx, transfer_rx) = mpsc::channel(128);
        let worker_client = WorkerSchedulerClient::new(scheduler_tx, cancel_token);
        let transfer_client = TransferSchedulerClient::new(transfer_tx);
        (
            Scheduler {
                slots: HashMap::new(),
                cancel_tokens: HashMap::new(),
                unprocessed_immediate_results: HashMap::new(),
                enqueued_requests: HashMap::new(),
                worker_rx: scheduler_rx,
                transfer_rx,
                iteration: 0,
                layers_complete: 0,
                iteration_complete: true,
            },
            worker_client,
            transfer_client,
        )
    }

    pub async fn run(&mut self) -> anyhow::Result<()> {
        loop {
            if !self.step().await {
                break;
            }
        }
        Ok(())
    }

    async fn step(&mut self) -> bool {
        if self.worker_rx.is_closed() || self.transfer_rx.is_closed() {
            return false;
        }

        tokio::select! {
            maybe_worker_msg = self.worker_rx.recv(), if !self.worker_rx.is_closed() => {
                match maybe_worker_msg {
                    Some(SchedulerMessage::StartIteration(new_iteration)) => {
                        self.start_iteration(new_iteration);
                    }
                    Some(SchedulerMessage::EndIteration(iteration)) => {
                        self.end_iteration(iteration);
                    }
                    Some(SchedulerMessage::UpdateLayersCompleted(last_layer_name, layers_completed)) => {
                        self.update_layers_completed(last_layer_name, layers_completed);
                    }
                    Some(SchedulerMessage::CreateSlot(request)) => {
                        self.add_slot(request);
                    }
                    Some(SchedulerMessage::RequestFinished(request_id)) => {
                        self.remove_slot(request_id);
                    }
                    Some(SchedulerMessage::EnqueueRequest(request)) => {
                        self.handle_worker_request(request);
                    }
                    None => {
                        return false;
                    }
                }
            }
            maybe_transfer_msg = self.transfer_rx.recv(), if !self.transfer_rx.is_closed() => {
                match maybe_transfer_msg {
                    Some(TransferToSchedulerMessage::ScheduleRequest(request)) => {
                        self.handle_scheduled_transfer_request(request);
                    }
                    Some(TransferToSchedulerMessage::ImmediateResult(result)) => {
                        self.handle_immediate_result(result);
                    }
                    None => {
                        return false;
                    }
                }
             }
        }
        true
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %req.request_id))]
    fn add_slot(&mut self, req: SchedulerCreateSlotDetails) {
        let request_id = req.request_id.clone();

        // In TP>1, multiple workers send CreateSlot for the same request_id.
        // ImmediateTransferResults can arrive before ANY worker's slot is created.
        //
        // We need to apply the buffered count to EVERY worker's slot, not just the first one.
        // Use `get` instead of `remove` to keep the buffered results available for all workers.
        // The buffered results will be cleared when the request is removed (finished).

        let slot = SchedulerSlot {
            completed: req.completed,
        };

        // Check for buffered ImmediateTransferResults that arrived before the slot was created.
        // Apply buffered count to this worker's slot.
        if let Some(buffered_results) = self.unprocessed_immediate_results.get(&request_id) {
            let num_buffered = buffered_results.len() as u64;

            // Sanity check: buffered results should never exceed expected count.
            // If this happens, there's a mismatch between leader's count and actual results.
            debug_assert!(
                num_buffered <= req.expected_immediate_ops,
                "buffered results ({}) exceed expected immediate ops ({})",
                num_buffered,
                req.expected_immediate_ops
            );

            // Use num_buffered (not expected_immediate_ops) because we only mark operations
            // as complete that have actually completed. Remaining results will arrive later
            // via handle_immediate_result() and increment the counter then.
            slot.completed.fetch_add(num_buffered, Ordering::Relaxed);
        }

        self.slots.insert(request_id, slot);
    }

    fn remove_slot(&mut self, request_id: String) {
        debug_assert!(self.slots.contains_key(&request_id), "slot not found");
        self.cancel_tokens.remove(&request_id);
        self.slots.remove(&request_id);

        let maybe_controller = self.enqueued_requests.remove(&request_id);
        debug_assert!(
            maybe_controller.is_none() || maybe_controller.unwrap().is_empty(),
            "any scheduled request should be removed and enqueued/scheduled before the slot is removed"
        );

        // In TP>1, buffered results are NOT removed in add_slot (they're applied to ALL workers).
        // Clean them up here when the request is finished.
        self.unprocessed_immediate_results.remove(&request_id);

        tracing::debug!(
            request_id,
            iteration = self.iteration,
            "engine state removing slot"
        );
    }

    fn handle_worker_request(&mut self, request: WorkerTransferRequest) {
        debug_assert!(
            self.slots.contains_key(&request.request_id),
            "slot does not exist"
        );

        let maybe_controller = self.try_prepare_controller(
            request.request_id,
            request.uuid,
            TransferRequestSource::Worker,
        );

        if let Some(controller) = maybe_controller {
            self.schedule_request(controller);
        }
    }

    fn start_iteration(&mut self, iteration: u64) {
        // tracing::debug!(iteration, "engine state updating iteration");
        // debug_assert!(
        //     self.iteration_complete,
        //     "previous iteration must be complete before starting a new iteration"
        // );
        debug_assert_eq!(
            self.iteration,
            iteration - 1,
            "iteration must be incremented by 1"
        );
        self.iteration = iteration;
        self.layers_complete = 0;
        self.iteration_complete = false;
    }

    fn end_iteration(&mut self, iteration: u64) {
        tracing::debug!(iteration, "engine state updating iteration");
        self.iteration_complete = true;
    }

    fn update_layers_completed(&mut self, last_layer_name: String, layers_completed: u32) {
        self.layers_complete = layers_completed;
        tracing::debug!(
            iteration = self.iteration,
            layers_completed,
            "layer {last_layer_name} is complete"
        );
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %result.request_id, operation_id = %result.uuid))]
    fn handle_immediate_result(&mut self, result: ImmediateTransferResult) {
        match self.slots.get_mut(&result.request_id) {
            Some(slot) => {
                slot.completed.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    "matched slot; incrementing completed counter to {}",
                    slot.completed.load(Ordering::Relaxed)
                );
            }
            None => {
                tracing::debug!("no slot found; adding to unprocessed immediate results");
                self.unprocessed_immediate_results
                    .entry(result.request_id)
                    .or_default()
                    .insert(result.uuid);
            }
        }
    }

    /// This function is used to handle the request from worker or transfer based on their arrival order.
    /// It returns Some(ScheduledTaskController) if both worker and transfer have arrived, or None if any of them has not arrived yet.
    ///
    /// More details:
    /// If no uuid is found in enqueued_requests, it means neither worker nor transfer has arrived yet.
    /// Then, we will insert controller into enqueued_requests (for transfer) or None (for worker) and return None.
    ///
    /// If uuid is found in enqueued_requests, it means either worker or transfer has arrived.
    /// Then, we check the incoming controller. If it is Some, it means worker has arrived first and we can return it.
    /// If it is None, it means the transfer has arrived first and we can return the existing controller.
    fn try_prepare_controller(
        &mut self,
        request_id: String,
        uuid: uuid::Uuid,
        incoming: TransferRequestSource,
    ) -> Option<ScheduledTaskController> {
        let entry = self.enqueued_requests.entry(request_id).or_default();
        match (entry.remove(&uuid), incoming) {
            (Some(TransferRequestSource::Worker), TransferRequestSource::Transfer(controller)) => {
                tracing::debug!("worker arrived first, then transfer ==> scheduling transfer");
                Some(controller)
            }
            (Some(TransferRequestSource::Transfer(controller)), TransferRequestSource::Worker) => {
                tracing::debug!("transfer arrived first, then worker ==> scheduling transfer");
                Some(controller)
            }
            (None, TransferRequestSource::Worker) => {
                tracing::debug!("worker arrived first; must wait for transfer");
                entry.insert(uuid, TransferRequestSource::Worker);
                None
            }
            (None, TransferRequestSource::Transfer(controller)) => {
                tracing::debug!("transfer arrived first; must wait for worker");
                entry.insert(uuid, TransferRequestSource::Transfer(controller));
                None
            }
            _ => {
                panic!("invalid combination of request sources");
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.leader_request.request_id))]
    fn handle_scheduled_transfer_request(&mut self, request: TransferScheduleRequest) {
        let controller = self.process_scheduled_transfer_request(request).unwrap();

        let maybe_controller = self.try_prepare_controller(
            controller.request.request_id.clone(),
            controller.request.uuid,
            TransferRequestSource::Transfer(controller),
        );

        if let Some(controller) = maybe_controller {
            tracing::debug!("scheduling transfer");
            self.schedule_request(controller);
        }
    }

    // this function will be a scheduler and will dispatch requests to be executed
    fn schedule_request(&mut self, xfer_req: ScheduledTaskController) {
        // tokio spawn execute_scheduled_transfer for first impl.  add fanciness later.
        self.execute_scheduled_transfer(xfer_req);
    }

    // this function will execute a transfer request, monitor its completion, and increment its
    // atomic completion counter when finished.
    //
    // this must tokio spawn and an indpendent task
    fn execute_scheduled_transfer(&mut self, xfer_req: ScheduledTaskController) {
        debug_assert!(
            self.slots.contains_key(&xfer_req.request.request_id),
            "slot not found"
        );
        let completed = self
            .slots
            .get(&xfer_req.request.request_id)
            .unwrap()
            .completed
            .clone();
        tokio::spawn(xfer_req.execute(SchedulingDecision::Execute, completed));
    }

    /// Translate the [`TransferScheduleRequest`] into a local [`ScheduledTaskController`]
    /// This function returns to the transfer client the [`ScheduledTaskHandle`]
    fn process_scheduled_transfer_request(
        &mut self,
        xfer_req: TransferScheduleRequest,
    ) -> anyhow::Result<ScheduledTaskController> {
        // Create the next stage communcication p2p channel between scheduler and client
        let (decision_tx, decision_rx) = oneshot::channel();

        // Get or create the cancel token for this request
        let cancel_token = self
            .cancel_tokens
            .entry(xfer_req.leader_request.request_id.clone())
            .or_default()
            .child_token();

        // Create the ScheduledTaskHandle to send to the client
        let task_handle = ScheduledTaskHandle {
            decision_rx,
            cancel_token,
        };

        // Send the ScheduledTaskHandle back to the client side
        xfer_req
            .response_tx
            .send(task_handle)
            .map_err(|_| anyhow::anyhow!("Failed to send scheduled task handle to xfer client"))?;

        // Create the ScheduledTaskController to locally trigger the exection of the scheduled transfer task
        let controller = ScheduledTaskController {
            request: xfer_req.leader_request,
            decision_tx,
        };

        Ok(controller)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScheduledTaskError {}

pub struct ScheduledTaskController {
    request: LeaderTransferRequest,
    decision_tx: oneshot::Sender<(SchedulingDecision, oneshot::Sender<anyhow::Result<()>>)>,
}

impl ScheduledTaskController {
    pub async fn execute(
        self,
        decision: SchedulingDecision,
        completed: Arc<AtomicU64>,
    ) -> anyhow::Result<()> {
        let (completion_tx, completion_rx) = oneshot::channel();
        self.decision_tx
            .send((decision, completion_tx))
            .map_err(|_| anyhow::anyhow!(DISCONNECTED_WARNING))?;
        let _ = completion_rx
            .await
            .map_err(|_| anyhow::anyhow!(DISCONNECTED_WARNING))?;
        completed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

enum TransferRequestSource {
    Worker,
    Transfer(ScheduledTaskController),
}

pub struct ScheduledTaskAsyncResult {
    completion_rx: oneshot::Receiver<anyhow::Result<()>>,
}

impl ScheduledTaskAsyncResult {
    pub async fn await_completion(self) -> anyhow::Result<()> {
        self.completion_rx.await.unwrap()
    }
}

pub struct SchedulerCreateSlotDetails {
    pub request_id: String,
    pub completed: Arc<AtomicU64>,
    /// Expected number of immediate (onboard) operations for this slot.
    pub expected_immediate_ops: u64,
}

pub struct SchedulerSlot {
    completed: Arc<AtomicU64>,
}

pub trait TaskScheduler {
    fn start_iteration(&mut self, iteration: u64) -> Result<(), SchedulerError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduler_lifecycle() {
        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, _transfer_client) = Scheduler::new(cancel_token);

        // create a slot
        worker_client.create_slot("test".to_string()).unwrap();

        // enqueue a request
        assert!(!scheduler.slots.contains_key("test"));
        scheduler.step().await;
        assert!(scheduler.slots.contains_key("test"));

        // test iteration triggers
        worker_client.start_next_iteration().unwrap();
        scheduler.step().await;
        assert_eq!(scheduler.iteration, 1);

        // test iteration end triggers
        worker_client.mark_iteration_complete().unwrap();
        scheduler.step().await;
        assert_eq!(scheduler.iteration, 1);
        assert!(scheduler.iteration_complete);
    }

    #[tokio::test]
    async fn test_transfer_immediate_arrives_first() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) = Scheduler::new(cancel_token);

        let operation_id = uuid::Uuid::new_v4();

        // on the transfer engine, a request arrives with a request type of immediate
        let request = LeaderTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
        };

        let handle = transfer_client
            .clone()
            .schedule_transfer(request)
            .await
            .unwrap();

        // the transfer engine will immediately return a completion handle
        assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

        // the completion handle will be marked as complete
        handle.mark_complete(Ok(())).await;

        assert_eq!(scheduler.unprocessed_immediate_results.len(), 0);
        scheduler.step().await;
        assert_eq!(scheduler.unprocessed_immediate_results.len(), 1);

        // the request is completed - create slot with expected_immediate_ops=1
        worker_client
            .create_slot_with_immediate_ops("test".to_string(), 1)
            .unwrap();

        assert!(!scheduler.slots.contains_key("test"));
        scheduler.step().await;
        assert!(scheduler.slots.contains_key("test"));

        // Buffered results are not removed in add_slot() - cleanup happens in remove_slot()
        // when the request finishes. This ensures all workers in TP>1 can have the buffered
        // count applied. The buffered count has already been applied to the slot's completed counter.
        assert_eq!(scheduler.unprocessed_immediate_results.len(), 1);

        // neither the worker nor the scheduler should have observed the completion yet
        // this is because the worker has not yet requested it
        assert_eq!(
            scheduler
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            worker_client
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // the worker has not issued any operations yet
        assert_eq!(worker_client.slots.get("test").unwrap().operations.len(), 0);

        // enqueue the operation so is_complete() will return true (completed=1, operations.len()=1)
        let worker_request = WorkerTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
        };
        worker_client.enqueue_request(worker_request);
        assert_eq!(worker_client.slots.get("test").unwrap().operations.len(), 1);
        assert!(worker_client.is_complete("test"));

        // verify that remove_slot() cleans up the buffered results
        assert_eq!(scheduler.unprocessed_immediate_results.len(), 1);
        worker_client.remove_slot(&"test".to_string());
        scheduler.step().await;

        // after remove_slot(), the buffered results should be cleaned up
        assert_eq!(scheduler.unprocessed_immediate_results.len(), 0);
        assert!(!scheduler.slots.contains_key("test"));
    }

    /// This test verifies that the scheduler can handle the case where the transfer engine's
    /// immediate result arrives after the worker has scheduled the operation.
    #[tokio::test]
    async fn test_transfer_immediate_arrives_last() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) = Scheduler::new(cancel_token);

        let operation_id = uuid::Uuid::new_v4();

        // on the transfer engine, a request arrives with a request type of immediate
        let request = LeaderTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
        };

        let handle = transfer_client
            .clone()
            .schedule_transfer(request)
            .await
            .unwrap();

        // the transfer engine will immediately return a completion handle
        assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

        // assume this is a long running operation so our worker can enqueue the operation worker-side before the transfer-side completes
        worker_client.create_slot("test".to_string()).unwrap();
        assert!(!scheduler.slots.contains_key("test"));
        scheduler.step().await;
        assert!(scheduler.slots.contains_key("test"));
        assert_eq!(scheduler.unprocessed_immediate_results.len(), 0);

        // the worker enqueues the operation
        let request = WorkerTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
        };

        // immediate requests are not passed to the scheduler, but the completion will be automatically
        // visible on the client via the shared atomic counter
        worker_client.enqueue_request(request);

        let worker_slot = worker_client.slots.get("test").unwrap();
        assert_eq!(worker_slot.operations.len(), 1);
        assert_eq!(worker_slot.completed.load(Ordering::Relaxed), 0);

        // the completion handle will be marked as complete
        handle.mark_complete(Ok(())).await;

        assert_eq!(scheduler.unprocessed_immediate_results.len(), 0);
        scheduler.step().await;
        assert_eq!(scheduler.unprocessed_immediate_results.len(), 0);

        // neither the worker nor the scheduler should have observed the completion yet
        // this is because the worker has not yet requested it
        assert_eq!(
            scheduler
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            worker_client
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // the worker has not issued any operations yet
        assert_eq!(worker_client.slots.get("test").unwrap().operations.len(), 1);
    }

    // this test verifies that the scheduler can handle the case where the transfer engine's   /// in this case, the request arrives first via the worker client, meaning it traverse
    #[tokio::test]
    async fn test_transfer_scheduled_arrives_first() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) = Scheduler::new(cancel_token);

        let operation_id = uuid::Uuid::new_v4();

        // on the transfer engine, a request arrives with a request type of scheduled
        let request = LeaderTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
        };

        // transfer arrives first
        let handle = tokio::spawn(transfer_client.schedule_transfer(request));
        scheduler.step().await;

        // enqueued_requests should contain <request id, <uuid, and Some(controller)>> since transfer arrived first
        assert_eq!(scheduler.enqueued_requests.get("test").unwrap().len(), 1);
        assert!(matches!(
            scheduler
                .enqueued_requests
                .get("test")
                .unwrap()
                .get(&operation_id),
            Some(TransferRequestSource::Transfer(_))
        ));

        worker_client.create_slot("test".to_string()).unwrap();
        assert!(!scheduler.slots.contains_key("test"));
        scheduler.step().await;
        assert!(scheduler.slots.contains_key("test"));

        let request = WorkerTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            transfer_type: TransferType::Store,
            request_type: RequestType::Scheduled,
        };

        // worker arrives last
        worker_client.enqueue_request(request);
        scheduler.step().await;

        let handle = handle.await.unwrap().unwrap();
        handle.mark_complete(Ok(())).await;

        // after worker arrives, <uuid, and Some(controller)> inserted by transfer should be removed from enqueued_requests
        assert_eq!(scheduler.enqueued_requests.get("test").unwrap().len(), 0);

        // wait a bit to make sure the scheduled transfer to complete
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert_eq!(
            worker_client
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            scheduler
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // make sure all operations are complete
        assert!(worker_client.slots.get("test").unwrap().is_complete());
    }

    #[tokio::test]
    async fn test_transfer_scheduled_arrives_last() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) = Scheduler::new(cancel_token);

        let operation_id = uuid::Uuid::new_v4();

        worker_client.create_slot("test".to_string()).unwrap();
        assert!(!scheduler.slots.contains_key("test"));
        scheduler.step().await;
        assert!(scheduler.slots.contains_key("test"));

        let request = WorkerTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            transfer_type: TransferType::Store,
            request_type: RequestType::Scheduled,
        };

        // worker arrives first
        worker_client.enqueue_request(request);
        scheduler.step().await;

        // enqueued_requests should contain <request id, <uuid, and None>> since worker arrived first
        assert_eq!(scheduler.enqueued_requests.get("test").unwrap().len(), 1);
        assert!(matches!(
            scheduler
                .enqueued_requests
                .get("test")
                .unwrap()
                .get(&operation_id),
            Some(TransferRequestSource::Worker)
        ));

        let request = LeaderTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
        };

        // transfer arrives last
        let handle = tokio::spawn(transfer_client.schedule_transfer(request));
        scheduler.step().await;
        let handle = handle.await.unwrap().unwrap();
        assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);
        handle.mark_complete(Ok(())).await;

        // after transfer arrives, <uuid, and None> inserted by worker should be removed from enqueued_requests
        assert_eq!(scheduler.enqueued_requests.get("test").unwrap().len(), 0);

        // wait a bit to make sure the scheduled transfer to complete
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert_eq!(
            worker_client
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            scheduler
                .slots
                .get("test")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // make sure all operations are complete
        assert!(worker_client.slots.get("test").unwrap().is_complete());
    }

    #[tokio::test]
    async fn test_coordinate_scheduled_transfer_execution() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, _worker_client, transfer_client) = Scheduler::new(cancel_token);

        let operation_id = uuid::Uuid::new_v4();

        // Create a scheduled transfer request
        let request = LeaderTransferRequest {
            request_id: "test".to_string(),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
        };

        // allows us to pause the transfer task after the scheduler decision is made
        // but before the transfer is marked as complete
        let (got_handle_tx, got_handle_rx) = oneshot::channel();

        // Spawn the schedule_transfer call which will await our coordination function
        let _transfer_task = tokio::spawn(async move {
            let handle = transfer_client
                .clone()
                .schedule_transfer(request)
                .await
                .unwrap();

            got_handle_tx
                .send(handle)
                .map_err(|_| {
                    anyhow::anyhow!("failed to send handle back on testing oneshot channel")
                })
                .unwrap();
        });

        assert!(got_handle_rx.is_empty());

        // Simulate the scheduler making a decision and coordinating the execution
        // We skip that logic and go straight to the point we have a controller
        let controller = match scheduler.transfer_rx.recv().await {
            Some(msg) => match msg {
                TransferToSchedulerMessage::ScheduleRequest(schedule_req) => scheduler
                    .process_scheduled_transfer_request(schedule_req)
                    .ok(),
                _ => {
                    unreachable!("unexpected message type");
                }
            },
            None => {
                unreachable!("channel closed");
            }
        };

        // we still do not have both sides
        // we have the scheduler side controller, but we must trigger the controller to get a handle on the transfer engine
        let scheduler_controller = controller.expect("Expected a controller from the scheduler");
        assert!(got_handle_rx.is_empty());

        // Simulate some work being done - wait until the test releases us
        let completed = Arc::new(AtomicU64::new(0));
        let scheduler_result = tokio::spawn(
            scheduler_controller.execute(SchedulingDecision::Execute, completed.clone()),
        );

        // simulate the transfer engine receiving the decision
        let transfer_handle = got_handle_rx.await.unwrap();

        assert_eq!(
            transfer_handle.scheduler_decision(),
            SchedulingDecision::Execute
        );

        // Mark the transfer as complete with success
        transfer_handle.mark_complete(Ok(())).await;

        // wait for the scheduler to complete
        scheduler_result.await.unwrap().unwrap();
        // after the scheduler completes, the completed counter should be 1
        assert_eq!(completed.load(Ordering::Relaxed), 1);
    }
}
