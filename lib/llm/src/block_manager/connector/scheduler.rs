use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use super::protocol::*;
use super::*;

use either::Either;
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
                scheduler_tx
                    .send(TransferToSchedulerMessage::ScheduleRequest(request))
                    .await?;
                Ok(response_rx.await?.wait_for_decision().await)
            }
        }
    }
}

pub struct WorkerSchedulerClient {
    slots: HashMap<String, WorkerSchedulerClientSlot>,
    scheduler_tx: mpsc::UnboundedSender<SchedulerMessage>,
    cancel_token: CancellationToken,
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
            cancel_token: CancellationToken::new(),
            iteration: 0,
            iteration_complete: true,
            layers_complete: 0,
        }
    }

    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    pub fn start_next_iteration(&mut self) -> Result<(), SchedulerError> {
        debug_assert!(
            self.iteration_complete,
            "previous iteration must be complete before starting a new iteration"
        );
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

pub struct WorkerSchedulerClientSlot {
    operations: Vec<uuid::Uuid>,
    completed: Arc<AtomicU64>,
    cancel_token: CancellationToken,
}

impl WorkerSchedulerClientSlot {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            operations: Vec::new(),
            completed: Arc::new(AtomicU64::new(0)),
            cancel_token,
        }
    }

    fn make_scheduler_slot_request(&self, request_id: String) -> SchedulerCreateSlotDetails {
        SchedulerCreateSlotDetails {
            request_id,
            completed: self.completed.clone(),
            cancel_token: self.cancel_token.clone(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Relaxed) == self.operations.len() as u64
    }
}

impl WorkerSchedulerClient {
    pub fn create_slot(&mut self, request_id: String) -> Result<(), SchedulerError> {
        // create a child token which will cancel on the parent but can be cancelled individually
        // with out effecting the parent
        let token = self.cancel_token.child_token();

        // create a request slot with the child token
        // this will be the local worker slot
        let slot = WorkerSchedulerClientSlot::new(token);
        let request = slot.make_scheduler_slot_request(request_id.clone());

        // insert the slot into the local worker slots map
        self.slots.insert(request_id, slot);

        // send a request to insert the slot into the engine state
        self.scheduler_tx
            .send(SchedulerMessage::CreateSlot(request))
            .map_err(|_| SchedulerError::Disconnected)?;
        Ok(())
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
        let slot = self.slots.get(request_id).expect("slot does not exist");
        slot.completed.load(Ordering::Relaxed) == slot.operations.len() as u64
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
    ///
    RequestFinished(String),
}

struct WorkerArrivedFirst;
struct TransferArrivedFirst(TransferScheduleRequest);

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

    // This object coordinates the two-stage execution of a scheduled transfer request
    // Because we require signals from both leader and worker, this object holds the first
    // signal to arrive while waiting for the other.
    enqueued_requests:
        HashMap<String, HashMap<uuid::Uuid, Either<WorkerArrivedFirst, TransferArrivedFirst>>>,

    // This object coordinates the two-stage execution of a scheduled transfer request.
    // If the scheduled request arrives first, the controller object will be Some; otherwise,
    // the worker-side request arrived first and it will be None.
    scheduled_requests: HashMap<String, HashMap<uuid::Uuid, Option<ScheduledTaskController>>>,

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
                scheduled_requests: HashMap::new(),
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
                        self.add_slot(request, self.iteration);
                    }
                    Some(SchedulerMessage::RequestFinished(request_id)) => {
                        self.remove_slot(request_id);
                    }
                    Some(SchedulerMessage::EnqueueRequest(request)) => {
                        self.handle_enqueue_request(request);
                    }
                    None => {
                        return false;
                    }
                    _ => {
                        panic!("received unexpected message from worker");
                    }

                }
            }
            maybe_transfer_msg = self.transfer_rx.recv(), if !self.transfer_rx.is_closed() => {
                match maybe_transfer_msg {
                    Some(TransferToSchedulerMessage::ScheduleRequest(request)) => {
                        self.handle_schedule_request(request);
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
    fn add_slot(&mut self, req: SchedulerCreateSlotDetails, created_at: u64) {
        let request_id = req.request_id.clone();
        debug_assert!(!self.slots.contains_key(&request_id), "slot already exists");
        tracing::debug!("engine state adding slot");
        let slot = SchedulerSlot::new(req, created_at);
        if let Some(unprocessed_results) = self.unprocessed_immediate_results.remove(&request_id) {
            tracing::debug!(
                "found {} unprocessed immediate results; adding to slot",
                unprocessed_results.len()
            );
            slot.completed
                .fetch_add(unprocessed_results.len() as u64, Ordering::Relaxed);
        }
        self.slots.insert(request_id, slot);
    }

    fn remove_slot(&mut self, request_id: String) {
        debug_assert!(self.slots.contains_key(&request_id), "slot not found");
        self.cancel_tokens.remove(&request_id);
        self.slots.remove(&request_id);
        tracing::debug!(
            request_id,
            iteration = self.iteration,
            "engine state removing slot"
        );
    }

    fn handle_enqueue_request(&mut self, request: WorkerTransferRequest) {
        debug_assert!(
            self.slots.contains_key(&request.request_id),
            "slot does not exist"
        );
        let slot = self
            .slots
            .get_mut(&request.request_id)
            .expect("slot does not exist");

        unimplemented!("@ziqi")
    }

    fn start_iteration(&mut self, iteration: u64) {
        tracing::debug!(iteration, "engine state updating iteration");
        debug_assert!(
            self.iteration_complete,
            "previous iteration must be complete before starting a new iteration"
        );
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

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.leader_request.request_id))]
    fn handle_schedule_request(&mut self, request: TransferScheduleRequest) {
        let request_id = request.leader_request.request_id.clone();

        // the slot may or may not exist
        // if it exists, the worker side may or may not have arrived first

        // tracing::debug!("engine state adding slot");
        // let slot = SchedulerSlot::new(request, self.iteration);
        // self.slots.insert(request_id, slot);

        // capture a clone of the atomic counter, and pass it through with the request

        unimplemented!("@ziqi")
    }

    // this function will be a scheduler and will dispatch requests to be executed
    fn schedule_request(&mut self, xfer_req: TransferScheduleRequest) {
        // tokio spawn execute_scheduled_transfer for first impl.  add fanciness later.
        unimplemented!("@ziqi")
    }

    // this function will execute a transfer request, monitor its completion, and increment its
    // atomic completion counter when finished.
    //
    // this must tokio spawn and an indpendent task
    fn execute_scheduled_transfer(&mut self, xfer_req: TransferScheduleRequest) {
        // this will issue the signal to the transfer engine to start the transfer
        // this is oneshot sender with a SchedulingDecision
        // then this will monitor the oneshot received of a result<()> to monitor completion
        // when completed, panic if error, otherwise increment the atomic counter

        unimplemented!("@ziqi")
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

    /// Coordinates the two-stage execution: sends a scheduling decision to a ScheduledTaskHandle
    /// and awaits the completion result from the TransferCompletionHandle.mark_complete() call.
    ///
    /// This function runs on the Scheduler side and bridges the decision-making with completion tracking.
    #[tracing::instrument(level = "debug", skip_all)]
    async fn coordinate_scheduled_transfer_execution(
        controller: ScheduledTaskController,
        decision: SchedulingDecision,
    ) -> anyhow::Result<()> {
        let (completion_tx, completion_rx) = oneshot::channel();

        // Send the scheduling decision along with the completion channel to the task handle
        if controller
            .decision_tx
            .send((decision, completion_tx))
            .is_err()
        {
            anyhow::bail!("Failed to send scheduling decision to task handle");
        }

        // Await the completion result from the TransferCompletionHandle.mark_complete() call
        match completion_rx.await {
            Ok(result) => result,
            Err(_) => anyhow::bail!("Failed to receive completion result from transfer handle"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScheduledTaskError {}

pub struct ScheduledTaskController {
    request: LeaderTransferRequest,
    decision_tx: oneshot::Sender<(SchedulingDecision, oneshot::Sender<anyhow::Result<()>>)>,
}

pub struct SchedulerCreateSlotDetails {
    pub request_id: String,
    pub completed: Arc<AtomicU64>,
    pub cancel_token: CancellationToken,
}

pub struct SchedulerSlot {
    request_id: String,
    cancel_token: CancellationToken,
    completed: Arc<AtomicU64>,
    created_at: u64,
}

impl SchedulerSlot {
    fn new(req: SchedulerCreateSlotDetails, created_at: u64) -> Self {
        Self {
            request_id: req.request_id,
            completed: req.completed,
            cancel_token: req.cancel_token,
            created_at,
        }
    }
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
        let (mut scheduler, mut worker_client, transfer_client) = Scheduler::new(cancel_token);

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

        let mut handle = transfer_client
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

        // the request is completed
        worker_client.create_slot("test".to_string()).unwrap();

        assert!(!scheduler.slots.contains_key("test"));
        scheduler.step().await;
        assert!(scheduler.slots.contains_key("test"));

        // the unprocessed results should now be processed
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
        assert_eq!(worker_client.slots.get("test").unwrap().operations.len(), 0);
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

        let mut handle = transfer_client
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

        let mut handle = tokio::spawn(transfer_client.clone().schedule_transfer(request));

        // the transfer engine will immediately return a completion handle
    }

    /// in this case, the request arrives first via the worker client, meaning it traverse
    #[tokio::test]
    async fn test_transfer_scheduled_arrives_last() {
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

        // let (tx_1, rx_1) = tokio::sync::oneshot::channel();

        // let mut handle = tokio::spawn(async move {
        //     let handle = transfer_client
        //         .clone()
        //         .schedule_transfer(request)
        //         .await
        //         .unwrap();
        //     //let _ = rx_1.await;

        //     handle.mark_complete(Ok(())).await;
        // });

        // test state of scheduler before first message arrives.

        // scheudler.step().await;

        // trigger task

        // task is running

        // advance transfer tx_1 - this will mark the end of the task and trigger the completion handle
        // drop(tx_1);

        // sleep for a bit

        // check atomic counter

        // test the first arrival state.

        // the transfer engine will immediately return a completion handle
    }

    // #[tokio::test]
    // async fn test_coordinate_scheduled_transfer_execution() {
    //     dynamo_runtime::logging::init();

    //     let cancel_token = CancellationToken::new();
    //     let (mut scheduler, mut worker_client, transfer_client) = Scheduler::new(cancel_token);

    //     let operation_id = uuid::Uuid::new_v4();

    //     // Create a scheduled transfer request
    //     let request = LeaderTransferRequest {
    //         request_id: "test".to_string(),
    //         uuid: operation_id,
    //         requirement: None,
    //         request_type: RequestType::Scheduled,
    //     };

    //     // allows us to pause the transfer task after the scheduler decision is made
    //     // but before the transfer is marked as complete
    //     let (execute_transfer_tx, execute_transfer_rx) = oneshot::channel();

    //     // Spawn the schedule_transfer call which will await our coordination function
    //     let transfer_task = tokio::spawn(async move {
    //         let mut handle = transfer_client
    //             .clone()
    //             .schedule_transfer(request)
    //             .await
    //             .unwrap();

    //         assert!(handle.scheduler_decision() == SchedulingDecision::Execute);

    //         // Simulate some work being done - wait until the test releases us
    //         let _ = execute_transfer_rx.await;

    //         // Mark the transfer as complete with success
    //         handle.mark_complete(Ok(())).await;
    //     });

    //     // In this test, we will manually circumvent the coordinated scheduling of the transfer task
    //     // with the paired worker request by tapping the scheduler's internal receive channel direclty
    //     // and
    //     // scheduler.step().await;

    //     // Get the pending schedule request from the scheduler's internal queue
    //     // In a real implementation, this would be handled by the scheduler's internal logic
    //     // For this test, we'll simulate finding a pending request and using our coordination function

    //     // Simulate the scheduler making a decision and coordinating the execution
    //     if let Some(msg) = scheduler.transfer_rx.try_recv().ok() {
    //         match msg {
    //             TransferToSchedulerMessage::ScheduleRequest(schedule_req) => {
    //                 let decision = SchedulingDecision::Execute;

    //                 // Extract the decision sender from the ScheduledTaskHandle that would be created
    //                 let (decision_tx, decision_rx) = oneshot::channel();
    //                 let cancel_token = CancellationToken::new();

    //                 // Create a mock ScheduledTaskHandle to send to the client
    //                 let task_handle = ScheduledTaskHandle {
    //                     request_id: schedule_req.leader_request.request_id.clone(),
    //                     uuid: schedule_req.leader_request.uuid,
    //                     decision_rx,
    //                     cancel_token: cancel_token.child_token(),
    //                 };

    //                 // Send the task handle back to the client side
    //                 if schedule_req.response_tx.send(task_handle).is_ok() {
    //                     // Now use our coordination function to send the decision and await completion
    //                     let coordination_result =
    //                         Scheduler::coordinate_scheduled_transfer_execution(
    //                             decision_tx,
    //                             decision,
    //                         )
    //                         .await;

    //                     // The coordination should succeed
    //                     assert!(coordination_result.is_ok());
    //                 }
    //             }
    //             _ => {}
    //         }
    //     }

    //     // Wait for the transfer task to complete
    //     transfer_handle.await.unwrap();
    // }
}
