// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Transfer executor scaffolding for connector integrations.
//! Slot state machine and transfer planning logic live under
//! `crate::v2::integrations::connector::slot`.

pub mod console;
pub use console::{ConnectorConsoleHook, ConsoleHookRef, NoopConsoleHook};

#[cfg(test)]
pub mod testing;

pub use crate::v2::integrations::connector::slot::*;

use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{error, warn};

pub trait RequestKey:
    std::hash::Hash
    + std::cmp::Eq
    + std::fmt::Debug
    + std::fmt::Display
    + tracing::Value
    + Clone
    + Send
    + Sync
    + 'static
{
}

impl RequestKey for String {}

pub struct SlotExecutor<D, B> {
    dispatcher: D,
    broadcaster: B,
}

impl<D, B> SlotExecutor<D, B> {
    pub fn new(dispatcher: D, broadcaster: B) -> Self {
        Self {
            dispatcher,
            broadcaster,
        }
    }

    pub fn into_parts(self) -> (D, B) {
        (self.dispatcher, self.broadcaster)
    }
}

impl<D, B> SlotExecutor<D, B>
where
    D: TransferDispatcher,
    B: SlotStateBroadcaster,
{
    pub fn execute(
        &self,
        slot: &mut Slot,
        mut actions: SlotActions,
    ) -> Result<(), SlotExecutionError<D::Error, B::Error>> {
        let request_id = slot.core().request_id().to_string();

        for notification in &actions.state_notifications {
            self.broadcaster
                .publish(&request_id, notification)
                .map_err(SlotExecutionError::Broadcast)?;
        }

        for transfer in actions.transfers.drain(..) {
            self.dispatch_transfer(slot, transfer)
                .map_err(SlotExecutionError::Dispatch)?;
        }

        if let Some(num_tokens) = actions.num_cached_device_tokens {
            slot.core_mut().set_cached_device_tokens(num_tokens);
        }
        if let Some(num_tokens) = actions.num_cached_host_tokens {
            slot.core_mut().set_cached_host_tokens(num_tokens);
        }
        if let Some(num_tokens) = actions.num_cached_disk_tokens {
            slot.core_mut().set_cached_disk_tokens(num_tokens);
        }

        Ok(())
    }

    fn dispatch_transfer(
        &self,
        slot: &mut Slot,
        transfer: PlannedTransfer,
    ) -> Result<(), D::Error> {
        let uuid = transfer.transfer_id;
        let metadata = slot.take_candidate_metadata(uuid);
        let record = TransferCandidateRecord::new(transfer.clone(), metadata, None);
        slot.record_candidate(record.clone());

        match self.dispatcher.dispatch(slot.core().request_id(), record) {
            Ok(()) => Ok(()),
            Err(err) => {
                let _ = slot.skip_candidate(uuid, TransferSkipReason::Cancelled);
                Err(err)
            }
        }
    }
}

#[derive(Debug)]
pub enum SlotExecutionError<D, B> {
    Dispatch(D),
    Broadcast(B),
}

pub trait TransferDispatcher {
    type Error;

    fn dispatch(
        &self,
        request_id: &str,
        candidate: TransferCandidateRecord,
    ) -> Result<(), Self::Error>;
}

pub trait SlotStateBroadcaster {
    type Error;

    fn publish(
        &self,
        request_id: &str,
        notification: &StateNotification,
    ) -> Result<(), Self::Error>;
}

pub trait TransferEvaluationSink {
    type Error;

    fn publish(
        &self,
        request_id: &str,
        outcome: TransferEvaluationOutcome,
    ) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct TransferCandidateMessage {
    pub request_id: String,
    pub candidate: TransferCandidateRecord,
}

#[derive(Debug)]
pub struct TransferEngineEnvelope {
    pub request_id: String,
    pub command: TransferEngineCommand,
}

#[derive(Clone)]
pub struct ChannelTransferDispatcher {
    tx: mpsc::UnboundedSender<TransferCandidateMessage>,
}

impl ChannelTransferDispatcher {
    pub fn new(tx: mpsc::UnboundedSender<TransferCandidateMessage>) -> Self {
        Self { tx }
    }
}

impl TransferDispatcher for ChannelTransferDispatcher {
    type Error = mpsc::error::SendError<TransferCandidateMessage>;

    fn dispatch(
        &self,
        request_id: &str,
        candidate: TransferCandidateRecord,
    ) -> Result<(), Self::Error> {
        let message = TransferCandidateMessage {
            request_id: request_id.to_string(),
            candidate,
        };
        self.tx.send(message)
    }
}

#[derive(Debug, Clone)]
pub struct SlotStateEvent {
    pub request_id: String,
    pub notification: StateNotification,
}

#[derive(Clone)]
pub struct BroadcastSlotStateBroadcaster {
    tx: broadcast::Sender<SlotStateEvent>,
}

impl BroadcastSlotStateBroadcaster {
    pub fn new(tx: broadcast::Sender<SlotStateEvent>) -> Self {
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SlotStateEvent> {
        self.tx.subscribe()
    }
}

impl SlotStateBroadcaster for BroadcastSlotStateBroadcaster {
    type Error = broadcast::error::SendError<SlotStateEvent>;

    fn publish(
        &self,
        request_id: &str,
        notification: &StateNotification,
    ) -> Result<(), Self::Error> {
        let event = SlotStateEvent {
            request_id: request_id.to_string(),
            notification: *notification,
        };
        self.tx.send(event).map(|_| ())
    }
}

#[derive(Clone)]
pub struct TransferEvaluationMessage {
    pub request_id: String,
    pub outcome: TransferEvaluationOutcome,
}

pub struct TransferEvaluationPipeline<E, S> {
    evaluator: E,
    sink: S,
}

impl<E, S> TransferEvaluationPipeline<E, S>
where
    E: TransferEvaluator,
    S: TransferEvaluationSink,
{
    pub fn new(evaluator: E, sink: S) -> Self {
        Self { evaluator, sink }
    }

    pub async fn drive(
        self,
        mut candidate_rx: mpsc::UnboundedReceiver<TransferCandidateMessage>,
        cancellation: CancellationToken,
    ) -> Result<(), TransferPipelineError<S::Error>> {
        let TransferEvaluationPipeline { evaluator, sink } = self;

        loop {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    break;
                }
                maybe_candidate = candidate_rx.recv() => {
                    match maybe_candidate {
                        Some(message) => {
                            let outcome = evaluator.evaluate(&message.request_id, message.candidate);
                            sink.publish(&message.request_id, outcome)
                                .map_err(TransferPipelineError::Sink)?;
                        }
                        None => break,
                    }
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub enum TransferPipelineError<S> {
    Sink(S),
}

#[derive(Clone)]
pub struct ChannelTransferEvaluationSink {
    tx: mpsc::UnboundedSender<TransferEvaluationMessage>,
}

impl ChannelTransferEvaluationSink {
    pub fn new(tx: mpsc::UnboundedSender<TransferEvaluationMessage>) -> Self {
        Self { tx }
    }
}

impl TransferEvaluationSink for ChannelTransferEvaluationSink {
    type Error = mpsc::error::SendError<TransferEvaluationMessage>;

    fn publish(
        &self,
        request_id: &str,
        outcome: TransferEvaluationOutcome,
    ) -> Result<(), Self::Error> {
        let message = TransferEvaluationMessage {
            request_id: request_id.to_string(),
            outcome,
        };
        self.tx.send(message)
    }
}

pub struct TransferPipelineRuntime<R>
where
    R: RequestKey + Eq + std::hash::Hash + Clone + Send + Sync + 'static + std::borrow::Borrow<str>,
{
    slots: Arc<DashMap<R, Arc<dyn TransferSlotHandle>>>,
    dispatcher: ChannelTransferDispatcher,
    evaluation_sink: ChannelTransferEvaluationSink,
    engine_tx: mpsc::UnboundedSender<TransferEngineEnvelope>,
    engine_rx: Mutex<Option<mpsc::UnboundedReceiver<TransferEngineEnvelope>>>,
    cancellation: CancellationToken,
    pipeline_handle: JoinHandle<()>,
    evaluation_handle: JoinHandle<()>,
}

impl<R> TransferPipelineRuntime<R>
where
    R: RequestKey + Eq + std::hash::Hash + Clone + Send + Sync + 'static + std::borrow::Borrow<str>,
{
    pub fn new(block_manager: Arc<MockBlockManager>) -> Self {
        let (candidate_tx, candidate_rx) = mpsc::unbounded_channel();
        let dispatcher = ChannelTransferDispatcher::new(candidate_tx);

        let (evaluation_tx, evaluation_rx) = mpsc::unbounded_channel();
        let evaluation_sink = ChannelTransferEvaluationSink::new(evaluation_tx);

        let (engine_tx, engine_rx) = mpsc::unbounded_channel();

        let slots: Arc<DashMap<R, Arc<dyn TransferSlotHandle>>> = Arc::new(DashMap::new());
        let cancellation = CancellationToken::new();

        let pipeline = TransferEvaluationPipeline::new(
            DefaultTransferEvaluator::new(block_manager.clone()),
            evaluation_sink.clone(),
        );
        let pipeline_cancel = cancellation.clone();
        let pipeline_handle = tokio::spawn(async move {
            let _ = pipeline.drive(candidate_rx, pipeline_cancel).await;
        });

        let eval_slots = slots.clone();
        let eval_engine = engine_tx.clone();
        let evaluation_cancel = cancellation.clone();
        let evaluation_handle = tokio::spawn(async move {
            let mut evaluation_rx = evaluation_rx;
            loop {
                tokio::select! {
                    _ = evaluation_cancel.cancelled() => {
                        break;
                    }
                    maybe_outcome = evaluation_rx.recv() => {
                        match maybe_outcome {
                            Some(message) => {
                                if let Some(slot_ref) =
                                    eval_slots.get(message.request_id.as_str())
                                {
                                    let slot_handle: Arc<dyn TransferSlotHandle> =
                                        Arc::clone(slot_ref.value());
                                    drop(slot_ref);
                                    if let Some(promoted) = slot_handle.apply_evaluation(message.outcome) {
                                        let envelope = TransferEngineEnvelope {
                                            request_id: message.request_id.clone(),
                                            command: promoted.into_engine_command(),
                                        };
                                        if let Err(err) = eval_engine.send(envelope) {
                                            error!(request_id = message.request_id, "failed to enqueue transfer command: {err:?}");
                                        }
                                    }
                                } else {
                                    warn!(request_id = message.request_id, "received transfer outcome for unknown slot");
                                }
                            }
                            None => break,
                        }
                    }
                }
            }
        });

        Self {
            slots,
            dispatcher,
            evaluation_sink,
            engine_tx,
            engine_rx: Mutex::new(Some(engine_rx)),
            cancellation,
            pipeline_handle,
            evaluation_handle,
        }
    }

    pub fn dispatcher(&self) -> ChannelTransferDispatcher {
        self.dispatcher.clone()
    }

    pub fn evaluation_sink(&self) -> ChannelTransferEvaluationSink {
        self.evaluation_sink.clone()
    }

    pub fn engine_receiver(&self) -> Option<mpsc::UnboundedReceiver<TransferEngineEnvelope>> {
        self.engine_rx.lock().unwrap().take()
    }

    pub fn slots(&self) -> Arc<DashMap<R, Arc<dyn TransferSlotHandle>>> {
        self.slots.clone()
    }

    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.clone()
    }
}

impl<R> Drop for TransferPipelineRuntime<R>
where
    R: RequestKey + Eq + std::hash::Hash + Clone + Send + Sync + 'static + std::borrow::Borrow<str>,
{
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.pipeline_handle.abort();
        self.evaluation_handle.abort();
    }
}
