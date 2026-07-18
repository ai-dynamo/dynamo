// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable live-request boundary for the Mocker schedulers.
//!
//! This module owns the common submit, output-demultiplexing, and cancellation
//! mechanics needed by network-facing mock engines. Requests and cancellations
//! share the scheduler's ordered command channel, so dropping a response stream
//! promptly releases queued or running scheduler state.

use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::engine::create_engine;
use crate::scheduler::{
    MockerMetrics, SchedulerCommand, SchedulerCommandEnvelope, SchedulerCommandResult,
    SchedulerHandle,
};

type RequestOutputs = Arc<DashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>;

/// A running Mocker scheduler with request-scoped output streams.
#[derive(Clone)]
pub struct LiveEngine {
    inner: Arc<LiveEngineInner>,
}

struct LiveEngineInner {
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    active: RequestOutputs,
    metrics_rx: tokio::sync::watch::Receiver<MockerMetrics>,
    cancel: CancellationToken,
    // The scheduler's drop guard owns its task lifetime.
    #[allow(dead_code)]
    scheduler: Box<dyn SchedulerHandle>,
}

impl LiveEngine {
    /// Start one live scheduler at `dp_rank`.
    pub fn start(args: MockEngineArgs, dp_rank: u32) -> anyhow::Result<Self> {
        let args = args
            .normalized()
            .context("invalid Mocker engine arguments")?;
        let cancel = CancellationToken::new();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let scheduler = create_engine(
            args,
            dp_rank,
            Some(output_tx),
            KvEventPublishers::default(),
            Some(cancel.clone()),
            FpmPublisher::default(),
        );
        let command_tx = scheduler.command_sender();
        let metrics_rx = scheduler.metrics_receiver();
        let active: RequestOutputs = Arc::new(DashMap::new());
        let dispatcher_active = Arc::clone(&active);
        let dispatcher_cancel = cancel.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = dispatcher_cancel.cancelled() => break,
                    batch = output_rx.recv() => {
                        let Some(batch) = batch else { break };
                        for signal in batch {
                            let uuid = signal.uuid;
                            let terminal = signal.completed;
                            if let Some(sender) = dispatcher_active.get(&uuid) {
                                let _ = sender.send(signal);
                            }
                            if terminal {
                                dispatcher_active.remove(&uuid);
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            inner: Arc::new(LiveEngineInner {
                command_tx,
                active,
                metrics_rx,
                cancel,
                scheduler,
            }),
        })
    }

    /// Submit a request and return its scoped output receiver.
    pub async fn submit(&self, mut request: DirectRequest) -> anyhow::Result<LiveRequest> {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        request.uuid = Some(uuid);
        let (tx, rx) = mpsc::unbounded_channel();
        match self.inner.active.entry(uuid) {
            Entry::Occupied(_) => bail!("request {uuid} is already active"),
            Entry::Vacant(entry) => {
                entry.insert(tx);
            }
        }

        let result = send_command(&self.inner.command_tx, SchedulerCommand::Submit(request)).await;
        match result {
            Ok(SchedulerCommandResult::Submitted(submitted)) if submitted == uuid => {}
            Ok(result) => {
                self.inner.active.remove(&uuid);
                bail!("unexpected scheduler submit result for {uuid}: {result:?}");
            }
            Err(error) => {
                self.inner.active.remove(&uuid);
                return Err(error);
            }
        }

        Ok(LiveRequest {
            uuid,
            rx,
            active: Arc::clone(&self.inner.active),
            command_tx: self.inner.command_tx.clone(),
        })
    }

    /// Cancel an active request and wait until the scheduler applies it.
    pub async fn cancel(&self, request_id: Uuid) -> anyhow::Result<bool> {
        self.inner.active.remove(&request_id);
        cancel_request(&self.inner.command_tx, request_id).await
    }

    /// Subscribe to live scheduler occupancy and KV metrics.
    pub fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.inner.metrics_rx.clone()
    }

    /// Number of response streams currently registered with the dispatcher.
    pub fn active_request_count(&self) -> usize {
        self.inner.active.len()
    }
}

impl Drop for LiveEngineInner {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Request-owned stream of Mocker output signals.
pub struct LiveRequest {
    uuid: Uuid,
    rx: mpsc::UnboundedReceiver<OutputSignal>,
    active: RequestOutputs,
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
}

impl LiveRequest {
    pub fn id(&self) -> Uuid {
        self.uuid
    }

    pub async fn recv(&mut self) -> Option<OutputSignal> {
        self.rx.recv().await
    }

    /// Cancel this request and wait for scheduler-side cleanup.
    pub async fn cancel(mut self) -> anyhow::Result<bool> {
        self.active.remove(&self.uuid);
        let result = cancel_request(&self.command_tx, self.uuid).await;
        // Prevent Drop from sending a duplicate best-effort cancellation.
        self.rx.close();
        result
    }
}

impl Drop for LiveRequest {
    fn drop(&mut self) {
        if self.active.remove(&self.uuid).is_none() {
            return;
        }
        let command_tx = self.command_tx.clone();
        let request_id = self.uuid;
        // Drop cannot await the ordered command acknowledgement. The task is
        // intentionally best-effort during runtime shutdown, when the whole
        // scheduler is already being cancelled.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(error) = cancel_request(&command_tx, request_id).await {
                    tracing::debug!(%request_id, %error, "live Mocker request cancellation failed");
                }
            });
        }
    }
}

async fn cancel_request(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    request_id: Uuid,
) -> anyhow::Result<bool> {
    match send_command(command_tx, SchedulerCommand::CancelRequest { request_id }).await? {
        SchedulerCommandResult::Applied => Ok(true),
        SchedulerCommandResult::Noop => Ok(false),
        result => Err(anyhow!(
            "unexpected scheduler cancellation result for {request_id}: {result:?}"
        )),
    }
}

async fn send_command(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    command: SchedulerCommand,
) -> anyhow::Result<SchedulerCommandResult> {
    let (reply, response) = oneshot::channel();
    command_tx
        .send(SchedulerCommandEnvelope { command, reply })
        .await
        .map_err(|_| anyhow!("Mocker scheduler is not accepting commands"))?;
    let effects = response
        .await
        .map_err(|_| anyhow!("Mocker scheduler dropped a command acknowledgement"))??;
    Ok(effects.result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::EngineType;

    fn args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .block_size(4)
            .num_gpu_blocks(128)
            .max_num_seqs(Some(8))
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(1000.0)
            .dp_size(1)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn streams_planned_tokens_to_the_owning_request() {
        let engine = LiveEngine::start(args(), 0).unwrap();
        let uuid = Uuid::from_u128(1);
        let mut request = engine
            .submit(DirectRequest {
                tokens: vec![1, 2, 3],
                max_output_tokens: 3,
                output_token_ids: Some(vec![41, 42, 43]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();

        let mut outputs = Vec::new();
        while let Some(signal) = request.recv().await {
            outputs.push((signal.token_id, signal.completed));
            if signal.completed {
                break;
            }
        }
        assert_eq!(
            outputs,
            vec![(Some(41), false), (Some(42), false), (Some(43), true)]
        );
        assert_eq!(engine.active_request_count(), 0);
    }

    #[tokio::test]
    async fn dropping_a_stream_releases_scheduler_state() {
        let mut slow_args = args();
        slow_args.speedup_ratio = 1.0;
        let engine = LiveEngine::start(slow_args, 0).unwrap();
        let request = engine
            .submit(DirectRequest {
                tokens: vec![1; 256],
                max_output_tokens: 10_000,
                uuid: Some(Uuid::from_u128(2)),
                ..Default::default()
            })
            .await
            .unwrap();
        drop(request);

        let mut metrics = engine.metrics_receiver();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let snapshot = metrics.borrow_and_update().clone();
                if snapshot.running_requests == 0 && snapshot.waiting_requests == 0 {
                    break;
                }
                metrics.changed().await.unwrap();
            }
        })
        .await
        .expect("request cancellation should release scheduler state");
        assert_eq!(engine.active_request_count(), 0);
    }

    #[tokio::test]
    async fn duplicate_request_id_does_not_replace_the_original_stream() {
        let engine = LiveEngine::start(args(), 0).unwrap();
        let uuid = Uuid::from_u128(3);
        let original = engine
            .submit(DirectRequest {
                tokens: vec![1, 2, 3],
                max_output_tokens: 1_000,
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();
        let duplicate = engine
            .submit(DirectRequest {
                tokens: vec![4, 5, 6],
                max_output_tokens: 1,
                uuid: Some(uuid),
                ..Default::default()
            })
            .await;
        let error = match duplicate {
            Ok(_) => panic!("duplicate request ID must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("already active"));
        assert_eq!(engine.active_request_count(), 1);
        original.cancel().await.unwrap();
        assert_eq!(engine.active_request_count(), 0);
    }
}
