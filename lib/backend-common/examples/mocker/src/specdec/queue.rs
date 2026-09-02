// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, Semaphore, mpsc};
use tokio::task::{JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::protocol::MAX_OUTPUT_TOKENS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenMode {
    Echo,
    Counter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationPhase {
    Queued,
    Prefilling,
    Emitting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueEvent {
    Queued,
    PrefillStarted,
    PrefillComplete,
    Token { index: u32, token_id: u32 },
    Complete { emitted_tokens: u32 },
    Cancelled { phase: CancellationPhase },
}

#[derive(Debug, Clone)]
pub struct JobSpec {
    pub request_id: Uuid,
    pub prompt_token_ids: Vec<u32>,
    pub max_output_tokens: u32,
    pub prefill_duration: Duration,
    pub token_interval: Duration,
    pub token_mode: TokenMode,
}

impl JobSpec {
    fn validate(&self) -> Result<(), QueueError> {
        if self.prompt_token_ids.is_empty() {
            return Err(QueueError::InvalidJob("prompt tokens must not be empty"));
        }
        if !(1..=MAX_OUTPUT_TOKENS).contains(&self.max_output_tokens) {
            return Err(QueueError::InvalidJob(
                "output token count exceeds protocol limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    pub queue_capacity: usize,
    pub concurrency: usize,
    pub output_capacity: usize,
}

impl SchedulerConfig {
    pub fn validate(self) -> Result<Self, QueueError> {
        if self.queue_capacity == 0 {
            return Err(QueueError::InvalidConfig("queue capacity must be positive"));
        }
        if self.concurrency == 0 {
            return Err(QueueError::InvalidConfig("concurrency must be positive"));
        }
        if self.output_capacity < 2 {
            return Err(QueueError::InvalidConfig(
                "output capacity must be at least two",
            ));
        }
        if self.queue_capacity > Semaphore::MAX_PERMITS
            || self.concurrency > Semaphore::MAX_PERMITS
            || self.output_capacity > Semaphore::MAX_PERMITS
        {
            return Err(QueueError::InvalidConfig(
                "queue capacity exceeds the Tokio primitive limit",
            ));
        }
        Ok(self)
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 32,
            concurrency: 4,
            output_capacity: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    InvalidConfig(&'static str),
    InvalidJob(&'static str),
    Full,
    ShuttingDown,
    DispatcherFailed,
}

impl fmt::Display for QueueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) | Self::InvalidJob(message) => {
                formatter.write_str(message)
            }
            Self::Full => formatter.write_str("fake-inference queue is full"),
            Self::ShuttingDown => formatter.write_str("fake-inference queue is shutting down"),
            Self::DispatcherFailed => formatter.write_str("fake-inference dispatcher failed"),
        }
    }
}

impl std::error::Error for QueueError {}

pub struct JobHandle {
    request_id: Uuid,
    cancel: CancellationToken,
    events: mpsc::Receiver<QueueEvent>,
}

impl JobHandle {
    pub fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    pub async fn recv(&mut self) -> Option<QueueEvent> {
        self.events.recv().await
    }
}

struct QueuedJob {
    spec: JobSpec,
    cancel: CancellationToken,
    events: mpsc::Sender<QueueEvent>,
}

pub struct FakeScheduler {
    sender: Mutex<Option<mpsc::Sender<QueuedJob>>>,
    cancel: CancellationToken,
    dispatcher: Mutex<Option<JoinHandle<()>>>,
    output_capacity: usize,
}

impl FakeScheduler {
    pub fn start(config: SchedulerConfig) -> Result<Arc<Self>, QueueError> {
        let config = config.validate()?;
        let (sender, receiver) = mpsc::channel(config.queue_capacity);
        let cancel = CancellationToken::new();
        let dispatcher_cancel = cancel.clone();
        let dispatcher = tokio::spawn(run_dispatcher(
            receiver,
            config.concurrency,
            dispatcher_cancel,
        ));
        Ok(Arc::new(Self {
            sender: Mutex::new(Some(sender)),
            cancel,
            dispatcher: Mutex::new(Some(dispatcher)),
            output_capacity: config.output_capacity,
        }))
    }

    pub async fn submit(&self, spec: JobSpec) -> Result<JobHandle, QueueError> {
        spec.validate()?;
        let request_id = spec.request_id;
        let cancel = CancellationToken::new();
        let (events, receiver) = mpsc::channel(self.output_capacity);
        events
            .try_send(QueueEvent::Queued)
            .map_err(|_| QueueError::DispatcherFailed)?;
        let sender = self
            .sender
            .lock()
            .await
            .as_ref()
            .cloned()
            .ok_or(QueueError::ShuttingDown)?;
        sender
            .try_send(QueuedJob {
                spec,
                cancel: cancel.clone(),
                events,
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => QueueError::Full,
                mpsc::error::TrySendError::Closed(_) => QueueError::ShuttingDown,
            })?;
        Ok(JobHandle {
            request_id,
            cancel,
            events: receiver,
        })
    }

    pub async fn shutdown(&self) -> Result<(), QueueError> {
        self.sender.lock().await.take();
        self.cancel.cancel();
        let task = self.dispatcher.lock().await.take();
        match task {
            Some(task) => task.await.map_err(|_| QueueError::DispatcherFailed),
            None => Ok(()),
        }
    }
}

impl Drop for FakeScheduler {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn run_dispatcher(
    mut receiver: mpsc::Receiver<QueuedJob>,
    concurrency: usize,
    cancel: CancellationToken,
) {
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut jobs = JoinSet::new();

    loop {
        while let Some(result) = jobs.try_join_next() {
            if let Err(error) = result {
                tracing::error!(%error, "fake-inference job task failed");
            }
        }
        let permit = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            permit = semaphore.clone().acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => break,
            },
        };
        let queued = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                drop(permit);
                break;
            }
            queued = receiver.recv() => queued,
        };
        let Some(queued) = queued else {
            drop(permit);
            break;
        };
        let job_cancel = cancel.clone();
        jobs.spawn(async move {
            let _permit = permit;
            run_job(queued, job_cancel).await;
        });
    }

    receiver.close();
    while let Ok(queued) = receiver.try_recv() {
        queued.cancel.cancel();
        let _ = queued.events.try_send(QueueEvent::Cancelled {
            phase: CancellationPhase::Queued,
        });
    }
    cancel.cancel();
    while let Some(result) = jobs.join_next().await {
        if let Err(error) = result {
            tracing::error!(%error, "fake-inference job task failed");
        }
    }
}

async fn run_job(queued: QueuedJob, shutdown: CancellationToken) {
    if cancelled(&queued, &shutdown, CancellationPhase::Queued) {
        return;
    }
    if !send_event(&queued, &shutdown, QueueEvent::PrefillStarted).await {
        return;
    }
    if !sleep_or_cancel(
        &queued,
        &shutdown,
        queued.spec.prefill_duration,
        CancellationPhase::Prefilling,
    )
    .await
    {
        return;
    }
    if !send_event(&queued, &shutdown, QueueEvent::PrefillComplete).await {
        return;
    }

    for index in 0..queued.spec.max_output_tokens {
        if !sleep_or_cancel(
            &queued,
            &shutdown,
            queued.spec.token_interval,
            CancellationPhase::Emitting,
        )
        .await
        {
            return;
        }
        let token_id = token_for(&queued.spec, index);
        if !send_event(&queued, &shutdown, QueueEvent::Token { index, token_id }).await {
            return;
        }
    }
    let _ = send_event(
        &queued,
        &shutdown,
        QueueEvent::Complete {
            emitted_tokens: queued.spec.max_output_tokens,
        },
    )
    .await;
}

fn token_for(spec: &JobSpec, index: u32) -> u32 {
    match spec.token_mode {
        TokenMode::Echo => spec.prompt_token_ids[index as usize % spec.prompt_token_ids.len()],
        TokenMode::Counter => {
            let bytes = spec.request_id.as_bytes();
            let seed = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            seed.wrapping_add(index)
        }
    }
}

fn cancelled(queued: &QueuedJob, shutdown: &CancellationToken, phase: CancellationPhase) -> bool {
    if queued.cancel.is_cancelled() || shutdown.is_cancelled() {
        let _ = queued.events.try_send(QueueEvent::Cancelled { phase });
        true
    } else {
        false
    }
}

async fn sleep_or_cancel(
    queued: &QueuedJob,
    shutdown: &CancellationToken,
    duration: Duration,
    phase: CancellationPhase,
) -> bool {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            let _ = queued.events.try_send(QueueEvent::Cancelled { phase });
            false
        }
        _ = queued.cancel.cancelled() => {
            let _ = queued.events.try_send(QueueEvent::Cancelled { phase });
            false
        }
        _ = tokio::time::sleep(duration) => true,
    }
}

async fn send_event(queued: &QueuedJob, shutdown: &CancellationToken, event: QueueEvent) -> bool {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => false,
        _ = queued.cancel.cancelled() => false,
        result = queued.events.send(event) => result.is_ok(),
    }
}
