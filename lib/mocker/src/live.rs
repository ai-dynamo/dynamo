// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable live-request boundary for the Mocker schedulers.
//!
//! This module owns the common submit, output-demultiplexing, and cancellation
//! mechanics needed by network-facing mock engines. Submission admission is an
//! owned operation, while cancellation uses a separate bounded scheduler lane
//! that remains responsive during a long modeled pass.

use std::sync::{Arc, Mutex, Weak};

use anyhow::{Context, anyhow, bail};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::engine::create_engine_with_output_sender;
use crate::scheduler::{
    MockerMetrics, SchedulerCancellationEnvelope, SchedulerCommand, SchedulerCommandEnvelope,
    SchedulerCommandResult, SchedulerHandle, SchedulerOutputSender,
};

#[derive(Default)]
struct RequestRoutes {
    by_client: DashMap<Uuid, Arc<RequestRoute>>,
    by_scheduler: DashMap<Uuid, Arc<RequestRoute>>,
}

type Routes = Arc<RequestRoutes>;

const SCHEDULER_OUTPUT_CAPACITY: usize = 8;

/// Maximum number of response signals retained across all live request
/// streams. A caller reserves its declared response length before scheduler
/// admission and releases it when the request stream is dropped.
pub const MAX_BUFFERED_OUTPUT_SIGNALS: usize = 32_768;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestState {
    Submitting,
    Active,
    Cancelling,
    Closed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RequestLifecycle {
    state: RequestState,
    stream_abandoned: bool,
    terminal_seen: bool,
}

struct RequestRoute {
    client_id: Uuid,
    scheduler_id: Uuid,
    output_tx: Mutex<Option<mpsc::Sender<OutputSignal>>>,
    output_closed: CancellationToken,
    lifecycle_tx: watch::Sender<RequestLifecycle>,
    cancel_lock: tokio::sync::Mutex<()>,
}

impl RequestRoute {
    fn new(client_id: Uuid, scheduler_id: Uuid, output_tx: mpsc::Sender<OutputSignal>) -> Self {
        let (lifecycle_tx, _) = watch::channel(RequestLifecycle {
            state: RequestState::Submitting,
            stream_abandoned: false,
            terminal_seen: false,
        });
        Self {
            client_id,
            scheduler_id,
            output_tx: Mutex::new(Some(output_tx)),
            output_closed: CancellationToken::new(),
            lifecycle_tx,
            cancel_lock: tokio::sync::Mutex::new(()),
        }
    }

    fn activate(&self) {
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state != RequestState::Submitting {
                return false;
            }
            lifecycle.state = RequestState::Active;
            true
        });
    }

    fn abandon_stream(&self) {
        self.close_output();
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.stream_abandoned {
                return false;
            }
            lifecycle.stream_abandoned = true;
            true
        });
    }

    async fn wait_for_admission(&self) -> bool {
        let mut lifecycle_rx = self.lifecycle_tx.subscribe();
        loop {
            match lifecycle_rx.borrow_and_update().state {
                RequestState::Submitting | RequestState::Cancelling => {}
                RequestState::Active => return true,
                RequestState::Closed => return false,
            }
            if lifecycle_rx.changed().await.is_err() {
                return false;
            }
        }
    }

    fn begin_cancellation(&self) -> bool {
        let mut started = false;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state == RequestState::Active {
                lifecycle.state = RequestState::Cancelling;
                started = true;
                return true;
            }
            false
        });
        started
    }

    fn finish_cancellation(&self, result: &anyhow::Result<bool>) -> bool {
        let mut remove = false;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state != RequestState::Cancelling {
                return false;
            }
            remove = match result {
                Ok(true) => true,
                Ok(false) => lifecycle.stream_abandoned || lifecycle.terminal_seen,
                Err(_) => lifecycle.terminal_seen,
            };
            lifecycle.state = if remove {
                RequestState::Closed
            } else {
                RequestState::Active
            };
            true
        });
        if remove {
            self.close_output();
        }
        remove
    }

    async fn send_output(&self, signal: OutputSignal) -> OutputDelivery {
        let output_tx = self.output_tx.lock().unwrap().as_ref().cloned();
        let Some(output_tx) = output_tx else {
            return OutputDelivery::Closed;
        };
        tokio::select! {
            biased;
            _ = self.output_closed.cancelled() => OutputDelivery::Closed,
            result = output_tx.send(signal) => match result {
                Ok(()) => OutputDelivery::Delivered,
                Err(_) => OutputDelivery::Closed,
            },
        }
    }

    /// Record a terminal signal and return whether the route can be removed.
    /// An in-flight cancellation retains it until the scheduler acknowledges
    /// cleanup; its scheduler ID is never reused by a replacement request.
    fn observe_terminal(&self) -> bool {
        self.close_output();
        let mut remove = false;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            lifecycle.terminal_seen = true;
            if lifecycle.state != RequestState::Cancelling {
                lifecycle.state = RequestState::Closed;
                remove = true;
            }
            true
        });
        remove
    }

    fn shutdown(&self) {
        self.close_output();
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state == RequestState::Closed {
                return false;
            }
            lifecycle.state = RequestState::Closed;
            true
        });
    }

    fn close_output(&self) {
        self.output_closed.cancel();
        self.output_tx.lock().unwrap().take();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputDelivery {
    Delivered,
    Closed,
}

/// A running Mocker scheduler with request-scoped output streams.
#[derive(Clone)]
pub struct LiveEngine {
    inner: Arc<LiveEngineInner>,
}

struct LiveEngineInner {
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    routes: Routes,
    metrics_rx: tokio::sync::watch::Receiver<MockerMetrics>,
    output_budget: Arc<Semaphore>,
    output_budget_capacity: usize,
    cancel: CancellationToken,
    runtime: Handle,
    // The scheduler's drop guard owns its task lifetime.
    #[allow(dead_code)]
    scheduler: Box<dyn SchedulerHandle>,
}

impl LiveEngine {
    /// Start one live scheduler at `dp_rank`.
    pub fn start(args: MockEngineArgs, dp_rank: u32) -> anyhow::Result<Self> {
        Self::start_with_output_gate(args, dp_rank, None, MAX_BUFFERED_OUTPUT_SIGNALS)
    }

    fn start_with_output_gate(
        args: MockEngineArgs,
        dp_rank: u32,
        output_gate: Option<watch::Receiver<bool>>,
        output_budget: usize,
    ) -> anyhow::Result<Self> {
        let runtime =
            Handle::try_current().context("LiveEngine::start requires an active Tokio runtime")?;
        let args = args
            .normalized()
            .context("invalid Mocker engine arguments")?;
        anyhow::ensure!(
            output_budget > 0,
            "output signal budget must be greater than 0"
        );
        let cancel = CancellationToken::new();
        let (output_tx, output_rx) = mpsc::channel::<Vec<OutputSignal>>(SCHEDULER_OUTPUT_CAPACITY);
        let scheduler = create_engine_with_output_sender(
            args,
            dp_rank,
            Some(SchedulerOutputSender::Bounded(output_tx)),
            KvEventPublishers::default(),
            Some(cancel.clone()),
            FpmPublisher::default(),
        );
        let command_tx = scheduler.command_sender();
        let cancellation_tx = scheduler.cancellation_sender();
        let metrics_rx = scheduler.metrics_receiver();
        let routes = Arc::new(RequestRoutes::default());
        runtime.spawn(run_output_dispatcher(
            output_rx,
            Arc::clone(&routes),
            cancellation_tx.clone(),
            runtime.clone(),
            cancel.clone(),
            output_gate,
        ));

        Ok(Self {
            inner: Arc::new(LiveEngineInner {
                command_tx,
                cancellation_tx,
                routes,
                metrics_rx,
                output_budget: Arc::new(Semaphore::new(output_budget)),
                output_budget_capacity: output_budget,
                cancel,
                runtime,
                scheduler,
            }),
        })
    }

    /// Submit a request and return its scoped output receiver.
    pub async fn submit(&self, mut request: DirectRequest) -> anyhow::Result<LiveRequest> {
        // Both scheduler cores treat an explicit token plan as authoritative.
        // Normalize that invariant before reserving delivery capacity so the
        // dispatcher can never receive more signals than this route can hold.
        let output_capacity = request
            .output_token_ids
            .as_ref()
            .map_or(request.max_output_tokens, Vec::len);
        anyhow::ensure!(
            output_capacity > 0,
            "live requests must generate at least one output token"
        );
        request.max_output_tokens = output_capacity;
        let output_permits = u32::try_from(output_capacity)
            .context("declared output length exceeds the live response budget representation")?;
        anyhow::ensure!(
            output_capacity <= self.inner.output_budget_capacity,
            "declared output length {output_capacity} exceeds the live response budget"
        );
        let output_budget = self
            .inner
            .output_budget
            .clone()
            .acquire_many_owned(output_permits)
            .await
            .map_err(|_| anyhow!("live response budget closed during submission"))?;
        let client_id = request.uuid.unwrap_or_else(Uuid::new_v4);
        let scheduler_id = Uuid::new_v4();
        request.uuid = Some(scheduler_id);
        // A request emits at most one signal per effective output token (or one
        // rejection signal). The engine-wide semaphore bounds the sum of these
        // per-request channels while allowing the shared dispatcher to keep
        // unrelated routes moving when one client reads slowly.
        let (tx, rx) = mpsc::channel(output_capacity);
        let route = Arc::new(RequestRoute::new(client_id, scheduler_id, tx));
        match self.inner.routes.by_client.entry(client_id) {
            Entry::Occupied(_) => bail!("request {client_id} is already active"),
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&route));
            }
        }
        match self.inner.routes.by_scheduler.entry(scheduler_id) {
            Entry::Occupied(_) => {
                remove_route(&self.inner.routes, &route);
                bail!("internal scheduler request ID collision");
            }
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&route));
            }
        }
        // Own the registration before the first await. The scheduler admission
        // task survives cancellation of this submit future, so stream-drop
        // cleanup can wait for admission before using the independent
        // cancellation lane.
        let live = LiveRequest {
            client_id,
            rx,
            route: Arc::downgrade(&route),
            routes: Arc::clone(&self.inner.routes),
            cancellation_tx: self.inner.cancellation_tx.clone(),
            runtime: self.inner.runtime.clone(),
            _output_budget: output_budget,
        };

        let routes = Arc::clone(&self.inner.routes);
        let submission_route = Arc::clone(&route);
        let command_tx = self.inner.command_tx.clone();
        let submission = self.inner.runtime.spawn(async move {
            let result = send_command(&command_tx, SchedulerCommand::Submit(request)).await;
            let admission = match result {
                Ok(SchedulerCommandResult::Submitted(submitted)) if submitted == scheduler_id => {
                    Ok(())
                }
                Ok(result) => Err(anyhow!(
                    "unexpected scheduler submit result for {client_id}: {result:?}"
                )),
                Err(error) => Err(error),
            };
            if admission.is_ok() {
                submission_route.activate();
            } else {
                submission_route.shutdown();
                remove_route(&routes, &submission_route);
            }
            admission
        });
        match submission.await {
            Ok(result) => result?,
            Err(error) => {
                route.shutdown();
                remove_route(&self.inner.routes, &route);
                return Err(anyhow!("live Mocker submission task failed: {error}"));
            }
        }

        Ok(live)
    }

    /// Cancel an active request and wait until the scheduler applies it.
    pub async fn cancel(&self, request_id: Uuid) -> anyhow::Result<bool> {
        let Some(route) = self
            .inner
            .routes
            .by_client
            .get(&request_id)
            .map(|entry| Arc::clone(entry.value()))
        else {
            return Ok(false);
        };
        // ID-based cancellation is an Abort boundary: stop forwarding the
        // response immediately so a backpressured dispatcher cannot delay the
        // scheduler cancellation acknowledgement.
        route.abandon_stream();
        await_cancellation(spawn_cancellation(
            &self.inner.runtime,
            self.inner.cancellation_tx.clone(),
            Arc::clone(&self.inner.routes),
            route,
            true,
        ))
        .await
    }

    /// Subscribe to live scheduler occupancy and KV metrics.
    pub fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.inner.metrics_rx.clone()
    }

    /// Number of response streams currently registered with the dispatcher.
    pub fn active_request_count(&self) -> usize {
        self.inner.routes.by_client.len()
    }
}

impl Drop for LiveEngineInner {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Request-owned stream of Mocker output signals.
pub struct LiveRequest {
    client_id: Uuid,
    rx: mpsc::Receiver<OutputSignal>,
    route: Weak<RequestRoute>,
    routes: Routes,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    runtime: Handle,
    _output_budget: OwnedSemaphorePermit,
}

impl LiveRequest {
    pub fn id(&self) -> Uuid {
        self.client_id
    }

    pub async fn recv(&mut self) -> Option<OutputSignal> {
        self.rx.recv().await
    }

    /// Cancel this request and wait for scheduler-side cleanup.
    pub async fn cancel(self) -> anyhow::Result<bool> {
        let Some(route) = self.route.upgrade() else {
            return Ok(false);
        };
        route.abandon_stream();
        await_cancellation(spawn_cancellation(
            &self.runtime,
            self.cancellation_tx.clone(),
            Arc::clone(&self.routes),
            route,
            true,
        ))
        .await
    }
}

impl Drop for LiveRequest {
    fn drop(&mut self) {
        let Some(route) = self.route.upgrade() else {
            return;
        };
        route.abandon_stream();
        drop(spawn_cancellation(
            &self.runtime,
            self.cancellation_tx.clone(),
            Arc::clone(&self.routes),
            route,
            true,
        ));
    }
}

fn remove_route(routes: &RequestRoutes, route: &Arc<RequestRoute>) -> bool {
    let removed = routes
        .by_client
        .remove_if(&route.client_id, |_, current| Arc::ptr_eq(current, route))
        .is_some();
    routes
        .by_scheduler
        .remove_if(&route.scheduler_id, |_, current| {
            Arc::ptr_eq(current, route)
        });
    removed
}

fn route_is_registered(routes: &RequestRoutes, route: &Arc<RequestRoute>) -> bool {
    routes
        .by_client
        .get(&route.client_id)
        .is_some_and(|current| Arc::ptr_eq(current.value(), route))
        && routes
            .by_scheduler
            .get(&route.scheduler_id)
            .is_some_and(|current| Arc::ptr_eq(current.value(), route))
}

#[allow(clippy::too_many_arguments)]
async fn run_output_dispatcher(
    mut output_rx: mpsc::Receiver<Vec<OutputSignal>>,
    routes: Routes,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    runtime: Handle,
    cancel: CancellationToken,
    mut output_gate: Option<watch::Receiver<bool>>,
) {
    loop {
        let output_enabled = output_gate.as_ref().is_none_or(|gate| *gate.borrow());
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            batch = output_rx.recv(), if output_enabled => {
                let Some(batch) = batch else { break };
                if !dispatch_output_batch(
                    batch,
                    &routes,
                    &runtime,
                    &cancellation_tx,
                    &cancel,
                ).await {
                    break;
                }
            }
            open = wait_for_output_gate(&mut output_gate), if output_gate.is_some() => {
                if !open {
                    output_gate = None;
                }
            }
        }
    }

    let active_routes = routes
        .by_client
        .iter()
        .map(|entry| Arc::clone(entry.value()))
        .collect::<Vec<_>>();
    for route in active_routes {
        route.shutdown();
    }
    routes.by_client.clear();
    routes.by_scheduler.clear();
}

async fn wait_for_output_gate(gate: &mut Option<watch::Receiver<bool>>) -> bool {
    let Some(gate) = gate else {
        std::future::pending::<()>().await;
        return false;
    };
    gate.changed().await.is_ok()
}

async fn dispatch_output_batch(
    batch: Vec<OutputSignal>,
    routes: &Routes,
    runtime: &Handle,
    cancellation_tx: &mpsc::Sender<SchedulerCancellationEnvelope>,
    cancel: &CancellationToken,
) -> bool {
    for mut signal in batch {
        let scheduler_id = signal.uuid;
        let terminal = signal.completed;
        let Some(route) = routes
            .by_scheduler
            .get(&scheduler_id)
            .map(|entry| Arc::clone(entry.value()))
        else {
            continue;
        };

        signal.uuid = route.client_id;
        let delivery = tokio::select! {
            biased;
            _ = cancel.cancelled() => return false,
            delivery = route.send_output(signal) => delivery,
        };
        if delivery == OutputDelivery::Closed {
            route.abandon_stream();
            drop(spawn_cancellation(
                runtime,
                cancellation_tx.clone(),
                Arc::clone(routes),
                Arc::clone(&route),
                true,
            ));
        }
        if terminal && route.observe_terminal() {
            remove_route(routes, &route);
        }
    }
    true
}

fn spawn_cancellation(
    runtime: &Handle,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    routes: Routes,
    route: Arc<RequestRoute>,
    abandon_stream: bool,
) -> tokio::task::JoinHandle<anyhow::Result<bool>> {
    runtime.spawn(async move {
        if !route.wait_for_admission().await {
            return Ok(false);
        }

        let _cancel_guard = route.cancel_lock.lock().await;
        if !route_is_registered(&routes, &route) {
            return Ok(false);
        }
        if abandon_stream {
            route.abandon_stream();
        }
        if !route.begin_cancellation() {
            return Ok(false);
        }

        let result = cancel_request(
            &cancellation_tx,
            route.scheduler_id,
            abandon_stream,
        )
        .await;
        if route.finish_cancellation(&result) {
            remove_route(&routes, &route);
        }
        if let Err(error) = &result {
            tracing::debug!(client_id = %route.client_id, scheduler_id = %route.scheduler_id, %error, "live Mocker request cancellation failed");
        }
        result
    })
}

async fn await_cancellation(
    cancellation: tokio::task::JoinHandle<anyhow::Result<bool>>,
) -> anyhow::Result<bool> {
    match cancellation.await {
        Ok(result) => result,
        Err(error) => Err(anyhow!("live Mocker cancellation task failed: {error}")),
    }
}

async fn cancel_request(
    cancellation_tx: &mpsc::Sender<SchedulerCancellationEnvelope>,
    request_id: Uuid,
    discard_pending_output: bool,
) -> anyhow::Result<bool> {
    let (reply, response) = oneshot::channel();
    cancellation_tx
        .send(SchedulerCancellationEnvelope {
            request_id,
            discard_pending_output,
            reply,
        })
        .await
        .map_err(|_| anyhow!("Mocker scheduler is not accepting cancellations"))?;
    let effects = response
        .await
        .map_err(|_| anyhow!("Mocker scheduler dropped a cancellation acknowledgement"))??;
    match effects.result {
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
mod tests;
