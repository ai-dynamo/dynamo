// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable live-request boundary for the Mocker schedulers.
//!
//! This module owns the common submit, output-demultiplexing, and cancellation
//! mechanics needed by network-facing mock engines. Submission admission is an
//! owned operation, while cancellation uses a separate bounded scheduler lane
//! that remains responsive during a long modeled pass.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, Weak};

use anyhow::{Context, anyhow, bail};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::engine::{LiveEngineScheduler, create_engine_with_event_sender};
use crate::scheduler::{
    LiveEngineEvent, MockerMetrics, SchedulerCancellationEnvelope, SchedulerCommand,
    SchedulerCommandEnvelope, SchedulerCommandResult, SchedulerEventSender, SchedulerHandle,
};

#[derive(Default)]
struct RequestRoutes {
    by_client: DashMap<Uuid, Arc<RequestRoute>>,
    by_scheduler: DashMap<Uuid, Arc<RequestRoute>>,
}

type Routes = Arc<RequestRoutes>;

const SCHEDULER_EVENT_CAPACITY: usize = 8;
const DEFAULT_REQUEST_OUTPUT_CAPACITY: usize = 8;

pub(crate) struct ObservedAdmission {
    pub(crate) event: crate::scheduler::AdmissionEvent,
    pub(crate) observed_at: tokio::time::Instant,
}

pub(crate) struct ObservedOutput {
    pub(crate) event: OutputSignal,
    pub(crate) observed_at: tokio::time::Instant,
}

pub(crate) struct LiveEngineOptions {
    pub(crate) kv_event_publishers: KvEventPublishers,
    pub(crate) admission_tx: Option<mpsc::UnboundedSender<ObservedAdmission>>,
    pub(crate) fpm_publisher: FpmPublisher,
    pub(crate) request_output_capacity: Option<NonZeroUsize>,
    pub(crate) allow_zero_output: bool,
}

impl Default for LiveEngineOptions {
    fn default() -> Self {
        Self {
            kv_event_publishers: KvEventPublishers::default(),
            admission_tx: None,
            fpm_publisher: FpmPublisher::default(),
            request_output_capacity: NonZeroUsize::new(DEFAULT_REQUEST_OUTPUT_CAPACITY),
            allow_zero_output: false,
        }
    }
}

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
    output_tx: Mutex<Option<mpsc::Sender<ObservedOutput>>>,
    lifecycle_tx: watch::Sender<RequestLifecycle>,
    cancel_lock: tokio::sync::Mutex<()>,
}

impl RequestRoute {
    fn new(client_id: Uuid, scheduler_id: Uuid, output_tx: mpsc::Sender<ObservedOutput>) -> Self {
        let (lifecycle_tx, _) = watch::channel(RequestLifecycle {
            state: RequestState::Submitting,
            stream_abandoned: false,
            terminal_seen: false,
        });
        Self {
            client_id,
            scheduler_id,
            output_tx: Mutex::new(Some(output_tx)),
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

    fn abandon_stream(&self) -> bool {
        self.close_output();
        let mut abandoned = false;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.stream_abandoned {
                return false;
            }
            lifecycle.stream_abandoned = true;
            abandoned = true;
            true
        });
        abandoned
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

    fn send_output(&self, output: ObservedOutput) -> OutputDelivery {
        let output_tx = self.output_tx.lock().unwrap().as_ref().cloned();
        let Some(output_tx) = output_tx else {
            return OutputDelivery::Closed;
        };
        match output_tx.try_send(output) {
            Ok(()) => OutputDelivery::Delivered,
            Err(mpsc::error::TrySendError::Full(_)) => OutputDelivery::Full,
            Err(mpsc::error::TrySendError::Closed(_)) => OutputDelivery::Closed,
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
        self.output_tx.lock().unwrap().take();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputDelivery {
    Delivered,
    Full,
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
    request_output_capacity: Option<NonZeroUsize>,
    allow_zero_output: bool,
    cancel: CancellationToken,
    runtime: Handle,
    tasks: Mutex<LiveEngineTasks>,
    // The scheduler's drop guard owns its task lifetime.
    #[allow(dead_code)]
    scheduler: Box<dyn SchedulerHandle>,
}

struct LiveEngineTasks {
    scheduler_actor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    dispatcher_supervisor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
}

impl LiveEngine {
    /// Start one live scheduler at `dp_rank`.
    pub fn start(args: MockEngineArgs, dp_rank: u32) -> anyhow::Result<Self> {
        Self::start_internal(args, dp_rank, LiveEngineOptions::default(), None)
    }

    pub(crate) fn start_with_options(
        args: MockEngineArgs,
        dp_rank: u32,
        options: LiveEngineOptions,
    ) -> anyhow::Result<Self> {
        Self::start_internal(args, dp_rank, options, None)
    }

    pub(crate) fn start_with_options_and_output_gate(
        args: MockEngineArgs,
        dp_rank: u32,
        options: LiveEngineOptions,
        output_gate: watch::Receiver<bool>,
    ) -> anyhow::Result<Self> {
        Self::start_internal(args, dp_rank, options, Some(output_gate))
    }

    #[cfg(test)]
    fn start_with_output_gate(
        args: MockEngineArgs,
        dp_rank: u32,
        output_gate: Option<watch::Receiver<bool>>,
        request_output_capacity: usize,
    ) -> anyhow::Result<Self> {
        let request_output_capacity = NonZeroUsize::new(request_output_capacity)
            .ok_or_else(|| anyhow!("request output capacity must be greater than 0"))?;
        Self::start_internal(
            args,
            dp_rank,
            LiveEngineOptions {
                request_output_capacity: Some(request_output_capacity),
                ..LiveEngineOptions::default()
            },
            output_gate,
        )
    }

    fn start_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        options: LiveEngineOptions,
        output_gate: Option<watch::Receiver<bool>>,
    ) -> anyhow::Result<Self> {
        let runtime =
            Handle::try_current().context("LiveEngine::start requires an active Tokio runtime")?;
        let args = args
            .normalized()
            .context("invalid Mocker engine arguments")?;
        let cancel = CancellationToken::new();
        let (event_tx, event_rx) = mpsc::channel::<LiveEngineEvent>(SCHEDULER_EVENT_CAPACITY);
        let LiveEngineScheduler {
            handle: scheduler,
            actor: scheduler_actor,
        } = create_engine_with_event_sender(
            args,
            dp_rank,
            Some(SchedulerEventSender::Ordered(event_tx)),
            options.kv_event_publishers,
            Some(cancel.clone()),
            options.fpm_publisher,
        );
        let command_tx = scheduler.command_sender();
        let cancellation_tx = scheduler.cancellation_sender();
        let metrics_rx = scheduler.metrics_receiver();
        let routes = Arc::new(RequestRoutes::default());
        let dispatcher = runtime.spawn(run_event_dispatcher(
            event_rx,
            Arc::clone(&routes),
            cancellation_tx.clone(),
            runtime.clone(),
            cancel.clone(),
            output_gate,
            options.admission_tx,
        ));
        let dispatcher_supervisor = runtime.spawn(supervise_event_dispatcher(
            dispatcher,
            Arc::clone(&routes),
            cancel.clone(),
        ));

        Ok(Self {
            inner: Arc::new(LiveEngineInner {
                command_tx,
                cancellation_tx,
                routes,
                metrics_rx,
                request_output_capacity: options.request_output_capacity,
                allow_zero_output: options.allow_zero_output,
                cancel,
                runtime,
                tasks: Mutex::new(LiveEngineTasks {
                    scheduler_actor: Some(scheduler_actor),
                    dispatcher_supervisor: Some(dispatcher_supervisor),
                }),
                scheduler,
            }),
        })
    }

    /// Submit a request and return its scoped output receiver.
    pub async fn submit(&self, mut request: DirectRequest) -> anyhow::Result<LiveRequest> {
        anyhow::ensure!(
            !self.inner.cancel.is_cancelled(),
            "live Mocker engine is not running"
        );
        // Both scheduler cores treat an explicit token plan as authoritative.
        // Normalize the effective length while keeping delivery buffering
        // independent of a caller's declared maximum response length.
        let output_length = request
            .output_token_ids
            .as_ref()
            .map_or(request.max_output_tokens, Vec::len);
        anyhow::ensure!(
            self.inner.allow_zero_output || output_length > 0,
            "live requests must generate at least one output token"
        );
        request.max_output_tokens = output_length;
        let client_id = request.uuid.unwrap_or_else(Uuid::new_v4);
        let scheduler_id = Uuid::new_v4();
        request.uuid = Some(scheduler_id);
        // The dispatcher never waits on a request stream. A caller that leaves
        // this fixed queue full is cancelled so it cannot block unrelated
        // requests or turn its declared response length into admission policy.
        let output_capacity = self.inner.request_output_capacity.map_or_else(
            || output_length.max(1),
            |capacity| output_length.max(1).min(capacity.get()),
        );
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
        if self.inner.cancel.is_cancelled() {
            route.shutdown();
            remove_route(&self.inner.routes, &route);
            bail!("live Mocker engine stopped during submission");
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

    pub(crate) async fn shutdown(&self) -> anyhow::Result<()> {
        self.inner.cancel.cancel();
        shutdown_routes(&self.inner.routes);
        let (scheduler_actor, dispatcher_supervisor) = {
            let mut tasks = self.inner.tasks.lock().unwrap();
            (
                tasks.scheduler_actor.take(),
                tasks.dispatcher_supervisor.take(),
            )
        };

        let mut first_error = None;
        if let Some(scheduler_actor) = scheduler_actor {
            match scheduler_actor.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => first_error = Some(error.context("live Mocker scheduler failed")),
                Err(error) => {
                    first_error = Some(anyhow!("live Mocker scheduler task failed: {error}"))
                }
            }
        }
        if let Some(dispatcher_supervisor) = dispatcher_supervisor {
            match dispatcher_supervisor.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) if first_error.is_none() => {
                    first_error = Some(error.context("live Mocker event dispatcher failed"))
                }
                Err(error) if first_error.is_none() => {
                    first_error = Some(anyhow!("live Mocker dispatcher supervisor failed: {error}"))
                }
                Ok(Err(_)) | Err(_) => {}
            }
        }
        shutdown_routes(&self.inner.routes);
        if let Some(error) = first_error {
            return Err(error);
        }
        anyhow::ensure!(
            self.inner.routes.by_client.is_empty() && self.inner.routes.by_scheduler.is_empty(),
            "live Mocker shutdown left active request routes"
        );
        Ok(())
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
    rx: mpsc::Receiver<ObservedOutput>,
    route: Weak<RequestRoute>,
    routes: Routes,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    runtime: Handle,
}

impl LiveRequest {
    pub fn id(&self) -> Uuid {
        self.client_id
    }

    pub async fn recv(&mut self) -> Option<OutputSignal> {
        self.recv_observed().await.map(|output| output.event)
    }

    pub(crate) async fn recv_observed(&mut self) -> Option<ObservedOutput> {
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
async fn run_event_dispatcher(
    mut event_rx: mpsc::Receiver<LiveEngineEvent>,
    routes: Routes,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    runtime: Handle,
    cancel: CancellationToken,
    mut output_gate: Option<watch::Receiver<bool>>,
    admission_tx: Option<mpsc::UnboundedSender<ObservedAdmission>>,
) -> anyhow::Result<()> {
    let mut pending_event = None;
    loop {
        if cancel.is_cancelled() {
            drop(pending_event.take());
            while event_rx.recv().await.is_some() {}
            return Ok(());
        }

        if matches!(pending_event.as_ref(), Some(LiveEngineEvent::Outputs(_)))
            && output_gate.as_ref().is_some_and(|gate| !*gate.borrow())
        {
            let Some(gate) = output_gate.as_mut() else {
                unreachable!("the output gate was checked above");
            };
            tokio::select! {
                biased;
                _ = cancel.cancelled() => continue,
                changed = gate.changed() => {
                    if changed.is_err() {
                        bail!("live Mocker output gate closed");
                    }
                }
            }
            continue;
        }

        let event = if let Some(event) = pending_event.take() {
            event
        } else {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => continue,
                event = event_rx.recv() => {
                    let Some(event) = event else {
                        if cancel.is_cancelled() {
                            return Ok(());
                        }
                        bail!("live Mocker ordered event lane closed unexpectedly");
                    };
                    event
                }
            }
        };

        match event {
            LiveEngineEvent::Admissions(batch) => {
                dispatch_admission_batch(batch, &routes, admission_tx.as_ref())?;
            }
            LiveEngineEvent::Outputs(batch)
                if output_gate.as_ref().is_some_and(|gate| !*gate.borrow()) =>
            {
                pending_event = Some(LiveEngineEvent::Outputs(batch));
            }
            LiveEngineEvent::Outputs(batch) => {
                if !dispatch_output_batch(batch, &routes, &runtime, &cancellation_tx, &cancel) {
                    return Ok(());
                }
            }
        }
    }
}

fn dispatch_admission_batch(
    batch: Vec<crate::scheduler::AdmissionEvent>,
    routes: &Routes,
    admission_tx: Option<&mpsc::UnboundedSender<ObservedAdmission>>,
) -> anyhow::Result<()> {
    let Some(admission_tx) = admission_tx else {
        return Ok(());
    };
    let observed_at = tokio::time::Instant::now();
    for mut admission in batch {
        let scheduler_id = admission.uuid;
        let Some(route) = routes
            .by_scheduler
            .get(&scheduler_id)
            .map(|entry| Arc::clone(entry.value()))
        else {
            continue;
        };
        admission.uuid = route.client_id;
        admission_tx
            .send(ObservedAdmission {
                event: admission,
                observed_at,
            })
            .map_err(|_| anyhow!("live Mocker admission receiver closed"))?;
    }
    Ok(())
}

async fn supervise_event_dispatcher(
    dispatcher: tokio::task::JoinHandle<anyhow::Result<()>>,
    routes: Routes,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let result = match dispatcher.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error),
        Err(error) => Err(anyhow!("live Mocker event dispatcher task failed: {error}")),
    };
    if let Err(error) = &result {
        tracing::error!(%error, "live Mocker event dispatcher failed");
    } else if !cancel.is_cancelled() {
        tracing::error!("live Mocker event dispatcher exited unexpectedly");
    }
    cancel.cancel();
    shutdown_routes(&routes);
    result
}

fn shutdown_routes(routes: &RequestRoutes) {
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

fn dispatch_output_batch(
    batch: Vec<OutputSignal>,
    routes: &Routes,
    runtime: &Handle,
    cancellation_tx: &mpsc::Sender<SchedulerCancellationEnvelope>,
    cancel: &CancellationToken,
) -> bool {
    let observed_at = tokio::time::Instant::now();
    for mut signal in batch {
        if cancel.is_cancelled() {
            return false;
        }
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
        let delivery = route.send_output(ObservedOutput {
            event: signal,
            observed_at,
        });
        if delivery != OutputDelivery::Delivered && route.abandon_stream() {
            if delivery == OutputDelivery::Full {
                tracing::debug!(
                    client_id = %route.client_id,
                    scheduler_id = %route.scheduler_id,
                    "cancelling live Mocker request with a full output stream"
                );
            }
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
