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
use dashmap::mapref::entry::Entry;
use futures::future::{BoxFuture, FutureExt, Shared};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::handoff::HandoffId;
use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::engine::{LiveEngineScheduler, create_engine_with_event_sender};
use crate::scheduler::{
    LiveEngineEvent, MockerMetrics, SchedulerCancellationEnvelope, SchedulerCommand,
    SchedulerCommandEnvelope, SchedulerCommandResult, SchedulerEventSender, SchedulerHandle,
};

mod handoff;
mod request;

pub use handoff::{LiveHandoffControl, LiveHandoffEvent, LiveHandoffEvents};

use handoff::{
    DestinationCancellation, HandoffRoutes, SharedHandoffRoutes, run_lifecycle_dispatcher,
    shutdown_handoff_routes, supervise_lifecycle_dispatcher,
};
use request::{
    ObservedOutput, OutputDelivery, RequestCancellation, RequestRoute, RequestRoutes, Routes,
    remove_route, route_is_registered, shutdown_routes,
};

const SCHEDULER_EVENT_CAPACITY: usize = 8;
const DEFAULT_REQUEST_OUTPUT_CAPACITY: usize = 8;

/// Runtime publishers used by one live Mocker scheduler.
#[derive(Clone)]
pub struct LiveEngineConfig {
    pub kv_event_publishers: KvEventPublishers,
    pub fpm_publisher: FpmPublisher,
    /// Maximum number of scheduler signals buffered for each request.
    ///
    /// `None` sizes the buffer for the request's complete output. This is useful
    /// for exact replay workloads whose scheduler may emit the whole response
    /// before the consumer gets a chance to drain it.
    pub request_output_capacity: Option<NonZeroUsize>,
}

impl Default for LiveEngineConfig {
    fn default() -> Self {
        Self {
            kv_event_publishers: KvEventPublishers::default(),
            fpm_publisher: FpmPublisher::default(),
            request_output_capacity: NonZeroUsize::new(DEFAULT_REQUEST_OUTPUT_CAPACITY),
        }
    }
}

pub(crate) struct ObservedAdmission {
    pub(crate) event: crate::scheduler::AdmissionEvent,
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

/// Map a wire-protocol request ID to a deterministic scheduler UUID.
pub fn stable_request_uuid(seed: u64, request_id: &str) -> Uuid {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(request_id.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    // Mark the digest as an RFC 4122 variant/version-4 UUID. The identifier
    // remains deterministic; these bits only make diagnostics parse cleanly.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

/// Produce deterministic, tokenizer-independent output token IDs.
pub fn deterministic_output_tokens(seed: u64, request_id: &str, count: usize) -> Vec<u32> {
    (0..count)
        .map(|position| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&seed.to_le_bytes());
            hasher.update(request_id.as_bytes());
            hasher.update(&(position as u64).to_le_bytes());
            let bytes = hasher.finalize();
            1_000 + (u32::from_le_bytes(bytes.as_bytes()[..4].try_into().unwrap()) % 31_000)
        })
        .collect()
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
    handoff_routes: SharedHandoffRoutes,
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
    lifecycle_supervisor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    shutdown: Option<SharedShutdown>,
}

type SharedShutdown = Shared<BoxFuture<'static, Result<(), Arc<str>>>>;

impl LiveEngine {
    /// Start one live scheduler at `dp_rank`.
    pub fn start(args: MockEngineArgs, dp_rank: u32) -> anyhow::Result<Self> {
        Self::start_internal(args, dp_rank, LiveEngineOptions::default(), None)
    }

    /// Start one live scheduler with runtime-owned KV and FPM publishers.
    pub fn start_with_config(
        args: MockEngineArgs,
        dp_rank: u32,
        config: LiveEngineConfig,
    ) -> anyhow::Result<Self> {
        Self::start_internal(
            args,
            dp_rank,
            LiveEngineOptions {
                kv_event_publishers: config.kv_event_publishers,
                fpm_publisher: config.fpm_publisher,
                request_output_capacity: config.request_output_capacity,
                ..LiveEngineOptions::default()
            },
            None,
        )
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
        let forward_admissions = options.admission_tx.is_some();
        let LiveEngineScheduler {
            handle: mut scheduler,
            actor: scheduler_actor,
        } = create_engine_with_event_sender(
            args,
            dp_rank,
            Some(SchedulerEventSender::Ordered {
                tx: event_tx,
                forward_admissions,
            }),
            options.kv_event_publishers,
            Some(cancel.clone()),
            options.fpm_publisher,
        );
        let command_tx = scheduler.command_sender();
        let cancellation_tx = scheduler.cancellation_sender();
        let metrics_rx = scheduler.metrics_receiver();
        let lifecycle_rx = scheduler
            .take_lifecycle_receiver()
            .expect("new live scheduler must expose one lifecycle receiver");
        let routes = Arc::new(RequestRoutes::default());
        let handoff_routes = Arc::new(HandoffRoutes::default());
        let dispatcher = runtime.spawn(run_event_dispatcher(
            event_rx,
            Arc::clone(&routes),
            command_tx.clone(),
            cancellation_tx.clone(),
            runtime.clone(),
            cancel.clone(),
            output_gate,
            options.admission_tx,
        ));
        let dispatcher_supervisor = runtime.spawn(supervise_event_dispatcher(
            dispatcher,
            Arc::clone(&routes),
            Arc::clone(&handoff_routes),
            cancel.clone(),
        ));
        let lifecycle_dispatcher = runtime.spawn(run_lifecycle_dispatcher(
            lifecycle_rx,
            Arc::clone(&handoff_routes),
            cancel.clone(),
        ));
        let lifecycle_supervisor = runtime.spawn(supervise_lifecycle_dispatcher(
            lifecycle_dispatcher,
            Arc::clone(&routes),
            Arc::clone(&handoff_routes),
            cancel.clone(),
        ));

        Ok(Self {
            inner: Arc::new(LiveEngineInner {
                command_tx,
                cancellation_tx,
                routes,
                handoff_routes,
                metrics_rx,
                request_output_capacity: options.request_output_capacity,
                allow_zero_output: options.allow_zero_output,
                cancel,
                runtime,
                tasks: Mutex::new(LiveEngineTasks {
                    scheduler_actor: Some(scheduler_actor),
                    dispatcher_supervisor: Some(dispatcher_supervisor),
                    lifecycle_supervisor: Some(lifecycle_supervisor),
                    shutdown: None,
                }),
                scheduler,
            }),
        })
    }

    /// Register a request route without submitting it to the scheduler.
    ///
    /// Disaggregated handoff commands consume the returned registration after
    /// their bootstrap session is ready. Dropping an unused registration
    /// removes the route and closes the paired response stream.
    pub fn prepare_request(
        &self,
        mut request: DirectRequest,
    ) -> anyhow::Result<(LiveRequestRegistration, LiveRequest)> {
        anyhow::ensure!(
            !self.inner.cancel.is_cancelled(),
            "live Mocker engine is not running"
        );
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
        if self.inner.cancel.is_cancelled() {
            route.shutdown();
            remove_route(&self.inner.routes, &route);
            bail!("live Mocker engine is not running");
        }

        let live = LiveRequest {
            client_id,
            rx,
            route: Arc::downgrade(&route),
            routes: Arc::clone(&self.inner.routes),
            command_tx: self.inner.command_tx.clone(),
            cancellation_tx: self.inner.cancellation_tx.clone(),
            runtime: self.inner.runtime.clone(),
        };
        let registration = LiveRequestRegistration {
            engine: Arc::downgrade(&self.inner),
            routes: Arc::clone(&self.inner.routes),
            prepared: Some(PreparedRequest { request, route }),
        };
        Ok((registration, live))
    }

    /// Submit a request and return its scoped output receiver.
    pub async fn submit(&self, request: DirectRequest) -> anyhow::Result<LiveRequest> {
        let (registration, live) = self.prepare_request(request)?;
        self.submit_prepared(registration, PreparedSubmission::Ordinary, None)
            .await?;
        Ok(live)
    }

    async fn submit_prepared(
        &self,
        mut registration: LiveRequestRegistration,
        submission: PreparedSubmission,
        command_guard: Option<tokio::sync::OwnedMutexGuard<()>>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.inner.cancel.is_cancelled(),
            "live Mocker engine is not running"
        );
        let PreparedRequest { request, route } = registration.take_for(&self.inner)?;
        let scheduler_id = route.scheduler_id;
        let client_id = route.client_id;
        let routes = Arc::clone(&self.inner.routes);
        let submission_route = Arc::clone(&route);
        let command_tx = self.inner.command_tx.clone();
        let task = self.inner.runtime.spawn(async move {
            let _command_guard = command_guard;
            let command = submission.command(request);
            let result = send_command(&command_tx, command).await;
            let admission = submission.validate(result, client_id, scheduler_id);
            if admission.is_ok() {
                submission_route.activate(submission.cancellation());
            } else {
                submission_route.shutdown();
                remove_route(&routes, &submission_route);
            }
            admission
        });
        match task.await {
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
        Ok(())
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
            self.inner.command_tx.clone(),
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

    pub async fn shutdown(&self) -> anyhow::Result<()> {
        self.inner.cancel.cancel();
        shutdown_routes(&self.inner.routes);
        shutdown_handoff_routes(&self.inner.handoff_routes);
        let shutdown = {
            let mut tasks = self.inner.tasks.lock().unwrap();
            if let Some(shutdown) = tasks.shutdown.as_ref() {
                shutdown.clone()
            } else {
                let shutdown = shutdown_engine(
                    tasks.scheduler_actor.take(),
                    tasks.dispatcher_supervisor.take(),
                    tasks.lifecycle_supervisor.take(),
                    Arc::clone(&self.inner.routes),
                    Arc::clone(&self.inner.handoff_routes),
                )
                .boxed()
                .shared();
                tasks.shutdown = Some(shutdown.clone());
                shutdown
            }
        };
        shutdown.await.map_err(|error| anyhow!("{error}"))
    }
}

async fn shutdown_engine(
    scheduler_actor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    dispatcher_supervisor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    lifecycle_supervisor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    routes: Routes,
    handoff_routes: SharedHandoffRoutes,
) -> Result<(), Arc<str>> {
    let mut first_error = None;
    if let Some(scheduler_actor) = scheduler_actor {
        match scheduler_actor.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => first_error = Some(error.context("live Mocker scheduler failed")),
            Err(error) => first_error = Some(anyhow!("live Mocker scheduler task failed: {error}")),
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
    if let Some(lifecycle_supervisor) = lifecycle_supervisor {
        match lifecycle_supervisor.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) if first_error.is_none() => {
                first_error = Some(error.context("live Mocker lifecycle dispatcher failed"))
            }
            Err(error) if first_error.is_none() => {
                first_error = Some(anyhow!(
                    "live Mocker lifecycle dispatcher supervisor failed: {error}"
                ))
            }
            Ok(Err(_)) | Err(_) => {}
        }
    }
    shutdown_routes(&routes);
    shutdown_handoff_routes(&handoff_routes);
    if let Some(error) = first_error {
        return Err(Arc::from(format!("{error:#}")));
    }
    if !routes.by_client.is_empty() || !routes.by_scheduler.is_empty() {
        return Err(Arc::from("live Mocker shutdown left active request routes"));
    }
    if !handoff_routes.is_empty() {
        return Err(Arc::from("live Mocker shutdown left active handoff routes"));
    }
    Ok(())
}

impl Drop for LiveEngineInner {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

struct PreparedRequest {
    request: DirectRequest,
    route: Arc<RequestRoute>,
}

/// An output route prepared for ordinary or disaggregated scheduler admission.
pub struct LiveRequestRegistration {
    engine: Weak<LiveEngineInner>,
    routes: Routes,
    prepared: Option<PreparedRequest>,
}

impl LiveRequestRegistration {
    fn take_for(&mut self, engine: &Arc<LiveEngineInner>) -> anyhow::Result<PreparedRequest> {
        let Some(owner) = self.engine.upgrade() else {
            bail!("live Mocker engine no longer exists");
        };
        anyhow::ensure!(
            Arc::ptr_eq(&owner, engine),
            "prepared request belongs to a different live Mocker engine"
        );
        self.prepared
            .take()
            .ok_or_else(|| anyhow!("prepared request was already consumed"))
    }
}

impl Drop for LiveRequestRegistration {
    fn drop(&mut self) {
        if let Some(prepared) = self.prepared.take() {
            prepared.route.shutdown();
            remove_route(&self.routes, &prepared.route);
        }
    }
}

#[derive(Clone)]
enum PreparedSubmission {
    Ordinary,
    Source(HandoffId),
    Destination(DestinationCancellation),
}

impl PreparedSubmission {
    fn cancellation(&self) -> RequestCancellation {
        match self {
            Self::Destination(cancellation) => {
                RequestCancellation::Destination(cancellation.clone())
            }
            Self::Ordinary | Self::Source(_) => RequestCancellation::Request,
        }
    }

    fn command(&self, request: DirectRequest) -> SchedulerCommand {
        match self {
            Self::Ordinary => SchedulerCommand::Submit(request),
            Self::Source(handoff_id) => SchedulerCommand::SubmitHandoffPrefill {
                handoff_id: *handoff_id,
                request,
            },
            Self::Destination(cancellation) => SchedulerCommand::ReserveDestination {
                handoff_id: cancellation.handoff_id(),
                request,
            },
        }
    }

    fn validate(
        &self,
        result: anyhow::Result<SchedulerCommandResult>,
        client_id: Uuid,
        scheduler_id: Uuid,
    ) -> anyhow::Result<()> {
        match (self, result) {
            (
                Self::Ordinary | Self::Source(_),
                Ok(SchedulerCommandResult::Submitted(submitted)),
            ) if submitted == scheduler_id => Ok(()),
            (
                Self::Destination(_),
                Ok(SchedulerCommandResult::DestinationAccepted { request_id }),
            ) if request_id == scheduler_id => Ok(()),
            (_, Ok(result)) => Err(anyhow!(
                "unexpected scheduler submit result for {client_id}: {result:?}"
            )),
            (_, Err(error)) => Err(error),
        }
    }
}

/// Request-owned stream of Mocker output signals.
pub struct LiveRequest {
    client_id: Uuid,
    rx: mpsc::Receiver<ObservedOutput>,
    route: Weak<RequestRoute>,
    routes: Routes,
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
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
            self.command_tx.clone(),
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
            self.command_tx.clone(),
            self.cancellation_tx.clone(),
            Arc::clone(&self.routes),
            route,
            true,
        ));
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_event_dispatcher(
    mut event_rx: mpsc::Receiver<LiveEngineEvent>,
    routes: Routes,
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
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
                if !dispatch_output_batch(
                    batch,
                    &routes,
                    &runtime,
                    &command_tx,
                    &cancellation_tx,
                    &cancel,
                ) {
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
    handoff_routes: SharedHandoffRoutes,
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
    shutdown_handoff_routes(&handoff_routes);
    result
}

fn dispatch_output_batch(
    batch: Vec<OutputSignal>,
    routes: &Routes,
    runtime: &Handle,
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
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
                command_tx.clone(),
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
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
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
        let Some(cancellation) = route.begin_cancellation() else {
            return Ok(false);
        };

        let result = match cancellation {
            RequestCancellation::Request => {
                cancel_request(
                    &cancellation_tx,
                    route.scheduler_id,
                    abandon_stream,
                )
                .await
            }
            RequestCancellation::Destination(cancellation) => cancellation.cancel(&command_tx).await,
        };
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

async fn cancel_destination(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    handoff_id: HandoffId,
) -> anyhow::Result<bool> {
    match send_command(
        command_tx,
        SchedulerCommand::CancelDestination { handoff_id },
    )
    .await?
    {
        SchedulerCommandResult::Applied => Ok(true),
        SchedulerCommandResult::Noop => Ok(false),
        result => Err(anyhow!(
            "unexpected scheduler destination cancellation result for {handoff_id:?}: {result:?}"
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
