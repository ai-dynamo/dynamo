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
#[cfg(test)]
use futures::stream::{FuturesUnordered, StreamExt};
use tokio::runtime::Handle;
#[cfg(test)]
use tokio::sync::watch;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::handoff::HandoffId;
use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::engine::{LiveEngineScheduler, create_engine_with_rank_sink};
#[cfg(test)]
use crate::grouped_scheduler::CompletionBoundaryTestControl;
use crate::grouped_scheduler::{
    CompletionBoundaryDrain, GroupedSchedulers, RankOutputSink, RankSinks,
    create_grouped_scheduler_with_rank_sinks,
};
use crate::scheduler::{
    AdmissionEvent, MockerMetrics, SchedulerCancellationEnvelope, SchedulerCommand,
    SchedulerCommandEnvelope, SchedulerCommandResult, SchedulerHandle,
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

const DEFAULT_REQUEST_OUTPUT_CAPACITY: usize = 8;

/// Controls how much output one live request may retain before it is consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestOutputBuffering {
    /// Cancel the request when its output queue reaches `capacity`.
    CancelOnOverflow { capacity: NonZeroUsize },
    /// Buffer the request's full declared response.
    FullResponse,
}

impl Default for RequestOutputBuffering {
    fn default() -> Self {
        Self::CancelOnOverflow {
            capacity: NonZeroUsize::new(DEFAULT_REQUEST_OUTPUT_CAPACITY).unwrap(),
        }
    }
}

impl RequestOutputBuffering {
    fn capacity_for(self, output_length: usize) -> usize {
        let output_length = output_length.max(1);
        match self {
            Self::CancelOnOverflow { capacity } => output_length.min(capacity.get()),
            Self::FullResponse => output_length,
        }
    }
}

/// Runtime publishers used by one live Mocker scheduler.
#[derive(Clone, Default)]
pub struct LiveEngineConfig {
    pub kv_event_publishers: KvEventPublishers,
    pub fpm_publisher: FpmPublisher,
}

pub(crate) struct ObservedAdmission {
    pub(crate) event: crate::scheduler::AdmissionEvent,
    pub(crate) observed_at: tokio::time::Instant,
}

#[derive(Clone)]
pub(crate) struct LiveRouteDelivery {
    routes: Routes,
    cancel: CancellationToken,
    #[cfg(test)]
    output_gate: Option<watch::Receiver<bool>>,
    admission_tx: Option<mpsc::UnboundedSender<ObservedAdmission>>,
}

impl LiveRouteDelivery {
    pub(crate) fn wants_admissions(&self) -> bool {
        self.admission_tx.is_some()
    }

    pub(crate) fn publish_admissions(&self, admissions: Vec<AdmissionEvent>) -> anyhow::Result<()> {
        if self.cancel.is_cancelled() {
            return Ok(());
        }
        dispatch_admission_batch(admissions, &self.routes, self.admission_tx.as_ref())
    }

    pub(crate) async fn publish_outputs(
        &self,
        outputs: Vec<OutputSignal>,
    ) -> anyhow::Result<Option<Vec<Uuid>>> {
        #[cfg(test)]
        if let Some(gate) = self.output_gate.as_ref()
            && !*gate.borrow()
        {
            return self.publish_gated_outputs(outputs, gate).await;
        }
        Ok(dispatch_output_batch(outputs, &self.routes, &self.cancel))
    }

    #[cfg(test)]
    async fn publish_gated_outputs(
        &self,
        outputs: Vec<OutputSignal>,
        gate: &watch::Receiver<bool>,
    ) -> anyhow::Result<Option<Vec<Uuid>>> {
        let mut failed = Vec::new();
        let Some(mut pending) = self.partition_gated_outputs(outputs, &mut failed) else {
            return Ok(None);
        };
        if pending.is_empty() {
            return Ok(Some(failed));
        }

        let mut gate = gate.clone();
        loop {
            if *gate.borrow_and_update() {
                let Some(delivered) = dispatch_output_batch(pending, &self.routes, &self.cancel)
                else {
                    return Ok(None);
                };
                failed.extend(delivered);
                return Ok(Some(failed));
            }

            let route_bypasses = FuturesUnordered::new();
            for signal in &pending {
                if let Some(route) = self
                    .routes
                    .by_scheduler
                    .get(&signal.uuid)
                    .map(|entry| Arc::clone(entry.value()))
                {
                    route_bypasses.push(async move { route.wait_for_output_gate_bypass().await });
                }
            }
            tokio::pin!(route_bypasses);
            tokio::select! {
                biased;
                _ = self.cancel.cancelled() => return Ok(None),
                changed = gate.changed() => {
                    if changed.is_err() {
                        bail!("live Mocker output gate closed");
                    }
                }
                _ = route_bypasses.next() => {
                    let Some(still_pending) = self.partition_gated_outputs(pending, &mut failed)
                    else {
                        return Ok(None);
                    };
                    pending = still_pending;
                    if pending.is_empty() {
                        return Ok(Some(failed));
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn partition_gated_outputs(
        &self,
        outputs: Vec<OutputSignal>,
        failed: &mut Vec<Uuid>,
    ) -> Option<Vec<OutputSignal>> {
        let mut pending = Vec::with_capacity(outputs.len());
        for signal in outputs {
            let waits_for_gate = self
                .routes
                .by_scheduler
                .get(&signal.uuid)
                .is_some_and(|route| !route.output_gate_bypass_requested());
            if waits_for_gate {
                pending.push(signal);
                continue;
            }
            failed.extend(dispatch_output_batch(
                vec![signal],
                &self.routes,
                &self.cancel,
            )?);
        }
        Some(pending)
    }
}

#[derive(Default)]
pub(crate) struct LiveEngineOptions {
    pub(crate) kv_event_publishers: KvEventPublishers,
    pub(crate) admission_tx: Option<mpsc::UnboundedSender<ObservedAdmission>>,
    pub(crate) fpm_publisher: FpmPublisher,
    pub(crate) request_output_buffering: RequestOutputBuffering,
    pub(crate) allow_zero_output: bool,
    #[cfg(test)]
    pub(crate) output_gate: Option<watch::Receiver<bool>>,
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
    request_output_buffering: RequestOutputBuffering,
    allow_zero_output: bool,
    group: Arc<LiveEngineGroup>,
    cancel: CancellationToken,
    runtime: Handle,
    tasks: Mutex<LiveEngineTasks>,
    // The scheduler's drop guard owns its task lifetime.
    #[allow(dead_code)]
    scheduler: Box<dyn SchedulerHandle>,
}

struct LiveEngineTasks {
    lifecycle_supervisor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    shutdown: Option<SharedShutdown>,
}

type SharedShutdown = Shared<BoxFuture<'static, Result<(), Arc<str>>>>;

struct LiveEngineGroup {
    cancel: CancellationToken,
    actor: Mutex<Option<tokio::task::JoinHandle<anyhow::Result<()>>>>,
    shutdown: Mutex<Option<SharedShutdown>>,
    completion_drain: CompletionBoundaryDrain,
}

impl LiveEngineGroup {
    fn new(
        cancel: CancellationToken,
        actor: tokio::task::JoinHandle<anyhow::Result<()>>,
        completion_drain: CompletionBoundaryDrain,
    ) -> Self {
        Self {
            cancel,
            actor: Mutex::new(Some(actor)),
            shutdown: Mutex::new(None),
            completion_drain,
        }
    }

    fn shutdown(&self) -> SharedShutdown {
        self.cancel.cancel();
        let mut shutdown = self.shutdown.lock().unwrap();
        if let Some(shutdown) = shutdown.as_ref() {
            return shutdown.clone();
        }
        let actor = self.actor.lock().unwrap().take();
        let future = async move {
            let Some(actor) = actor else {
                return Ok(());
            };
            match actor.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(error)) => Err(Arc::from(format!(
                    "live Mocker scheduler failed: {error:#}"
                ))),
                Err(error) => Err(Arc::from(format!(
                    "live Mocker scheduler task failed: {error}"
                ))),
            }
        }
        .boxed()
        .shared();
        *shutdown = Some(future.clone());
        future
    }
}

impl Drop for LiveEngineGroup {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl LiveEngine {
    /// Start one live scheduler at `dp_rank`.
    pub fn start(args: MockEngineArgs, dp_rank: u32) -> anyhow::Result<Self> {
        Self::start_internal(args, dp_rank, LiveEngineOptions::default())
    }

    /// Start one live scheduler with runtime-owned KV and FPM publishers.
    pub fn start_with_config(
        args: MockEngineArgs,
        dp_rank: u32,
        config: LiveEngineConfig,
    ) -> anyhow::Result<Self> {
        Self::start_with_config_and_request_output_buffering(
            args,
            dp_rank,
            config,
            RequestOutputBuffering::default(),
        )
    }

    /// Start one live scheduler with runtime-owned publishers and an explicit
    /// per-request output buffering policy.
    pub fn start_with_config_and_request_output_buffering(
        args: MockEngineArgs,
        dp_rank: u32,
        config: LiveEngineConfig,
        request_output_buffering: RequestOutputBuffering,
    ) -> anyhow::Result<Self> {
        Self::start_internal(
            args,
            dp_rank,
            LiveEngineOptions {
                kv_event_publishers: config.kv_event_publishers,
                fpm_publisher: config.fpm_publisher,
                request_output_buffering,
                ..LiveEngineOptions::default()
            },
        )
    }

    /// Start all attention-DP ranks as one logical grouped engine.
    ///
    /// Each returned [`LiveEngine`] retains the latest-main rank-scoped request
    /// and handoff API, while all ranks share one scheduler actor and one
    /// [`aisimulate_core::engine::generalized::GeneralizedMockerEngine`] barrier.
    pub fn start_grouped_with_configs(
        args: MockEngineArgs,
        configs: Vec<LiveEngineConfig>,
    ) -> anyhow::Result<Vec<Self>> {
        Self::start_grouped_with_configs_and_request_output_buffering(
            args,
            configs,
            RequestOutputBuffering::default(),
        )
    }

    /// Start all attention-DP ranks with an explicit per-request output
    /// buffering policy.
    pub fn start_grouped_with_configs_and_request_output_buffering(
        args: MockEngineArgs,
        configs: Vec<LiveEngineConfig>,
        request_output_buffering: RequestOutputBuffering,
    ) -> anyhow::Result<Vec<Self>> {
        let options = configs
            .into_iter()
            .map(|config| LiveEngineOptions {
                kv_event_publishers: config.kv_event_publishers,
                fpm_publisher: config.fpm_publisher,
                request_output_buffering,
                ..LiveEngineOptions::default()
            })
            .collect();
        Self::start_grouped_with_options(args, options)
    }

    pub(crate) fn start_grouped_with_options(
        args: MockEngineArgs,
        options: Vec<LiveEngineOptions>,
    ) -> anyhow::Result<Vec<Self>> {
        let runtime = Handle::try_current()
            .context("LiveEngine::start_grouped_with_options requires an active Tokio runtime")?;
        let args = args
            .normalized()
            .context("invalid Mocker engine arguments")?;
        anyhow::ensure!(
            options.len() == args.dp_size as usize,
            "grouped live Mocker requires one options value per DP rank: expected {}, got {}",
            args.dp_size,
            options.len()
        );

        let cancel = CancellationToken::new();
        let mut route_sets = Vec::with_capacity(options.len());
        let mut rank_sinks = Vec::with_capacity(options.len());
        for options_for_rank in &options {
            let routes = Arc::new(RequestRoutes::default());
            rank_sinks.push(RankSinks {
                output: RankOutputSink::Routes(LiveRouteDelivery {
                    routes: Arc::clone(&routes),
                    cancel: cancel.clone(),
                    #[cfg(test)]
                    output_gate: options_for_rank.output_gate.clone(),
                    admission_tx: options_for_rank.admission_tx.clone(),
                }),
                kv_event_publishers: options_for_rank.kv_event_publishers.clone(),
                fpm_publisher: options_for_rank.fpm_publisher.clone(),
            });
            route_sets.push(routes);
        }

        let GroupedSchedulers {
            schedulers,
            actor,
            completion_drain,
        } = create_grouped_scheduler_with_rank_sinks(args, rank_sinks, Some(cancel.clone()))?;
        let group = Arc::new(LiveEngineGroup::new(cancel, actor, completion_drain));
        schedulers
            .into_iter()
            .zip(route_sets)
            .zip(options)
            .map(|((scheduler, routes), options)| {
                Self::from_scheduler(
                    runtime.clone(),
                    scheduler,
                    Arc::clone(&group),
                    routes,
                    options,
                )
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn start_with_options(
        args: MockEngineArgs,
        dp_rank: u32,
        options: LiveEngineOptions,
    ) -> anyhow::Result<Self> {
        Self::start_internal(args, dp_rank, options)
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
                request_output_buffering: RequestOutputBuffering::CancelOnOverflow {
                    capacity: request_output_capacity,
                },
                output_gate,
                ..LiveEngineOptions::default()
            },
        )
    }

    fn start_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        options: LiveEngineOptions,
    ) -> anyhow::Result<Self> {
        let runtime =
            Handle::try_current().context("LiveEngine::start requires an active Tokio runtime")?;
        let args = args
            .normalized()
            .context("invalid Mocker engine arguments")?;
        let group_cancel = CancellationToken::new();
        let routes = Arc::new(RequestRoutes::default());
        let LiveEngineScheduler {
            handle: scheduler,
            actor: scheduler_actor,
            completion_drain,
        } = create_engine_with_rank_sink(
            args,
            dp_rank,
            RankSinks {
                output: RankOutputSink::Routes(LiveRouteDelivery {
                    routes: Arc::clone(&routes),
                    cancel: group_cancel.clone(),
                    #[cfg(test)]
                    output_gate: options.output_gate.clone(),
                    admission_tx: options.admission_tx.clone(),
                }),
                kv_event_publishers: options.kv_event_publishers.clone(),
                fpm_publisher: options.fpm_publisher.clone(),
            },
            Some(group_cancel.clone()),
        )?;
        let group = Arc::new(LiveEngineGroup::new(
            group_cancel,
            scheduler_actor,
            completion_drain,
        ));
        Self::from_scheduler(runtime, scheduler, group, routes, options)
    }

    fn from_scheduler(
        runtime: Handle,
        mut scheduler: Box<dyn SchedulerHandle>,
        group: Arc<LiveEngineGroup>,
        routes: Routes,
        options: LiveEngineOptions,
    ) -> anyhow::Result<Self> {
        let cancel = group.cancel.child_token();
        let command_tx = scheduler.command_sender();
        let cancellation_tx = scheduler.cancellation_sender();
        let metrics_rx = scheduler.metrics_receiver();
        let lifecycle_rx = scheduler
            .take_lifecycle_receiver()
            .expect("new live scheduler must expose one lifecycle receiver");
        let handoff_routes = Arc::new(HandoffRoutes::default());
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
                request_output_buffering: options.request_output_buffering,
                allow_zero_output: options.allow_zero_output,
                group,
                cancel,
                runtime,
                tasks: Mutex::new(LiveEngineTasks {
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
        let output_length = request.effective_max_output_tokens();
        anyhow::ensure!(
            self.inner.allow_zero_output || output_length > 0,
            "live requests must generate at least one output token"
        );
        request.max_output_tokens = output_length;
        let client_id = request.uuid.unwrap_or_else(Uuid::new_v4);
        let scheduler_id = Uuid::new_v4();
        request.uuid = Some(scheduler_id);
        let output_capacity = self
            .inner
            .request_output_buffering
            .capacity_for(output_length);
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
            #[cfg(test)]
            drop_bypasses_output_gate: true,
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

    pub(crate) async fn drain_completion_boundary(&self) -> anyhow::Result<()> {
        self.inner.group.completion_drain.wait().await
    }

    #[cfg(test)]
    pub(crate) fn pause_completion_boundary_before_finish(&self) -> CompletionBoundaryTestControl {
        self.inner.group.completion_drain.pause_before_finish()
    }

    #[cfg(test)]
    pub(crate) fn group_is_cancelled(&self) -> bool {
        self.inner.group.cancel.is_cancelled()
    }

    pub async fn shutdown(&self) -> anyhow::Result<()> {
        let group_shutdown = self.inner.group.shutdown();
        self.inner.cancel.cancel();
        shutdown_routes(&self.inner.routes);
        shutdown_handoff_routes(&self.inner.handoff_routes);
        let shutdown = {
            let mut tasks = self.inner.tasks.lock().unwrap();
            if let Some(shutdown) = tasks.shutdown.as_ref() {
                shutdown.clone()
            } else {
                let shutdown = shutdown_engine(
                    group_shutdown,
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
    group_shutdown: SharedShutdown,
    lifecycle_supervisor: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    routes: Routes,
    handoff_routes: SharedHandoffRoutes,
) -> Result<(), Arc<str>> {
    let mut first_error = group_shutdown.await.err().map(|error| anyhow!("{error}"));
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
    #[cfg(test)]
    drop_bypasses_output_gate: bool,
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
        #[cfg(test)]
        let request = {
            let mut request = self;
            request.drop_bypasses_output_gate = false;
            request
        };
        #[cfg(not(test))]
        let request = self;
        let Some(route) = request.route.upgrade() else {
            return Ok(false);
        };
        route.abandon_stream();
        await_cancellation(spawn_cancellation(
            &request.runtime,
            request.command_tx.clone(),
            request.cancellation_tx.clone(),
            Arc::clone(&request.routes),
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
        #[cfg(test)]
        if self.drop_bypasses_output_gate {
            route.request_output_gate_bypass();
        }
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

fn dispatch_output_batch(
    batch: Vec<OutputSignal>,
    routes: &Routes,
    cancel: &CancellationToken,
) -> Option<Vec<Uuid>> {
    let observed_at = tokio::time::Instant::now();
    let mut failed = Vec::new();
    for mut signal in batch {
        if cancel.is_cancelled() {
            return None;
        }
        let scheduler_id = signal.uuid;
        let terminal = signal.completed;
        let Some(route) = routes
            .by_scheduler
            .get(&scheduler_id)
            .map(|entry| Arc::clone(entry.value()))
        else {
            failed.push(scheduler_id);
            continue;
        };

        signal.uuid = route.client_id;
        let delivery = route.send_output(ObservedOutput {
            event: signal,
            observed_at,
        });
        match delivery {
            OutputDelivery::Delivered => {
                if terminal && route.observe_terminal() {
                    remove_route(routes, &route);
                }
                continue;
            }
            OutputDelivery::Full => {
                let newly_abandoned = route.abandon_stream();
                if newly_abandoned {
                    tracing::debug!(
                        client_id = %route.client_id,
                        scheduler_id = %route.scheduler_id,
                        "cancelling live Mocker request with a full output stream"
                    );
                }
            }
            OutputDelivery::Closed => {
                route.abandon_stream();
            }
        }
        // Retire the route before the grouped completion dispatcher cancels
        // the native request so a replacement cannot receive stale output.
        route.shutdown();
        remove_route(routes, &route);
        failed.push(scheduler_id);
    }
    Some(failed)
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
