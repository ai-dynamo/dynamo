// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility facade from Dynamo's live scheduler handles to one grouped
//! AISimulate engine.

use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroU32;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
use aisimulate_engine::KvEventData;
use aisimulate_engine::generalized::{
    EngineEffects, EngineIdentity, EnginePassCompleted, GeneralizedMockerEngine, RankIdentity,
    SchedulerCommand as EngineSchedulerCommand,
};
use aisimulate_engine::{
    Admission, Command, CommandEffects, CommandResult, ForwardPassMetrics, KvEvent, Metrics,
    PassCompletionEffects,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use dynamo_kv_router::protocols::StorageTier;
#[cfg(test)]
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use futures::stream::{FuturesUnordered, StreamExt};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[cfg(test)]
use crate::common::protocols::ForwardPassSnapshot;
use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal, RawKvEvent,
};
use crate::engine_adapter::{EngineComponents, engine_components, engine_factory};
use crate::engine_observations::{dynamo_forward_pass_snapshot, dynamo_kv_event};
use crate::generalized_live::{
    GroupedLiveDriverConfig, GroupedLiveEngineHandle, GroupedLiveEvent, GroupedLiveRuntime,
    GroupedPassBoundary, spawn_grouped_live_engine,
};
use crate::scheduler::{
    AdmissionEvent, MockerMetrics, SchedulerCancellationEnvelope, SchedulerCommand,
    SchedulerCommandEffects, SchedulerCommandEnvelope, SchedulerCommandResult,
    SchedulerEventSendError, SchedulerEventSender, SchedulerHandle, SchedulerLifecycleEvent,
    handoff_channel_capacity,
};

/// Per-rank Dynamo sinks consumed by [`create_grouped_scheduler`].
#[derive(Clone, Default)]
pub struct GroupedSchedulerRankSinks {
    pub output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
    pub kv_event_publishers: KvEventPublishers,
    pub fpm_publisher: FpmPublisher,
}

/// One logical grouped engine exposed through the existing per-rank scheduler
/// handle contract.
pub struct GroupedSchedulers {
    pub schedulers: Vec<Box<dyn SchedulerHandle>>,
    pub actor: JoinHandle<Result<()>>,
    pub(crate) completion_drain: CompletionBoundaryDrain,
}

#[derive(Clone)]
pub(crate) struct CompletionBoundaryDrain {
    receiver: watch::Receiver<bool>,
    #[cfg(test)]
    test_gate: Arc<CompletionBoundaryTestGate>,
}

impl CompletionBoundaryDrain {
    pub(crate) async fn wait(&self) -> Result<()> {
        let mut receiver = self.receiver.clone();
        loop {
            if !*receiver.borrow_and_update() {
                return Ok(());
            }
            receiver
                .changed()
                .await
                .context("grouped completion-boundary acknowledgement lane closed")?;
        }
    }

    #[cfg(test)]
    pub(crate) fn pause_before_finish(&self) -> CompletionBoundaryTestControl {
        self.test_gate.paused.store(true, Ordering::Release);
        CompletionBoundaryTestControl {
            gate: Arc::clone(&self.test_gate),
        }
    }
}

struct CompletionBoundaryTracker {
    sender: watch::Sender<bool>,
    #[cfg(test)]
    test_gate: Arc<CompletionBoundaryTestGate>,
}

impl CompletionBoundaryTracker {
    fn new() -> (Self, CompletionBoundaryDrain) {
        let (sender, receiver) = watch::channel(false);
        #[cfg(test)]
        let test_gate = Arc::new(CompletionBoundaryTestGate::default());
        (
            Self {
                sender,
                #[cfg(test)]
                test_gate: Arc::clone(&test_gate),
            },
            CompletionBoundaryDrain {
                receiver,
                #[cfg(test)]
                test_gate,
            },
        )
    }

    fn enter(&self) -> CompletionBoundaryGuard {
        debug_assert!(!*self.sender.borrow());
        self.sender.send_replace(true);
        CompletionBoundaryGuard {
            sender: self.sender.clone(),
        }
    }

    async fn before_finish(&self) {
        #[cfg(test)]
        if self.test_gate.paused.load(Ordering::Acquire) {
            self.test_gate.reached.notify_one();
            self.test_gate.release.notified().await;
            self.test_gate.paused.store(false, Ordering::Release);
        }
    }
}

struct CompletionBoundaryGuard {
    sender: watch::Sender<bool>,
}

impl Drop for CompletionBoundaryGuard {
    fn drop(&mut self) {
        self.sender.send_replace(false);
    }
}

#[cfg(test)]
#[derive(Default)]
struct CompletionBoundaryTestGate {
    paused: AtomicBool,
    reached: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[cfg(test)]
pub(crate) struct CompletionBoundaryTestControl {
    gate: Arc<CompletionBoundaryTestGate>,
}

#[cfg(test)]
impl CompletionBoundaryTestControl {
    pub(crate) async fn wait_until_reached(&self) {
        self.gate.reached.notified().await;
    }

    pub(crate) fn release(self) {
        self.gate.release.notify_one();
    }
}

/// Construct one generalized engine and a rank-fixed compatibility handle for
/// each attention-DP rank.
pub fn create_grouped_scheduler(
    args: MockEngineArgs,
    rank_sinks: Vec<GroupedSchedulerRankSinks>,
    cancellation_token: Option<CancellationToken>,
) -> Result<GroupedSchedulers> {
    let rank_sinks = rank_sinks
        .into_iter()
        .map(|sinks| GroupedSchedulerRankEventSinks {
            event_tx: sinks.output_tx.map(SchedulerEventSender::from),
            kv_event_publishers: sinks.kv_event_publishers,
            fpm_publisher: sinks.fpm_publisher,
        })
        .collect();
    create_grouped_scheduler_with_event_senders(args, rank_sinks, cancellation_token)
}

pub(crate) struct GroupedSchedulerRankEventSinks {
    pub(crate) event_tx: Option<SchedulerEventSender>,
    pub(crate) kv_event_publishers: KvEventPublishers,
    pub(crate) fpm_publisher: FpmPublisher,
}

pub(crate) fn create_grouped_scheduler_with_event_senders(
    args: MockEngineArgs,
    rank_sinks: Vec<GroupedSchedulerRankEventSinks>,
    cancellation_token: Option<CancellationToken>,
) -> Result<GroupedSchedulers> {
    let emit_kv_events = rank_sinks
        .iter()
        .any(|sinks| !sinks.kv_event_publishers.is_empty());
    let emit_kv_token_ids = rank_sinks
        .iter()
        .any(|sinks| sinks.kv_event_publishers.raw_enabled());
    let components = engine_components(args, emit_kv_events, emit_kv_token_ids)?;
    let dp_size = NonZeroU32::new(components.args.dp_size)
        .context("grouped scheduler dp_size must be positive")?;
    ensure!(
        rank_sinks.len() == dp_size.get() as usize,
        "grouped scheduler requires one sink bundle per DP rank: expected {}, got {}",
        dp_size,
        rank_sinks.len()
    );

    let identity = EngineIdentity::new(0);
    let rank_identities = (0..dp_size.get())
        .map(|dp_rank| identity.rank(dp_rank, dp_size))
        .collect();
    create_grouped_scheduler_from_components(
        components,
        rank_sinks,
        rank_identities,
        cancellation_token,
    )
}

/// Construct the historical one-rank scheduler facade while retaining the
/// caller's externally visible DP-rank identity.
pub(crate) fn create_single_rank_scheduler_with_event_sender(
    args: MockEngineArgs,
    dp_rank: u32,
    rank_sink: GroupedSchedulerRankEventSinks,
    cancellation_token: Option<CancellationToken>,
) -> Result<GroupedSchedulers> {
    let emit_kv_events = !rank_sink.kv_event_publishers.is_empty();
    let emit_kv_token_ids = rank_sink.kv_event_publishers.raw_enabled();
    let components = engine_components(args, emit_kv_events, emit_kv_token_ids)?;
    let configured_dp_size = components.args.dp_size.max(dp_rank.saturating_add(1));
    let configured_dp_size = NonZeroU32::new(configured_dp_size)
        .context("single-rank scheduler dp_size must be positive")?;
    let rank_identity = EngineIdentity::new(0).rank(dp_rank, configured_dp_size);
    create_grouped_scheduler_from_components(
        components,
        vec![rank_sink],
        vec![rank_identity],
        cancellation_token,
    )
}

fn create_grouped_scheduler_from_components(
    components: EngineComponents,
    rank_sinks: Vec<GroupedSchedulerRankEventSinks>,
    rank_identities: Vec<RankIdentity>,
    cancellation_token: Option<CancellationToken>,
) -> Result<GroupedSchedulers> {
    ensure!(
        !rank_sinks.is_empty(),
        "grouped scheduler requires at least one rank"
    );
    ensure!(
        rank_sinks.len() == rank_identities.len(),
        "grouped scheduler rank identity count {} does not match sink count {}",
        rank_identities.len(),
        rank_sinks.len()
    );
    let rank_count =
        u32::try_from(rank_sinks.len()).context("grouped scheduler rank count must fit in u32")?;
    let dp_size =
        NonZeroU32::new(rank_count).context("grouped scheduler rank count must fit in u32")?;

    let control_capacity = handoff_channel_capacity(&components.args)
        .checked_mul(dp_size.get() as usize)
        .context("grouped scheduler control capacity overflow")?;
    let event_capacity = control_capacity.max(dp_size.get() as usize * 4).max(64);
    let factory = engine_factory(components.rank, components.timing)?;
    // Existing live schedulers seed every process-local worker from DP rank.
    // A logical worker therefore retains worker_id=0 at this compatibility
    // boundary; Replayer-owned fleets supply their own stable worker IDs.
    let engine = GeneralizedMockerEngine::new_with_rank_factory(
        EngineIdentity::new(0),
        dp_size,
        |local_identity| {
            let identity = rank_identities[local_identity.dp_rank as usize];
            factory.build_rank(identity)
        },
    )?;
    let cancel_token = cancellation_token.unwrap_or_default();
    let GroupedLiveRuntime {
        handle: grouped_handle,
        events,
        actor: engine_actor,
    } = spawn_grouped_live_engine(
        engine,
        GroupedLiveDriverConfig {
            control_capacity,
            event_capacity,
        },
        Some(cancel_token.clone()),
    )?;

    let compatibility = Arc::new(CompatibilityState::new(components.args.clone()));
    let pending = Arc::new(Mutex::new(HashMap::new()));
    let (response_monitor_tx, response_monitor_rx) = mpsc::channel(control_capacity);
    let cancel_guard = Arc::new(CompatibilityCancelGuard(cancel_token.clone()));
    let mut dispatch_ranks = Vec::with_capacity(dp_size.get() as usize);
    let mut schedulers: Vec<Box<dyn SchedulerHandle>> = Vec::with_capacity(dp_size.get() as usize);
    let mut child_tasks = Vec::new();

    for (local_dp_rank, (sinks, rank_identity)) in
        rank_sinks.into_iter().zip(rank_identities).enumerate()
    {
        let local_dp_rank = local_dp_rank as u32;
        let external_dp_rank = rank_identity.dp_rank;
        let (request_tx, request_rx) = mpsc::channel(control_capacity);
        let (command_tx, command_rx) = mpsc::channel(control_capacity);
        let (cancellation_tx, cancellation_rx) = mpsc::channel(control_capacity);
        let (lifecycle_tx, lifecycle_rx) = mpsc::channel(control_capacity);
        let (metrics_tx, metrics_rx) = watch::channel(MockerMetrics::new(
            external_dp_rank,
            0,
            components.args.num_gpu_blocks as u64,
        ));

        dispatch_ranks.push(RankDispatch {
            external_dp_rank,
            event_tx: sinks.event_tx,
            kv_event_publishers: sinks.kv_event_publishers,
            fpm_publisher: sinks.fpm_publisher,
            lifecycle_tx,
            metrics_tx,
        });
        schedulers.push(Box::new(FixedRankSchedulerHandle {
            request_tx,
            command_tx,
            cancellation_tx,
            lifecycle_rx: Some(lifecycle_rx),
            metrics_rx,
            _cancel_guard: Arc::clone(&cancel_guard),
        }));

        let bridge = RankBridgeContext {
            dp_rank: local_dp_rank,
            grouped: grouped_handle.clone(),
            compatibility: Arc::clone(&compatibility),
            pending: Arc::clone(&pending),
            response_monitor: response_monitor_tx.clone(),
            cancel: cancel_token.clone(),
        };
        child_tasks.push(tokio::spawn(run_rank_control_bridge(
            request_rx,
            command_rx,
            bridge.clone(),
        )));
        child_tasks.push(tokio::spawn(run_rank_cancellation_bridge(
            cancellation_rx,
            bridge,
        )));
    }

    let (completion_tracker, completion_drain) = CompletionBoundaryTracker::new();
    child_tasks.push(tokio::spawn(run_effect_dispatcher(
        events,
        dispatch_ranks,
        Arc::clone(&compatibility),
        Arc::clone(&pending),
        cancel_token.clone(),
        completion_tracker,
    )));
    drop(response_monitor_tx);
    child_tasks.push(tokio::spawn(run_command_response_monitor(
        response_monitor_rx,
        Arc::clone(&compatibility),
        Arc::clone(&pending),
        cancel_token.clone(),
    )));
    child_tasks.push(engine_actor);
    let actor = supervise_grouped_tasks(child_tasks, cancel_token);

    Ok(GroupedSchedulers {
        schedulers,
        actor,
        completion_drain,
    })
}

struct CompatibilityCancelGuard(CancellationToken);

impl Drop for CompatibilityCancelGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct FixedRankSchedulerHandle {
    request_tx: mpsc::Sender<DirectRequest>,
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    lifecycle_rx: Option<mpsc::Receiver<SchedulerLifecycleEvent>>,
    metrics_rx: watch::Receiver<MockerMetrics>,
    _cancel_guard: Arc<CompatibilityCancelGuard>,
}

impl SchedulerHandle for FixedRankSchedulerHandle {
    fn request_sender(&self) -> mpsc::Sender<DirectRequest> {
        self.request_tx.clone()
    }

    fn metrics_receiver(&self) -> watch::Receiver<MockerMetrics> {
        self.metrics_rx.clone()
    }

    fn command_sender(&self) -> mpsc::Sender<SchedulerCommandEnvelope> {
        self.command_tx.clone()
    }

    fn cancellation_sender(&self) -> mpsc::Sender<SchedulerCancellationEnvelope> {
        self.cancellation_tx.clone()
    }

    fn take_lifecycle_receiver(&mut self) -> Option<mpsc::Receiver<SchedulerLifecycleEvent>> {
        self.lifecycle_rx.take()
    }
}

struct PendingCommand {
    reply: Option<oneshot::Sender<Result<SchedulerCommandEffects>>>,
    on_success: Vec<Cleanup>,
    on_suppressed_output: Vec<Cleanup>,
    on_error: Vec<Cleanup>,
}

struct MonitoredCommandResponse {
    command_id: u64,
    context: &'static str,
    response: oneshot::Receiver<Result<()>>,
}

mod compatibility;
use compatibility::{Cleanup, CompatibilityState};

#[derive(Clone)]
struct RankBridgeContext {
    dp_rank: u32,
    grouped: GroupedLiveEngineHandle,
    compatibility: Arc<CompatibilityState>,
    pending: Arc<Mutex<HashMap<u64, PendingCommand>>>,
    response_monitor: mpsc::Sender<MonitoredCommandResponse>,
    cancel: CancellationToken,
}

async fn run_rank_control_bridge(
    mut request_rx: mpsc::Receiver<DirectRequest>,
    mut command_rx: mpsc::Receiver<SchedulerCommandEnvelope>,
    bridge: RankBridgeContext,
) -> Result<()> {
    let mut request_open = true;
    let mut command_open = true;
    while request_open || command_open {
        tokio::select! {
            biased;
            _ = bridge.cancel.cancelled() => return Ok(()),
            command = command_rx.recv(), if command_open => {
                match command {
                    Some(command) => {
                        forward_command(
                            command.command,
                            false,
                            Some(command.reply),
                            &bridge,
                        ).await;
                    }
                    None => command_open = false,
                }
            }
            request = request_rx.recv(), if request_open => {
                match request {
                    Some(request) => {
                        forward_command(
                            SchedulerCommand::Submit(request),
                            false,
                            None,
                            &bridge,
                        ).await;
                    }
                    None => request_open = false,
                }
            }
        }
    }
    Ok(())
}

async fn run_rank_cancellation_bridge(
    mut cancellation_rx: mpsc::Receiver<SchedulerCancellationEnvelope>,
    bridge: RankBridgeContext,
) -> Result<()> {
    loop {
        tokio::select! {
            biased;
            _ = bridge.cancel.cancelled() => return Ok(()),
            cancellation = cancellation_rx.recv() => {
                let Some(cancellation) = cancellation else {
                    return Ok(());
                };
                forward_command(
                    SchedulerCommand::CancelRequest {
                        request_id: cancellation.request_id,
                    },
                    cancellation.discard_pending_output,
                    Some(cancellation.reply),
                    &bridge,
                ).await;
            }
        }
    }
}

async fn forward_command(
    command: SchedulerCommand,
    discard_pending_output: bool,
    reply: Option<oneshot::Sender<Result<SchedulerCommandEffects>>>,
    bridge: &RankBridgeContext,
) {
    let translated = match translate_command(command, discard_pending_output, &bridge.compatibility)
    {
        Ok(command) => command,
        Err(error) => {
            if let Some(reply) = reply {
                let _ = reply.send(Err(error));
            } else {
                tracing::warn!(
                    dp_rank = bridge.dp_rank,
                    error = ?error,
                    "failed to translate grouped live request"
                );
            }
            return;
        }
    };
    let command_id = match bridge.grouped.reserve_command_id() {
        Ok(command_id) => command_id,
        Err(error) => {
            if let Some(reply) = reply {
                let _ = reply.send(Err(error));
            }
            return;
        }
    };
    bridge.pending.lock().insert(
        command_id,
        PendingCommand {
            reply,
            on_success: translated.on_success,
            on_suppressed_output: translated.on_suppressed_output,
            on_error: translated.on_error,
        },
    );
    let queued = match bridge
        .grouped
        .enqueue_reserved_command(
            command_id,
            EngineSchedulerCommand::new(bridge.dp_rank, translated.command),
        )
        .await
    {
        Ok(queued) => queued,
        Err(error) => {
            fail_pending_command(command_id, error, &bridge.compatibility, &bridge.pending);
            return;
        }
    };
    debug_assert_eq!(queued.command_id, command_id);
    // Do not serialize either scheduler input lane on the actor
    // acknowledgement. The bridge must keep draining a burst so every command
    // already queued at a pass boundary can enter the same actor snapshot.
    // One shared monitor handles exceptional response paths without spawning a
    // Tokio task per command.
    if bridge
        .response_monitor
        .send(MonitoredCommandResponse {
            command_id,
            context: "applying command",
            response: queued.response,
        })
        .await
        .is_err()
    {
        fail_pending_command(
            command_id,
            anyhow!("grouped live command response monitor is closed"),
            &bridge.compatibility,
            &bridge.pending,
        );
    }
}

async fn run_command_response_monitor(
    mut monitored_rx: mpsc::Receiver<MonitoredCommandResponse>,
    compatibility: Arc<CompatibilityState>,
    pending: Arc<Mutex<HashMap<u64, PendingCommand>>>,
    cancel: CancellationToken,
) -> Result<()> {
    let mut responses = FuturesUnordered::new();
    let mut monitor_open = true;
    loop {
        if !monitor_open && responses.is_empty() {
            return Ok(());
        }
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            monitored = monitored_rx.recv(), if monitor_open => {
                match monitored {
                    Some(monitored) => responses.push(async move {
                        let result = monitored.response.await;
                        (monitored.command_id, monitored.context, result)
                    }),
                    None => monitor_open = false,
                }
            }
            completed = responses.next(), if !responses.is_empty() => {
                let Some((command_id, context, result)) = completed else {
                    continue;
                };
                match result {
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => {
                        fail_pending_command(command_id, error, &compatibility, &pending);
                    }
                    Err(error) => {
                        fail_pending_command(
                            command_id,
                            anyhow!(
                                "grouped live engine stopped before {context} for command {command_id}: {error}"
                            ),
                            &compatibility,
                            &pending,
                        );
                    }
                }
            }
        }
    }
}

struct TranslatedCommand {
    command: Command,
    on_success: Vec<Cleanup>,
    on_suppressed_output: Vec<Cleanup>,
    on_error: Vec<Cleanup>,
}

fn translate_command(
    command: SchedulerCommand,
    discard_pending_output: bool,
    compatibility: &CompatibilityState,
) -> Result<TranslatedCommand> {
    let (command, on_success, on_suppressed_output, on_error) = match command {
        SchedulerCommand::Submit(request) => {
            let request = compatibility.native_request(request);
            let request_id = request.request_id;
            (
                Command::Submit(request),
                Vec::new(),
                Vec::new(),
                vec![Cleanup::Request(request_id)],
            )
        }
        SchedulerCommand::CancelRequest { request_id } => (
            Command::CancelRequest {
                request_id,
                discard_pending_output,
            },
            Vec::new(),
            vec![Cleanup::Request(request_id)],
            Vec::new(),
        ),
        SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request,
        } => {
            let engine_handoff = compatibility.engine_handoff(handoff_id)?;
            let request = compatibility.native_request(request);
            let request_id = request.request_id;
            compatibility.mark_source(handoff_id, request_id);
            (
                Command::SubmitHandoffPrefill {
                    handoff_id: engine_handoff,
                    request,
                },
                Vec::new(),
                Vec::new(),
                vec![
                    Cleanup::Request(request_id),
                    Cleanup::SourceHandoff(handoff_id),
                ],
            )
        }
        SchedulerCommand::ReleaseSource { handoff_id } => (
            Command::ReleaseSource {
                handoff_id: compatibility.engine_handoff(handoff_id)?,
            },
            vec![Cleanup::SourceHandoff(handoff_id)],
            Vec::new(),
            Vec::new(),
        ),
        SchedulerCommand::CancelSource { handoff_id } => (
            Command::CancelSource {
                handoff_id: compatibility.engine_handoff(handoff_id)?,
            },
            vec![Cleanup::SourceHandoff(handoff_id)],
            Vec::new(),
            Vec::new(),
        ),
        SchedulerCommand::ReserveDestination {
            handoff_id,
            request,
        } => {
            let engine_handoff = compatibility.engine_handoff(handoff_id)?;
            let request = compatibility.native_request(request);
            let request_id = request.request_id;
            compatibility.mark_destination(handoff_id, request_id);
            (
                Command::ReserveDestination {
                    handoff_id: engine_handoff,
                    request,
                },
                Vec::new(),
                Vec::new(),
                vec![
                    Cleanup::Request(request_id),
                    Cleanup::DestinationHandoff(handoff_id),
                ],
            )
        }
        SchedulerCommand::ActivateDestination { handoff_id } => (
            Command::ActivateDestination {
                handoff_id: compatibility.engine_handoff(handoff_id)?,
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ),
        SchedulerCommand::CancelDestination { handoff_id } => (
            Command::CancelDestination {
                handoff_id: compatibility.engine_handoff(handoff_id)?,
            },
            vec![Cleanup::DestinationHandoff(handoff_id)],
            Vec::new(),
            Vec::new(),
        ),
    };
    Ok(TranslatedCommand {
        command,
        on_success,
        on_suppressed_output,
        on_error,
    })
}

fn fail_pending_command(
    command_id: u64,
    error: anyhow::Error,
    compatibility: &CompatibilityState,
    pending: &Mutex<HashMap<u64, PendingCommand>>,
) {
    let pending = pending.lock().remove(&command_id);
    let Some(pending) = pending else {
        return;
    };
    for cleanup in pending.on_error {
        compatibility.apply_cleanup(cleanup);
    }
    if let Some(reply) = pending.reply {
        let _ = reply.send(Err(error));
    } else {
        tracing::warn!(command_id, error = ?error, "grouped live request failed");
    }
}

mod dispatch;
#[cfg(test)]
use dispatch::{
    DeferredCommandPublication, completion_metrics, dispatch_command_effects,
    publish_pass_router_effects,
};
use dispatch::{RankDispatch, run_effect_dispatcher};

fn supervise_grouped_tasks(
    tasks: Vec<JoinHandle<Result<()>>>,
    cancel: CancellationToken,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let mut children = tasks.into_iter().collect::<FuturesUnordered<_>>();

        let first = tokio::select! {
            biased;
            _ = cancel.cancelled() => None,
            result = children.next() => result,
        };
        cancel.cancel();

        let mut first_error = match first {
            Some(Ok(Err(error))) => Some(error),
            Some(Err(error)) => Some(anyhow!(
                "grouped scheduler supervisor task panicked: {error}"
            )),
            _ => None,
        };
        while let Some(result) = children.next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    first_error.get_or_insert(error);
                }
                Err(error) => {
                    first_error.get_or_insert_with(|| {
                        anyhow!("grouped scheduler supervisor task panicked: {error}")
                    });
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    })
}

#[cfg(test)]
mod tests;
