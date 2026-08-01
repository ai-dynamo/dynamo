// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio/wall-clock driver for one logical generalized mock engine.
//!
//! The AISimulate engine remains runtime-neutral: it eagerly computes a pass
//! and returns an absolute modeled completion time. This module owns the live
//! concerns around that contract: bounded control lanes, wall-clock sleeps,
//! mid-pass cancellation, attention-DP barrier release, and ordered effect
//! delivery. Dynamo-specific transport and metric publication stay outside the
//! driver and consume [`GroupedLiveEvent`] values.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use aisimulate_engine::generalized::{
    EngineEffects, EnginePassCompleted, EnginePassStarted, GeneralizedMockerEngine,
    SchedulerCommand,
};
use aisimulate_engine::{
    Command, CommandEffects, PassCompletionEffects, PassStartEffects, SchedulerRank,
};
use anyhow::{Context, Result, anyhow, bail};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
#[cfg(test)]
use uuid::Uuid;

#[cfg(not(test))]
use crate::common::utils::sleep_until_precise;

/// Engine type driven by the native live runtime.
pub type GroupedEngine = GeneralizedMockerEngine<SchedulerRank>;

/// Bounded-channel sizing for one grouped live engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedLiveDriverConfig {
    /// Capacity shared independently by the ordinary-command and cancellation
    /// lanes.
    pub control_capacity: usize,
    /// Capacity of the ordered neutral-effect lane.
    pub event_capacity: usize,
}

impl Default for GroupedLiveDriverConfig {
    fn default() -> Self {
        Self {
            control_capacity: 64,
            event_capacity: 64,
        }
    }
}

impl GroupedLiveDriverConfig {
    fn validate(self) -> Result<Self> {
        if self.control_capacity == 0 {
            bail!("grouped live control capacity must be positive");
        }
        if self.event_capacity == 0 {
            bail!("grouped live event capacity must be positive");
        }
        Ok(self)
    }
}

/// Ordered, runtime-neutral effects released by the grouped live driver.
///
/// A Dynamo adapter consumes this stream and owns router event publication,
/// request output delivery, lifecycle transport, FPM publication, and
/// production metric updates. Backpressure on this lane deliberately pauses
/// the engine before it exposes later effects.
#[derive(Debug)]
pub enum GroupedLiveEvent {
    /// One command was applied. This is enqueued before the command caller is
    /// acknowledged.
    CommandApplied {
        command_id: u64,
        pass_in_flight: bool,
        is_request_cancellation: bool,
        effects: EngineEffects<CommandEffects>,
    },
    /// Start-visible admissions and KV events for a grouped pass.
    PassStarted(EnginePassStarted<PassStartEffects>),
    /// Completion-visible outputs, lifecycle events, KV events, and metrics.
    ///
    /// The adapter owns this boundary until it calls
    /// [`GroupedPassBoundary::finish`]. It may synchronously apply cleanup
    /// commands first (for example when an output receiver has closed),
    /// preventing the actor from starting another pass with stale ownership.
    PassCompleted {
        completed: EnginePassCompleted<PassCompletionEffects>,
        boundary: GroupedPassBoundary,
    },
}

struct ControlEnvelope {
    command_id: u64,
    command: SchedulerCommand<Command>,
    reply: oneshot::Sender<Result<()>>,
}

enum BoundaryRequest {
    Apply {
        command: SchedulerCommand<Command>,
        reply: oneshot::Sender<Result<EngineEffects<CommandEffects>>>,
    },
    Finish,
}

/// Adapter-owned handle that keeps a completed pass at its publication
/// boundary until Dynamo has handled delivery-dependent cleanup.
#[derive(Debug)]
pub struct GroupedPassBoundary {
    request_tx: mpsc::Sender<BoundaryRequest>,
}

impl GroupedPassBoundary {
    pub(crate) async fn apply_command(
        &self,
        command: SchedulerCommand<Command>,
    ) -> Result<EngineEffects<CommandEffects>> {
        let (reply, response) = oneshot::channel();
        self.request_tx
            .send(BoundaryRequest::Apply { command, reply })
            .await
            .map_err(|_| anyhow!("grouped live pass boundary is closed"))?;
        response
            .await
            .context("grouped live engine stopped while applying a boundary command")?
    }

    pub(crate) async fn finish(self) -> Result<()> {
        self.request_tx
            .send(BoundaryRequest::Finish)
            .await
            .map_err(|_| anyhow!("grouped live pass boundary is closed"))
    }
}

struct LiveCancelGuard(CancellationToken);

impl Drop for LiveCancelGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Cloneable control handle for one logical live engine.
///
/// Ordinary commands preserve FIFO order on one bounded lane. Request
/// cancellation has a separate lane so it can suppress retained output while
/// a modeled pass is in flight even when an ordinary command is deferred.
#[derive(Clone)]
pub struct GroupedLiveEngineHandle {
    command_tx: mpsc::Sender<ControlEnvelope>,
    cancellation_tx: mpsc::Sender<ControlEnvelope>,
    #[cfg(test)]
    cancel_token: CancellationToken,
    next_command_id: Arc<AtomicU64>,
    _cancel_guard: Arc<LiveCancelGuard>,
}

impl GroupedLiveEngineHandle {
    /// Apply one rank-addressed command and wait until its effects have been
    /// placed on the ordered event lane.
    ///
    /// [`Command::CancelRequest`] is automatically routed through the
    /// dedicated cancellation lane. Other commands retain ordinary FIFO
    /// ordering.
    #[cfg(test)]
    async fn apply_command(&self, command: SchedulerCommand<Command>) -> Result<()> {
        let queued = self.enqueue_command(command).await?;
        queued
            .response
            .await
            .context("grouped live engine stopped before acknowledging a command")?
    }

    /// Enqueue a command without waiting for its modeled application.
    ///
    /// The compatibility adapter uses the returned correlation ID to delay a
    /// legacy acknowledgement until the matching [`GroupedLiveEvent`] has
    /// been published through Dynamo's existing sinks.
    #[cfg(test)]
    async fn enqueue_command(
        &self,
        command: SchedulerCommand<Command>,
    ) -> Result<QueuedGroupedLiveCommand> {
        let command_id = self.reserve_command_id()?;
        self.enqueue_reserved_command(command_id, command).await
    }

    pub(crate) fn reserve_command_id(&self) -> Result<u64> {
        self.next_command_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow!("grouped live command ID overflow"))
    }

    pub(crate) async fn enqueue_reserved_command(
        &self,
        command_id: u64,
        command: SchedulerCommand<Command>,
    ) -> Result<QueuedGroupedLiveCommand> {
        let is_cancellation = matches!(command.command, Command::CancelRequest { .. });
        let sender = if is_cancellation {
            &self.cancellation_tx
        } else {
            &self.command_tx
        };
        let (reply, response) = oneshot::channel();
        sender
            .send(ControlEnvelope {
                command_id,
                command,
                reply,
            })
            .await
            .map_err(|_| anyhow!("grouped live engine control lane is closed"))?;
        Ok(QueuedGroupedLiveCommand {
            command_id,
            response,
        })
    }

    /// Cancel one request through the mid-pass-safe cancellation lane.
    #[cfg(test)]
    async fn cancel_request(&self, dp_rank: u32, request_id: Uuid) -> Result<()> {
        self.apply_command(SchedulerCommand::new(
            dp_rank,
            Command::CancelRequest {
                request_id,
                discard_pending_output: true,
            },
        ))
        .await
    }

    /// Request an orderly actor shutdown.
    #[cfg(test)]
    fn shutdown(&self) {
        self.cancel_token.cancel();
    }
}

pub(crate) struct QueuedGroupedLiveCommand {
    pub(crate) command_id: u64,
    pub(crate) response: oneshot::Receiver<Result<()>>,
}

/// Spawned grouped live engine and its Dynamo-owned effect boundary.
pub struct GroupedLiveRuntime {
    pub handle: GroupedLiveEngineHandle,
    pub events: mpsc::Receiver<GroupedLiveEvent>,
    pub actor: JoinHandle<Result<()>>,
}

/// Drive one single-rank or attention-DP engine using Tokio's wall
/// clock.
///
/// `cancel_token` may be shared with the owning Dynamo component. Dropping the
/// last [`GroupedLiveEngineHandle`] also cancels it.
pub fn spawn_grouped_live_engine(
    engine: GroupedEngine,
    config: GroupedLiveDriverConfig,
    cancel_token: Option<CancellationToken>,
) -> Result<GroupedLiveRuntime> {
    let config = config.validate()?;
    let (command_tx, command_rx) = mpsc::channel(config.control_capacity);
    let (cancellation_tx, cancellation_rx) = mpsc::channel(config.control_capacity);
    let (event_tx, events) = mpsc::channel(config.event_capacity);
    let cancel_token = cancel_token.unwrap_or_default();
    let actor_cancel_token = cancel_token.clone();
    let cancel_guard = Arc::new(LiveCancelGuard(cancel_token.clone()));
    let next_command_id = Arc::new(AtomicU64::new(0));
    let clock_origin = Instant::now();
    let actor = tokio::spawn(async move {
        GroupedLiveActor {
            engine,
            command_rx,
            cancellation_rx,
            event_tx,
            cancel_token: actor_cancel_token,
            clock_origin,
            deferred_commands: VecDeque::new(),
        }
        .run()
        .await
    });
    Ok(GroupedLiveRuntime {
        handle: GroupedLiveEngineHandle {
            command_tx,
            cancellation_tx,
            #[cfg(test)]
            cancel_token,
            next_command_id,
            _cancel_guard: cancel_guard,
        },
        events,
        actor,
    })
}

struct GroupedLiveActor {
    engine: GroupedEngine,
    command_rx: mpsc::Receiver<ControlEnvelope>,
    cancellation_rx: mpsc::Receiver<ControlEnvelope>,
    event_tx: mpsc::Sender<GroupedLiveEvent>,
    cancel_token: CancellationToken,
    clock_origin: Instant,
    deferred_commands: VecDeque<ControlEnvelope>,
}

impl GroupedLiveActor {
    async fn run(&mut self) -> Result<()> {
        loop {
            if self.cancel_token.is_cancelled() {
                return Ok(());
            }

            self.process_due_internal_work().await?;
            if !self.engine.is_ready() {
                if !self.wait_for_idle_work().await? {
                    return Ok(());
                }
                continue;
            }

            // Bound this turn to work already queued at the readiness boundary,
            // so a continuously refilled control lane cannot starve a pass.
            self.apply_idle_control_snapshot().await?;
            if !self.engine.is_ready() {
                continue;
            }

            let started_at_ms = self.elapsed_ms();
            let Some(started) = self.engine.execute_pass(started_at_ms)? else {
                continue;
            };
            let pass_id = started.pass_id;
            let end_ms = started.end_ms;
            let zero_duration = end_ms <= started_at_ms;
            self.publish(GroupedLiveEvent::PassStarted(started)).await?;

            if !self.wait_for_pass_boundary(end_ms).await? {
                return Ok(());
            }
            let completed_at_ms = self.elapsed_ms().max(end_ms);
            let completed = self.engine.complete_pass(pass_id, completed_at_ms)?;
            let (boundary_tx, boundary_rx) = mpsc::channel(1);
            self.publish(GroupedLiveEvent::PassCompleted {
                completed,
                boundary: GroupedPassBoundary {
                    request_tx: boundary_tx,
                },
            })
            .await?;
            if !self.serve_pass_boundary(boundary_rx).await? {
                return Ok(());
            }
            self.apply_post_pass_controls().await?;

            // A zero-duration engine can remain continuously ready. Yielding
            // keeps cancellation and sibling tasks responsive without adding
            // modeled latency.
            if zero_duration {
                tokio::task::yield_now().await;
            }
        }
    }

    fn elapsed_ms(&self) -> f64 {
        self.clock_origin.elapsed().as_secs_f64() * 1_000.0
    }

    async fn publish(&self, event: GroupedLiveEvent) -> Result<()> {
        tokio::select! {
            biased;
            result = self.event_tx.send(event) => {
                result.map_err(|_| anyhow!("grouped live engine event lane is closed"))
            },
            _ = self.cancel_token.cancelled() => {
                bail!("grouped live engine stopped while publishing effects")
            },
        }
    }

    async fn serve_pass_boundary(
        &mut self,
        mut requests: mpsc::Receiver<BoundaryRequest>,
    ) -> Result<bool> {
        loop {
            let request = tokio::select! {
                biased;
                _ = self.cancel_token.cancelled() => return Ok(false),
                request = requests.recv() => request,
            };
            let Some(request) = request else {
                bail!("grouped live pass boundary adapter stopped without finishing");
            };
            match request {
                BoundaryRequest::Apply { command, reply } => {
                    let result = self
                        .engine
                        .apply_command_effects(command, self.elapsed_ms());
                    let _ = reply.send(result);
                }
                BoundaryRequest::Finish => return Ok(true),
            }
        }
    }

    async fn apply_control(&mut self, envelope: ControlEnvelope) -> Result<()> {
        let command_id = envelope.command_id;
        let is_request_cancellation =
            matches!(&envelope.command.command, Command::CancelRequest { .. });
        let now_ms = self.elapsed_ms();
        let result = self.engine.apply_command_effects(envelope.command, now_ms);
        match result {
            Ok(effects) => {
                let event = GroupedLiveEvent::CommandApplied {
                    command_id,
                    pass_in_flight: false,
                    is_request_cancellation,
                    effects,
                };
                if let Err(error) = self.publish(event).await {
                    let _ = envelope.reply.send(Err(anyhow!(error.to_string())));
                    return Err(error);
                }
                let _ = envelope.reply.send(Ok(()));
                Ok(())
            }
            Err(error) => {
                let _ = envelope.reply.send(Err(error));
                Ok(())
            }
        }
    }

    async fn wait_for_idle_work(&mut self) -> Result<bool> {
        let deadline_ms = self.engine.next_internal_deadline_ms();
        let deadline = sleep_until_ms(self.clock_origin, deadline_ms);
        tokio::pin!(deadline);
        tokio::select! {
            biased;
            _ = self.cancel_token.cancelled() => Ok(false),
            cancellation = self.cancellation_rx.recv() => {
                let Some(cancellation) = cancellation else {
                    return Ok(false);
                };
                self.apply_control(cancellation).await?;
                Ok(true)
            }
            command = self.command_rx.recv() => {
                let Some(command) = command else {
                    return Ok(false);
                };
                self.apply_control(command).await?;
                Ok(true)
            }
            _ = &mut deadline, if deadline_ms.is_some() => {
                self.process_due_internal_work().await?;
                Ok(true)
            }
        }
    }

    async fn apply_idle_control_snapshot(&mut self) -> Result<()> {
        let cancellation_count = self.cancellation_rx.len();
        let command_count = self.command_rx.len();
        for _ in 0..cancellation_count {
            let Ok(cancellation) = self.cancellation_rx.try_recv() else {
                break;
            };
            self.apply_control(cancellation).await?;
        }
        for _ in 0..command_count {
            let Ok(command) = self.command_rx.try_recv() else {
                break;
            };
            self.apply_control(command).await?;
        }
        Ok(())
    }

    async fn wait_for_pass_boundary(&mut self, end_ms: f64) -> Result<bool> {
        let pass_deadline = sleep_until_ms(self.clock_origin, Some(end_ms));
        tokio::pin!(pass_deadline);
        let mut accept_commands = true;
        loop {
            let internal_deadline_ms = self.engine.next_internal_deadline_ms();
            let internal_deadline = sleep_until_ms(self.clock_origin, internal_deadline_ms);
            tokio::pin!(internal_deadline);
            tokio::select! {
                biased;
                _ = self.cancel_token.cancelled() => return Ok(false),
                cancellation = self.cancellation_rx.recv() => {
                    let Some(cancellation) = cancellation else {
                        return Ok(false);
                    };
                    self.apply_control_during_pass(cancellation).await?;
                }
                _ = &mut pass_deadline => return Ok(true),
                _ = &mut internal_deadline, if internal_deadline_ms.is_some() => {
                    self.process_due_internal_work().await?;
                }
                command = self.command_rx.recv(), if accept_commands => {
                    let Some(command) = command else {
                        return Ok(false);
                    };
                    if command_can_apply_during_pass(&command.command.command) {
                        self.apply_control_during_pass(command).await?;
                    } else {
                        self.deferred_commands.push_back(command);
                        accept_commands = false;
                    }
                }
            }
        }
    }

    async fn apply_control_during_pass(&mut self, envelope: ControlEnvelope) -> Result<()> {
        let command_id = envelope.command_id;
        let is_request_cancellation =
            matches!(&envelope.command.command, Command::CancelRequest { .. });
        let now_ms = self.elapsed_ms();
        let result = self.engine.apply_command_effects(envelope.command, now_ms);
        match result {
            Ok(effects) => {
                let event = GroupedLiveEvent::CommandApplied {
                    command_id,
                    pass_in_flight: true,
                    is_request_cancellation,
                    effects,
                };
                if let Err(error) = self.publish(event).await {
                    let _ = envelope.reply.send(Err(anyhow!(error.to_string())));
                    return Err(error);
                }
                let _ = envelope.reply.send(Ok(()));
            }
            Err(error) => {
                let _ = envelope.reply.send(Err(error));
            }
        }
        Ok(())
    }

    async fn apply_post_pass_controls(&mut self) -> Result<()> {
        let cancellation_count = self.cancellation_rx.len();
        let command_count = self.command_rx.len();
        for _ in 0..cancellation_count {
            let Ok(cancellation) = self.cancellation_rx.try_recv() else {
                break;
            };
            self.apply_control(cancellation).await?;
        }
        while let Some(command) = self.deferred_commands.pop_front() {
            self.apply_control(command).await?;
        }
        for _ in 0..command_count {
            let Ok(command) = self.command_rx.try_recv() else {
                break;
            };
            self.apply_control(command).await?;
        }
        Ok(())
    }

    async fn process_due_internal_work(&mut self) -> Result<()> {
        let now_ms = self.elapsed_ms();
        if !self
            .engine
            .next_internal_deadline_ms()
            .is_some_and(|deadline| deadline <= now_ms)
        {
            return Ok(());
        }
        self.engine.process_internal_work(now_ms)?;
        Ok(())
    }
}

fn command_can_apply_during_pass(command: &Command) -> bool {
    matches!(
        command,
        Command::SubmitHandoffPrefill { .. } | Command::ReserveDestination { .. }
    )
}

async fn sleep_until_ms(origin: Instant, deadline_ms: Option<f64>) {
    let Some(deadline_ms) = deadline_ms else {
        std::future::pending::<()>().await;
        return;
    };
    let deadline = origin + Duration::from_secs_f64(deadline_ms.max(0.0) / 1_000.0);
    #[cfg(test)]
    tokio::time::sleep_until(deadline).await;
    #[cfg(not(test))]
    sleep_until_precise(deadline.into_std()).await;
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use aisimulate_engine::generalized::EngineIdentity;
    use aisimulate_engine::{
        CommandResult, EngineConfig, EngineFactory, HandoffId, Request, TimingModelConfig,
        WorkerType,
    };

    use super::*;

    fn request(id: u128, prompt_len: usize, output_len: usize) -> Request {
        Request {
            request_id: Uuid::from_u128(id),
            tokens: (0..prompt_len as u32).collect(),
            max_output_tokens: output_len,
            output_token_ids: Some((0..output_len as u32).map(|token| token + 10_000).collect()),
        }
    }

    fn runtime(dp_size: u32, pass_ms: f64) -> GroupedLiveRuntime {
        runtime_with_config(
            dp_size,
            EngineConfig {
                num_gpu_blocks: 128,
                block_size: 4,
                max_num_seqs: 8,
                max_num_batched_tokens: 256,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: pass_ms,
                    decode_ms: 0.0,
                },
                ..EngineConfig::default()
            },
        )
    }

    fn runtime_with_config(dp_size: u32, config: EngineConfig) -> GroupedLiveRuntime {
        let engine = EngineFactory::new(config)
            .unwrap()
            .build(EngineIdentity::new(7), NonZeroU32::new(dp_size).unwrap())
            .unwrap();
        spawn_grouped_live_engine(engine, GroupedLiveDriverConfig::default(), None).unwrap()
    }

    async fn next_event(events: &mut mpsc::Receiver<GroupedLiveEvent>) -> GroupedLiveEvent {
        events.recv().await.expect("live actor must remain active")
    }

    #[tokio::test(start_paused = true)]
    async fn attention_dp_releases_completion_only_at_the_shared_boundary() {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime(2, 100.0);
        let rank0 =
            handle.apply_command(SchedulerCommand::new(0, Command::Submit(request(1, 4, 1))));
        let rank1 =
            handle.apply_command(SchedulerCommand::new(1, Command::Submit(request(2, 8, 1))));
        let (rank0, rank1) = tokio::join!(rank0, rank1);
        rank0.unwrap();
        rank1.unwrap();

        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        let GroupedLiveEvent::PassStarted(started) = next_event(&mut events).await else {
            panic!("expected grouped pass start");
        };
        assert_eq!(started.participating_ranks.get(), 2);
        assert_eq!(started.by_rank.len(), 2);

        tokio::time::advance(Duration::from_millis(99)).await;
        tokio::task::yield_now().await;
        assert!(events.try_recv().is_err());
        tokio::time::advance(Duration::from_millis(1)).await;
        let GroupedLiveEvent::PassCompleted {
            completed,
            boundary,
        } = next_event(&mut events).await
        else {
            panic!("expected grouped pass completion");
        };
        assert_eq!(completed.effects.by_rank.len(), 2);
        boundary.finish().await.unwrap();

        handle.shutdown();
        actor.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_suppresses_retained_output_during_a_grouped_pass() {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime(1, 100.0);
        let request_id = Uuid::from_u128(11);
        handle
            .apply_command(SchedulerCommand::new(0, Command::Submit(request(11, 4, 1))))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::PassStarted(_)
        ));

        handle.cancel_request(0, request_id).await.unwrap();
        let GroupedLiveEvent::CommandApplied {
            pass_in_flight,
            effects,
            ..
        } = next_event(&mut events).await
        else {
            panic!("expected cancellation effects");
        };
        assert!(pass_in_flight);
        assert!(effects.by_rank[0].effects.suppressed_pending_output);

        tokio::time::advance(Duration::from_millis(100)).await;
        let GroupedLiveEvent::PassCompleted {
            completed,
            boundary,
        } = next_event(&mut events).await
        else {
            panic!("expected grouped pass completion");
        };
        assert!(completed.effects.by_rank[0].effects.outputs.is_empty());
        boundary.finish().await.unwrap();

        handle.shutdown();
        actor.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn queued_cancellation_preempts_an_overdue_pass_boundary() {
        let mut engine = EngineFactory::new(EngineConfig {
            num_gpu_blocks: 128,
            block_size: 4,
            max_num_seqs: 8,
            max_num_batched_tokens: 256,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 100.0,
                decode_ms: 0.0,
            },
            ..EngineConfig::default()
        })
        .unwrap()
        .build(EngineIdentity::new(7), NonZeroU32::new(1).unwrap())
        .unwrap();
        let request_id = Uuid::from_u128(12);
        engine
            .apply_command_effects(
                SchedulerCommand::new(0, Command::Submit(request(12, 4, 2))),
                0.0,
            )
            .unwrap();
        let started = engine.execute_pass(0.0).unwrap().unwrap();

        let (_command_tx, command_rx) = mpsc::channel(1);
        let (cancellation_tx, cancellation_rx) = mpsc::channel(1);
        let (event_tx, mut events) = mpsc::channel(1);
        let (reply, mut response) = oneshot::channel();
        cancellation_tx
            .send(ControlEnvelope {
                command_id: 99,
                command: SchedulerCommand::new(
                    0,
                    Command::CancelRequest {
                        request_id,
                        discard_pending_output: true,
                    },
                ),
                reply,
            })
            .await
            .unwrap();

        // Both the pass timer and cancellation receive are ready on the first
        // poll. The dedicated cancellation lane must win so scheduler/KV
        // cleanup and retained-output suppression happen before completion.
        let mut actor = GroupedLiveActor {
            engine,
            command_rx,
            cancellation_rx,
            event_tx,
            cancel_token: CancellationToken::new(),
            clock_origin: Instant::now() - Duration::from_millis(200),
            deferred_commands: VecDeque::new(),
        };
        assert!(actor.wait_for_pass_boundary(started.end_ms).await.unwrap());
        response
            .try_recv()
            .expect("queued cancellation must be applied before the overdue boundary")
            .unwrap();

        let GroupedLiveEvent::CommandApplied {
            command_id,
            pass_in_flight,
            is_request_cancellation,
            effects,
        } = events
            .try_recv()
            .expect("cancellation effects must precede pass completion")
        else {
            panic!("expected cancellation effects");
        };
        assert_eq!(command_id, 99);
        assert!(pass_in_flight);
        assert!(is_request_cancellation);
        assert!(effects.by_rank[0].effects.suppressed_pending_output);
        assert_eq!(
            effects.by_rank[0].effects.retired_requests,
            vec![request_id]
        );

        let completed = actor
            .engine
            .complete_pass(started.pass_id, actor.elapsed_ms().max(started.end_ms))
            .unwrap();
        assert!(completed.effects.by_rank[0].effects.outputs.is_empty());
    }

    async fn noop_cancellation_outcome(discard_pending_output: bool) -> (bool, usize) {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime(1, 100.0);
        let request_id = Uuid::from_u128(21);
        handle
            .apply_command(SchedulerCommand::new(0, Command::Submit(request(21, 4, 1))))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::PassStarted(_)
        ));

        handle
            .apply_command(SchedulerCommand::new(
                0,
                Command::CancelRequest {
                    request_id,
                    discard_pending_output,
                },
            ))
            .await
            .unwrap();
        let GroupedLiveEvent::CommandApplied { effects, .. } = next_event(&mut events).await else {
            panic!("expected cancellation effects");
        };
        assert_eq!(effects.by_rank[0].effects.result, CommandResult::Noop);
        let suppressed = effects.by_rank[0].effects.suppressed_pending_output;

        tokio::time::advance(Duration::from_millis(100)).await;
        let GroupedLiveEvent::PassCompleted {
            completed,
            boundary,
        } = next_event(&mut events).await
        else {
            panic!("expected grouped pass completion");
        };
        let outputs = completed.effects.by_rank[0].effects.outputs.len();
        boundary.finish().await.unwrap();
        handle.shutdown();
        actor.await.unwrap().unwrap();
        (suppressed, outputs)
    }

    #[tokio::test(start_paused = true)]
    async fn noop_cancellation_without_discard_preserves_pending_output() {
        let (suppressed, outputs) = noop_cancellation_outcome(false).await;
        assert!(!suppressed);
        assert_eq!(outputs, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn explicit_discard_suppresses_pending_output_after_noop_cancellation() {
        let (suppressed, outputs) = noop_cancellation_outcome(true).await;
        assert!(suppressed);
        assert_eq!(outputs, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn held_source_waits_for_release_without_empty_passes() {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime_with_config(
            1,
            EngineConfig {
                worker_type: WorkerType::Prefill,
                num_gpu_blocks: 128,
                block_size: 4,
                max_num_seqs: 8,
                max_num_batched_tokens: 256,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 10.0,
                    decode_ms: 0.0,
                },
                ..EngineConfig::default()
            },
        );
        let handoff_id = HandoffId::from(Uuid::from_u128(51));
        handle
            .apply_command(SchedulerCommand::new(
                0,
                Command::SubmitHandoffPrefill {
                    handoff_id,
                    request: request(52, 4, 1),
                },
            ))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::PassStarted(_)
        ));
        tokio::time::advance(Duration::from_millis(10)).await;
        let GroupedLiveEvent::PassCompleted {
            completed,
            boundary,
        } = next_event(&mut events).await
        else {
            panic!("expected source-hold pass completion");
        };
        assert!(matches!(
            completed.effects.by_rank[0].effects.lifecycle_events.as_slice(),
            [aisimulate_engine::LifecycleEvent::SourceHeld { handoff_id: observed, .. }]
                if *observed == handoff_id
        ));
        boundary.finish().await.unwrap();

        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            events.try_recv().is_err(),
            "a held source must not generate effect-free passes"
        );

        handle
            .apply_command(SchedulerCommand::new(
                0,
                Command::ReleaseSource { handoff_id },
            ))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        handle.shutdown();
        actor.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn reserved_destination_waits_for_activation_without_empty_passes() {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime_with_config(
            1,
            EngineConfig {
                worker_type: WorkerType::Decode,
                num_gpu_blocks: 128,
                block_size: 4,
                max_num_seqs: 8,
                max_num_batched_tokens: 256,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 10.0,
                    decode_ms: 0.0,
                },
                ..EngineConfig::default()
            },
        );
        let handoff_id = HandoffId::from(Uuid::from_u128(61));
        handle
            .apply_command(SchedulerCommand::new(
                0,
                Command::ReserveDestination {
                    handoff_id,
                    request: request(62, 4, 1),
                },
            ))
            .await
            .unwrap();
        let GroupedLiveEvent::CommandApplied { effects, .. } = next_event(&mut events).await else {
            panic!("expected destination reservation effects");
        };
        assert!(matches!(
            effects.by_rank[0].effects.result,
            CommandResult::DestinationAccepted { .. }
        ));

        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            events.try_recv().is_err(),
            "a reserved destination must not generate effect-free passes"
        );

        handle
            .apply_command(SchedulerCommand::new(
                0,
                Command::CancelDestination { handoff_id },
            ))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        handle.shutdown();
        actor.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn midpass_idle_sibling_completes_with_group_metrics() {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime_with_config(
            2,
            EngineConfig {
                worker_type: WorkerType::Decode,
                num_gpu_blocks: 128,
                block_size: 4,
                max_num_seqs: 8,
                max_num_batched_tokens: 256,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 100.0,
                    decode_ms: 0.0,
                },
                ..EngineConfig::default()
            },
        );
        handle
            .apply_command(SchedulerCommand::new(0, Command::Submit(request(71, 4, 1))))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        let GroupedLiveEvent::PassStarted(started) = next_event(&mut events).await else {
            panic!("expected grouped pass start");
        };
        let group_duration_ms = started.end_ms - started.started_at_ms;

        let handoff_id = HandoffId::from(Uuid::from_u128(72));
        handle
            .apply_command(SchedulerCommand::new(
                1,
                Command::ReserveDestination {
                    handoff_id,
                    request: request(73, 4, 1),
                },
            ))
            .await
            .unwrap();
        let GroupedLiveEvent::CommandApplied {
            pass_in_flight,
            effects,
            ..
        } = next_event(&mut events).await
        else {
            panic!("expected idle-sibling command effects");
        };
        assert!(pass_in_flight);
        assert_eq!(effects.by_rank[0].dp_rank, 1);

        tokio::time::advance(Duration::from_millis(100)).await;
        let GroupedLiveEvent::PassCompleted {
            completed,
            boundary,
        } = next_event(&mut events).await
        else {
            panic!("expected grouped pass completion");
        };
        let idle = completed
            .effects
            .by_rank
            .iter()
            .find(|rank| rank.dp_rank == 1)
            .expect("idle sibling must cross the shared completion boundary");
        assert_eq!(
            idle.effects.forward_pass_metrics.duration_ms,
            group_duration_ms
        );
        boundary.finish().await.unwrap();

        handle
            .apply_command(SchedulerCommand::new(
                1,
                Command::CancelDestination { handoff_id },
            ))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        handle.shutdown();
        actor.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_does_not_end_a_pass_with_unrelated_pending_output() {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime(1, 100.0);
        let cancelled =
            handle.apply_command(SchedulerCommand::new(0, Command::Submit(request(31, 4, 2))));
        let unrelated =
            handle.apply_command(SchedulerCommand::new(0, Command::Submit(request(32, 4, 1))));
        let (cancelled, unrelated) = tokio::join!(cancelled, unrelated);
        cancelled.unwrap();
        unrelated.unwrap();
        for _ in 0..2 {
            assert!(matches!(
                next_event(&mut events).await,
                GroupedLiveEvent::CommandApplied { .. }
            ));
        }
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::PassStarted(_)
        ));

        handle
            .apply_command(SchedulerCommand::new(
                0,
                Command::CancelRequest {
                    request_id: Uuid::from_u128(31),
                    discard_pending_output: true,
                },
            ))
            .await
            .unwrap();
        let GroupedLiveEvent::CommandApplied { effects, .. } = next_event(&mut events).await else {
            panic!("expected cancellation effects");
        };
        assert_eq!(effects.by_rank[0].effects.result, CommandResult::Applied);
        assert!(effects.by_rank[0].effects.suppressed_pending_output);
        assert_eq!(effects.by_rank[0].effects.metrics.running_requests, 0);
        assert_eq!(effects.by_rank[0].effects.metrics.waiting_requests, 0);

        tokio::time::advance(Duration::from_millis(99)).await;
        tokio::task::yield_now().await;
        assert!(
            events.try_recv().is_err(),
            "empty occupancy must not release unrelated completion effects early"
        );
        tokio::time::advance(Duration::from_millis(1)).await;
        let GroupedLiveEvent::PassCompleted {
            completed,
            boundary,
        } = next_event(&mut events).await
        else {
            panic!("expected grouped pass completion");
        };
        assert_eq!(completed.effects.by_rank[0].effects.outputs.len(), 1);
        assert_eq!(
            completed.effects.by_rank[0].effects.outputs[0].request_id,
            Uuid::from_u128(32)
        );
        boundary.finish().await.unwrap();
        handle.shutdown();
        actor.await.unwrap().unwrap();
    }

    // Regression: a productive zero-duration engine must yield so another
    // current-thread task can trigger external shutdown.
    #[tokio::test(flavor = "current_thread")]
    async fn external_shutdown_stops_a_nonempty_zero_duration_progress_loop() {
        let GroupedLiveRuntime {
            handle,
            mut events,
            actor,
        } = runtime(1, 0.0);
        handle
            .apply_command(SchedulerCommand::new(
                0,
                Command::Submit(request(41, 4, 32)),
            ))
            .await
            .unwrap();
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::CommandApplied { .. }
        ));
        assert!(matches!(
            next_event(&mut events).await,
            GroupedLiveEvent::PassStarted(_)
        ));
        let GroupedLiveEvent::PassCompleted { boundary, .. } = next_event(&mut events).await else {
            panic!("expected zero-duration pass completion");
        };

        let external = handle.clone();
        let shutdown = tokio::spawn(async move {
            tokio::task::yield_now().await;
            external.shutdown();
        });
        boundary.finish().await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), actor)
            .await
            .expect("zero-duration engine monopolized the current-thread runtime")
            .unwrap()
            .unwrap();
        shutdown.await.unwrap();
    }
}
