// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publication of neutral generalized-engine effects through Dynamo sinks.

use super::*;

#[derive(Clone)]
pub(super) struct RankDispatch {
    pub(super) external_dp_rank: u32,
    pub(super) event_tx: Option<SchedulerEventSender>,
    pub(super) kv_event_publishers: KvEventPublishers,
    pub(super) fpm_publisher: FpmPublisher,
    pub(super) lifecycle_tx: mpsc::Sender<SchedulerLifecycleEvent>,
    pub(super) metrics_tx: watch::Sender<MockerMetrics>,
}

struct PendingRankPublication {
    dp_rank: u32,
    lifecycle: Vec<SchedulerLifecycleEvent>,
    metrics: Metrics,
}

enum OutputPublication {
    Delivered(Vec<Uuid>),
    Cancelled,
}

enum CompletionDispatch {
    Completed,
    Cancelled,
}

#[derive(Default)]
pub(super) struct DeferredCommandPublication {
    pub(super) kv: Vec<KvEvent>,
    pub(super) metrics: Option<Metrics>,
}

pub(super) async fn run_effect_dispatcher(
    mut events: mpsc::Receiver<GroupedLiveEvent>,
    ranks: Vec<RankDispatch>,
    compatibility: Arc<CompatibilityState>,
    pending: Arc<Mutex<HashMap<u64, PendingCommand>>>,
    cancel: CancellationToken,
    completion_tracker: CompletionBoundaryTracker,
) -> Result<()> {
    let mut deferred_commands = (0..ranks.len())
        .map(|_| DeferredCommandPublication::default())
        .collect::<Vec<_>>();
    loop {
        let event = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            event = events.recv() => event,
        };
        let Some(event) = event else {
            return Ok(());
        };
        match event {
            GroupedLiveEvent::CommandApplied {
                command_id,
                pass_in_flight,
                is_request_cancellation,
                effects,
                ..
            } => {
                dispatch_command_effects(
                    command_id,
                    effects,
                    pass_in_flight,
                    is_request_cancellation,
                    &ranks,
                    &compatibility,
                    &pending,
                    &mut deferred_commands,
                )
                .await?;
            }
            GroupedLiveEvent::PassStarted(started) => {
                for rank in started.by_rank {
                    let dispatch = rank_dispatch(&ranks, rank.dp_rank)?;
                    dispatch.publish_admissions(rank.effects.admissions).await?;
                    dispatch.publish_kv(rank.effects.kv_events);
                }
            }
            GroupedLiveEvent::PassCompleted {
                completed,
                boundary,
            } => {
                let _completion_guard = completion_tracker.enter();
                dispatch_pass_completion(
                    completed,
                    boundary,
                    &ranks,
                    &compatibility,
                    &mut deferred_commands,
                    &cancel,
                    &completion_tracker,
                )
                .await?;
                ensure!(
                    deferred_commands
                        .iter()
                        .all(|deferred| deferred.kv.is_empty() && deferred.metrics.is_none()),
                    "grouped pass completion omitted deferred command effects for a rank"
                );
            }
        }
    }
}

async fn dispatch_pass_completion(
    completed: EnginePassCompleted<PassCompletionEffects>,
    boundary: GroupedPassBoundary,
    ranks: &[RankDispatch],
    compatibility: &CompatibilityState,
    deferred_commands: &mut [DeferredCommandPublication],
    cancel: &CancellationToken,
    completion_tracker: &CompletionBoundaryTracker,
) -> Result<()> {
    let dispatch_result = async {
        let mut publications = Vec::with_capacity(completed.effects.by_rank.len());
        let mut delivery_failures = Vec::new();
        for rank in completed.effects.by_rank {
            let dispatch = rank_dispatch(ranks, rank.dp_rank)?;
            let effects = rank.effects;
            publish_pass_router_effects(
                dispatch,
                effects.kv_events,
                &mut deferred_commands
                    .get_mut(rank.dp_rank as usize)
                    .context("deferred command effect rank is out of range")?
                    .kv,
                effects.forward_pass_metrics,
            );
            let outputs = effects
                .outputs
                .into_iter()
                .map(|output| compatibility.output_signal(output))
                .collect();
            match dispatch.publish_outputs(outputs).await? {
                OutputPublication::Delivered(failed_requests) => {
                    delivery_failures.extend(
                        failed_requests
                            .into_iter()
                            .map(|request_id| (rank.dp_rank, request_id)),
                    );
                }
                OutputPublication::Cancelled => return Ok(CompletionDispatch::Cancelled),
            }
            let lifecycle = effects
                .lifecycle_events
                .into_iter()
                .map(|event| compatibility.lifecycle_event(event))
                .collect::<Result<Vec<_>>>()?;
            publications.push(PendingRankPublication {
                dp_rank: rank.dp_rank,
                lifecycle,
                // A command applied mid-pass snapshots state before the
                // grouped boundary has completed. Consume that deferred
                // snapshot, but always publish the authoritative metrics
                // refreshed by `complete_pass`/`complete_idle_group_pass`.
                metrics: completion_metrics(
                    &mut deferred_commands[rank.dp_rank as usize].metrics,
                    effects.metrics,
                ),
            });
        }

        for (dp_rank, request_id) in delivery_failures {
            let command_result = boundary
                .apply_command(EngineSchedulerCommand::new(
                    dp_rank,
                    Command::CancelRequest {
                        request_id,
                        discard_pending_output: true,
                    },
                ))
                .await;
            // The output transport no longer owns this request regardless of
            // whether the engine had already retired it.
            compatibility.apply_cleanup(Cleanup::Request(request_id));
            let effects = command_result?;
            merge_boundary_command_effects(effects, ranks, compatibility, &mut publications)?;
        }

        for publication in publications {
            let dispatch = rank_dispatch(ranks, publication.dp_rank)?;
            dispatch.publish_lifecycle(publication.lifecycle).await;
            dispatch.publish_metrics(publication.metrics);
        }
        Ok(CompletionDispatch::Completed)
    }
    .await;

    // Always release the actor, including sink/conversion error paths. The
    // primary publication error remains the one returned to the supervisor.
    // A cancellation observed by the ordered output lane means the actor is
    // already shutting down, so there is no boundary left to release.
    let finish_result = if matches!(&dispatch_result, Ok(CompletionDispatch::Cancelled)) {
        Ok(())
    } else {
        completion_tracker.before_finish().await;
        finish_boundary_or_cancel(boundary.finish(), cancel).await
    };
    match dispatch_result {
        Err(error) => Err(error),
        Ok(CompletionDispatch::Cancelled) => Ok(()),
        Ok(CompletionDispatch::Completed) => finish_result,
    }
}

async fn finish_boundary_or_cancel<F>(finish: F, cancel: &CancellationToken) -> Result<()>
where
    F: std::future::Future<Output = Result<()>>,
{
    tokio::pin!(finish);
    tokio::select! {
        biased;
        result = &mut finish => result,
        _ = cancel.cancelled() => Ok(()),
    }
}

pub(super) fn completion_metrics(deferred: &mut Option<Metrics>, completed: Metrics) -> Metrics {
    deferred.take();
    completed
}

fn merge_boundary_command_effects(
    effects: EngineEffects<CommandEffects>,
    ranks: &[RankDispatch],
    compatibility: &CompatibilityState,
    publications: &mut [PendingRankPublication],
) -> Result<()> {
    ensure!(
        effects.by_rank.len() == 1,
        "output-delivery cleanup returned {} rank effect batches",
        effects.by_rank.len()
    );
    let rank = effects
        .by_rank
        .into_iter()
        .next()
        .expect("one rank effect was validated");
    let dispatch = rank_dispatch(ranks, rank.dp_rank)?;
    let effects = rank.effects;
    dispatch.publish_kv(effects.kv_events);
    let lifecycle = effects
        .lifecycle_events
        .into_iter()
        .map(|event| compatibility.lifecycle_event(event))
        .collect::<Result<Vec<_>>>()?;
    let publication = publications
        .iter_mut()
        .find(|publication| publication.dp_rank == rank.dp_rank)
        .context("output-delivery cleanup referenced a rank absent from pass completion")?;
    publication.lifecycle.extend(lifecycle);
    publication.metrics = effects.metrics;
    Ok(())
}

pub(super) fn publish_pass_router_effects(
    dispatch: &RankDispatch,
    completion_kv: Vec<KvEvent>,
    deferred_command_kv: &mut Vec<KvEvent>,
    fpm: ForwardPassMetrics,
) {
    dispatch.publish_kv(completion_kv);
    dispatch.publish_kv(std::mem::take(deferred_command_kv));
    dispatch.publish_fpm(fpm);
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_command_effects(
    command_id: u64,
    effects: EngineEffects<CommandEffects>,
    pass_in_flight: bool,
    is_request_cancellation: bool,
    ranks: &[RankDispatch],
    compatibility: &CompatibilityState,
    pending: &Mutex<HashMap<u64, PendingCommand>>,
    deferred_commands: &mut [DeferredCommandPublication],
) -> Result<()> {
    ensure!(
        effects.by_rank.len() == 1,
        "native command {command_id} returned {} rank effect batches",
        effects.by_rank.len()
    );
    let rank = effects
        .by_rank
        .into_iter()
        .next()
        .expect("one rank effect was validated");
    let dispatch = rank_dispatch(ranks, rank.dp_rank)?;
    let mut effects = rank.effects;
    let immediate_empty_metrics = (pass_in_flight
        && is_request_cancellation
        && effects.result == CommandResult::Applied
        && effects.metrics.running_requests == 0
        && effects.metrics.waiting_requests == 0)
        .then(|| effects.metrics.clone());
    if pass_in_flight {
        ensure!(
            effects.lifecycle_events.is_empty(),
            "mid-pass native command {command_id} produced lifecycle effects"
        );
        let deferred = deferred_commands
            .get_mut(rank.dp_rank as usize)
            .context("deferred command effect rank is out of range")?;
        deferred.kv.append(&mut effects.kv_events);
        deferred.metrics = Some(effects.metrics.clone());
    } else {
        dispatch.publish_kv(std::mem::take(&mut effects.kv_events));
    }
    for request_id in effects.retired_requests.drain(..) {
        compatibility.apply_cleanup(Cleanup::Request(request_id));
    }
    let lifecycle = effects
        .lifecycle_events
        .into_iter()
        .map(|event| compatibility.lifecycle_event(event))
        .collect::<Result<Vec<_>>>()?;
    let result = scheduler_command_result(effects.result, effects.suppressed_pending_output);
    let pending = pending.lock().remove(&command_id);
    if let Some(pending) = pending {
        // Idle command KV effects are visible before acknowledgement. Mid-pass
        // effects are acknowledged immediately but stay hidden until the
        // current grouped pass reaches its completion boundary.
        if let Some(reply) = pending.reply {
            let _ = reply.send(Ok(SchedulerCommandEffects {
                result,
                lifecycle_events: Vec::new(),
                kv_events: Vec::new(),
            }));
        }
        if !pass_in_flight {
            dispatch.publish_lifecycle(lifecycle).await;
            dispatch.publish_metrics(effects.metrics);
        }
        if effects.suppressed_pending_output {
            for cleanup in pending.on_suppressed_output {
                compatibility.apply_cleanup(cleanup);
            }
        }
        for cleanup in pending.on_success {
            compatibility.apply_cleanup(cleanup);
        }
    } else if !pass_in_flight {
        dispatch.publish_lifecycle(lifecycle).await;
        dispatch.publish_metrics(effects.metrics);
    }
    if let Some(metrics) = immediate_empty_metrics {
        // Match the historical live boundary: a cancellation that removes
        // the last scheduler-owned request updates occupancy immediately,
        // even though the modeled pass still owns its completion boundary.
        // KV and lifecycle effects remain deferred with that pass.
        dispatch.publish_metrics(metrics);
    }
    Ok(())
}

fn scheduler_command_result(
    result: CommandResult,
    suppressed_pending_output: bool,
) -> SchedulerCommandResult {
    match result {
        CommandResult::Submitted(request_id) => SchedulerCommandResult::Submitted(request_id),
        CommandResult::DestinationAccepted { request_id } => {
            SchedulerCommandResult::DestinationAccepted { request_id }
        }
        CommandResult::Applied => SchedulerCommandResult::Applied,
        // The native request may already have retired into the in-flight pass
        // while its output is still pending. Suppressing that retained output
        // is observable cancellation work at Dynamo's compatibility boundary.
        CommandResult::Noop if suppressed_pending_output => SchedulerCommandResult::Applied,
        CommandResult::Noop => SchedulerCommandResult::Noop,
    }
}

fn rank_dispatch(ranks: &[RankDispatch], dp_rank: u32) -> Result<&RankDispatch> {
    ranks.get(dp_rank as usize).ok_or_else(|| {
        anyhow!(
            "grouped live effect referenced DP rank {dp_rank}, but only {} ranks exist",
            ranks.len()
        )
    })
}

impl RankDispatch {
    async fn publish_admissions(&self, admissions: Vec<Admission>) -> Result<()> {
        let Some(sender) = self.event_tx.as_ref() else {
            return Ok(());
        };
        let admissions = admissions
            .into_iter()
            .map(|admission| AdmissionEvent {
                uuid: admission.request_id,
                reused_input_tokens: admission.reused_input_tokens,
            })
            .collect::<Vec<_>>();
        match sender.send_admissions(&admissions).await {
            Ok(()) | Err(SchedulerEventSendError::Cancelled) => Ok(()),
            Err(SchedulerEventSendError::OrderedLaneClosed) => {
                bail!("grouped live ordered admission lane is closed")
            }
            Err(SchedulerEventSendError::OutputClosed(_)) => {
                bail!("grouped live admission unexpectedly used an output-only lane")
            }
        }
    }

    /// Publish output and return requests whose output-only consumer closed.
    async fn publish_outputs(&self, outputs: Vec<OutputSignal>) -> Result<OutputPublication> {
        let Some(sender) = self.event_tx.as_ref() else {
            return Ok(OutputPublication::Delivered(Vec::new()));
        };
        if outputs.is_empty() {
            return Ok(OutputPublication::Delivered(Vec::new()));
        }
        match sender.send_outputs(outputs).await {
            Ok(()) => Ok(OutputPublication::Delivered(Vec::new())),
            Err(SchedulerEventSendError::OutputClosed(signals)) => {
                Ok(OutputPublication::Delivered(
                    signals
                        .into_iter()
                        .map(|signal| signal.uuid)
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                ))
            }
            Err(SchedulerEventSendError::OrderedLaneClosed) => {
                bail!("grouped live ordered output lane is closed")
            }
            Err(SchedulerEventSendError::Cancelled) => Ok(OutputPublication::Cancelled),
        }
    }

    fn publish_kv(&self, events: Vec<KvEvent>) {
        if events.is_empty() {
            return;
        }
        let mut raw_events = Vec::with_capacity(events.len());
        for event in events {
            if event.dp_rank != self.external_dp_rank {
                tracing::warn!(
                    expected_dp_rank = self.external_dp_rank,
                    event_dp_rank = event.dp_rank,
                    "dropping native KV event with mismatched DP rank"
                );
                continue;
            }
            let (event, block_token_ids) = dynamo_kv_event(event);
            raw_events.push(RawKvEvent {
                event,
                block_token_ids,
                storage_tier: StorageTier::Device,
            });
        }
        let normal_events = raw_events
            .iter()
            .map(|event| (event.event.clone(), event.storage_tier))
            .collect();
        if let Err(error) = self
            .kv_event_publishers
            .publish_event_sink_batch_only(normal_events)
        {
            tracing::warn!(dp_rank = self.external_dp_rank, error = ?error, "failed to publish grouped native KV events");
        }
        if let Err(error) = self.kv_event_publishers.publish_raw_batch(raw_events) {
            tracing::warn!(dp_rank = self.external_dp_rank, error = ?error, "failed to publish grouped raw KV events");
        }
    }

    fn publish_fpm(&self, metrics: ForwardPassMetrics) {
        let snapshot = dynamo_forward_pass_snapshot(self.external_dp_rank, metrics);
        if let Err(error) = self.fpm_publisher.publish(snapshot) {
            tracing::warn!(dp_rank = self.external_dp_rank, error = ?error, "failed to publish grouped forward-pass metrics");
        }
    }

    async fn publish_lifecycle(&self, events: Vec<SchedulerLifecycleEvent>) {
        for event in events {
            if self.lifecycle_tx.send(event).await.is_err() {
                return;
            }
        }
    }

    pub(super) fn publish_metrics(&self, metrics: Metrics) {
        let _ = self.metrics_tx.send(MockerMetrics {
            dp_rank: metrics.dp_rank,
            active_decode_blocks: metrics.active_blocks,
            total_blocks: metrics.total_blocks,
            gpu_cache_usage_perc: metrics.cache_usage,
            running_requests: metrics.running_requests,
            waiting_requests: metrics.waiting_requests,
            vllm_preemptions_total: metrics.preemptions_total,
            sglang_cache_hit_tokens: metrics.sglang_cache_hit_tokens,
            sglang_cache_total_tokens: metrics.sglang_cache_total_tokens,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ready_boundary_error_wins_over_cancellation() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        let error = finish_boundary_or_cancel(
            async { Err(anyhow!("unexpected boundary failure")) },
            &cancel,
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("unexpected boundary failure"));
    }

    #[tokio::test]
    async fn cancellation_ends_a_pending_boundary_wait_orderly() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        finish_boundary_or_cancel(std::future::pending(), &cancel)
            .await
            .unwrap();
    }
}
