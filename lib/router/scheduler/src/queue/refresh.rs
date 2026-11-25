// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The optional overlap refresh of a request that waited in the queue.

use super::*;

/// Scheduler work that arrives during an optional overlap refresh.
#[allow(clippy::large_enum_variant)]
enum RefreshEvent {
    ClassTimer,
    Capacity,
    Command(AdmissionCommand),
}

/// How the optional overlap refresh of an extracted candidate ended. Every ending but `Finished`
/// drops the refresh unfinished, and a candidate still dispatched keeps its queued scores. A
/// transient return value, never stored, so its refreshed scores stay unboxed.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(super) enum RefreshEnd {
    /// The scores to use, or `None` to keep the queued ones.
    Finished(Option<RefreshedOverlap>),
    /// The candidate's due time passed first.
    DueTimePassed,
    /// The candidate's response closed first.
    ResponseClosed,
    /// The caller released the candidate's lifecycle first. The candidate leaves unanswered,
    /// as a queued entry that cleanup removes does.
    Released,
    /// Every command sender is gone. The actor stops once the candidate is settled.
    Shutdown,
    /// The refresh outlasted [`OVERLAP_REFRESH_TIMEOUT`].
    TimedOut,
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    /// Runs the optional overlap refresh of an extracted candidate while class timers, capacity
    /// wakes, and commands are handled, each in a new actor turn. A command applied here is
    /// answered at once; the drain it calls for is the one that resumes. The candidate has left
    /// its class queue, so class expiry no longer covers it: the refresh races its own due time.
    /// That due time, its response closing, its lifecycle cleanup, shutdown, or the refresh
    /// timeout (see [`OverlapScoresRefresh`]) ends the refresh early.
    pub(super) async fn refresh_candidate(
        &mut self,
        queued: &mut QueuedRequest,
        decay_now: Instant,
    ) -> RefreshEnd {
        let refresher = self.overlap_scores_refresh.clone();
        // One timeout for the whole refresh: the events handled below never restart it.
        let refresh = tokio::time::timeout(
            OVERLAP_REFRESH_TIMEOUT,
            refresh_overlap(
                refresher.as_deref(),
                self.overlap_refresh_after,
                queued.block_hashes.as_deref(),
                queued.request.retain_kv_transfer_chain,
                queued.enqueue_at,
                decay_now,
            ),
        );
        tokio::pin!(refresh);
        let capacity_updates = Arc::clone(&self.capacity_updates);
        let due_at = queued.due_at;
        loop {
            let timer = self.class_timers.next_deadline();
            let response = &mut queued.request.resp_tx;
            let event = tokio::select! {
                biased;
                refreshed = &mut refresh => {
                    return refreshed.map_or(RefreshEnd::TimedOut, RefreshEnd::Finished);
                }
                _ = async {
                    if let Some(response) = response {
                        response.closed().await;
                    }
                } => return RefreshEnd::ResponseClosed,
                _ = async {
                    if let Some(due_at) = due_at {
                        tokio::time::sleep_until(due_at).await;
                    }
                }, if due_at.is_some() => return RefreshEnd::DueTimePassed,
                _ = async {
                    if let Some(timer) = timer {
                        tokio::time::sleep_until(timer).await;
                    }
                }, if timer.is_some() => RefreshEvent::ClassTimer,
                _ = capacity_updates.notified() => RefreshEvent::Capacity,
                command = self.commands.next() => match command {
                    Some(command) => RefreshEvent::Command(command),
                    None => return RefreshEnd::Shutdown,
                },
            };
            self.turn_prepared.begin_turn();
            match event {
                RefreshEvent::ClassTimer => {
                    self.expire_due_classes(Instant::now());
                }
                RefreshEvent::Capacity => {
                    let cleanup_ready = self.drain_cleanup();
                    self.recheck_capacity_updates(cleanup_ready.then_some(RecheckScope::All));
                }
                RefreshEvent::Command(command) => {
                    let (drain, reply) = self.apply_command(command, true);
                    match drain {
                        Drain::None => {}
                        Drain::Enqueued => self.recheck_capacity_updates(None),
                        Drain::Full => self.recheck_capacity_updates(Some(RecheckScope::All)),
                    }
                    if let Some(reply) = reply {
                        reply.send();
                    }
                }
            }
            // Cleanup may have released the candidate's own lifecycle. Its caller is gone, and a
            // booking made now would have no owner to free it.
            if queued
                .lifecycle_transfer
                .as_ref()
                .is_some_and(|transfer| !transfer.is_armed())
            {
                return RefreshEnd::Released;
            }
        }
    }
}
