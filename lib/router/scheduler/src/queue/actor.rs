// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The actor's loop: one turn per command, capacity wake or timer.

use super::*;

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    /// Runs one turn per command, capacity wake, or class timer until every sender is gone.
    pub(super) async fn run(mut self) {
        let mut commands_since_cleanup = 0usize;
        let capacity_updates = Arc::clone(&self.capacity_updates);
        let mut affinity_recheck = tokio::time::interval(AFFINITY_RECHECK_INTERVAL);
        affinity_recheck.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            self.turn_prepared.begin_turn();
            let due_at = match (self.class_timers.next_deadline(), self.gate_recheck_at) {
                (Some(class_due), Some(gate_due)) => Some(class_due.min(gate_due)),
                (class_due, gate_due) => class_due.or(gate_due),
            };
            let has_affinity_backlog = self.pending.has_affinity_backlog();
            let command = tokio::select! {
                command = self.commands.next() => command,
                _ = affinity_recheck.tick(), if has_affinity_backlog => {
                    self.pending.recheck_affinity_lanes();
                    self.handle_update(None, None).await;
                    continue;
                }
                _ = capacity_updates.notified() => {
                    let drain_cleanup =
                        self.commands.is_empty() || commands_since_cleanup == 256;
                    if drain_cleanup {
                        commands_since_cleanup = 0;
                    }
                    self.handle_capacity_updates(drain_cleanup).await;
                    continue;
                }
                _ = async {
                    if let Some(due_at) = due_at {
                        tokio::time::sleep_until(due_at).await;
                    }
                }, if due_at.is_some() => {
                    // A class deadline expires its roots, which unblocks their lanes on its own;
                    // only the gate's hold timer re-asks the lanes it holds.
                    let gate_due = self.gate_recheck_at.is_some_and(|at| at <= Instant::now());
                    if gate_due {
                        self.gate_recheck_at = None;
                    }
                    self.handle_update(None, gate_due.then_some(RecheckScope::All)).await;
                    continue;
                }
            };
            let Some(command) = command else { break };
            commands_since_cleanup += 1;
            let drain_cleanup = self.commands.is_empty() || commands_since_cleanup == 256;
            if drain_cleanup {
                commands_since_cleanup = 0;
            }
            let (drain, reply) = self.apply_command(command, drain_cleanup);
            self.drain_after_command(drain, reply).await;
        }
        self.drain_cleanup();

        let class_counters = Arc::clone(&self.class_counters);
        for entry in self.pending.drain() {
            class_counters[entry.class_index()].sub(entry.snapshot());
            let mut request = entry.into_payload().request;
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
        }
    }

    pub(super) async fn drain_after_command(&mut self, drain: Drain, reply: Option<CommandReply>) {
        match drain {
            Drain::None => {
                if let Some(reply) = reply {
                    reply.send();
                }
            }
            Drain::Enqueued => self.handle_update(reply, None).await,
            Drain::Full => self.handle_update(reply, Some(RecheckScope::All)).await,
        }
    }

    /// Applies one command without draining: returns the drain it calls for and its reply.
    pub(super) fn apply_command(
        &mut self,
        command: AdmissionCommand,
        drain_cleanup: bool,
    ) -> (Drain, Option<CommandReply>) {
        match command {
            AdmissionCommand::Enqueue {
                request,
                attempt_tx,
                block_hashes,
                prepared,
                queue_metadata,
                lease,
                ack_tx,
            } => {
                let lifecycle_transfer = lease
                    .as_ref()
                    .and_then(|lease| lease.transfer.as_ref().map(Arc::clone));
                // Released entries leave before the arrival's class limits count them.
                let cleanup_ready = drain_cleanup && self.drain_cleanup();
                let enqueue_ready = self.handle_enqueue(
                    request,
                    attempt_tx,
                    lifecycle_transfer,
                    block_hashes,
                    prepared,
                    queue_metadata,
                );
                let drain = if cleanup_ready {
                    Drain::Full
                } else if enqueue_ready {
                    Drain::Enqueued
                } else {
                    Drain::None
                };
                (drain, Some(CommandReply::Enqueue { ack_tx, lease }))
            }
            AdmissionCommand::SelectWithoutAdmission { request, resp_tx } => {
                let result = self.select_without_admission_inner(request, Instant::now());
                (
                    Drain::full_if(drain_cleanup && self.drain_cleanup()),
                    Some(CommandReply::SelectWithoutAdmission { resp_tx, result }),
                )
            }
            AdmissionCommand::PolicyConfig { update, response } => {
                // A cancelled or timed-out caller must not leave a queued mutation behind.
                if !response.is_closed() {
                    let result = match update {
                        Some(update) => {
                            let result = self.selector.update_runtime_config(update);
                            if result.is_ok() {
                                // A replacement may need fewer inputs; prepare what it reads.
                                self.candidate_preparer
                                    .store(self.selector.candidate_preparer().map(Arc::new));
                            }
                            result
                        }
                        None => self.selector.runtime_config(),
                    };
                    let _ = response.send(result);
                }
                (Drain::full_if(drain_cleanup && self.drain_cleanup()), None)
            }
            AdmissionCommand::Update { ack_tx } => {
                if drain_cleanup {
                    self.drain_cleanup();
                }
                (Drain::Full, Some(CommandReply::Update { ack_tx }))
            }
            AdmissionCommand::Cleanup => (Drain::full_if(self.drain_cleanup()), None),
        }
    }
}
