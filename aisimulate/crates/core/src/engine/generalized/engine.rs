// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use anyhow::{Context, Result, bail, ensure};

use super::contracts::{
    CommandContext, EngineEffects, EngineIdentity, EnginePassCompleted, EnginePassStarted,
    GeneralizedEngineConfig, PassId, RankEffects, RankEngine, RankIdentity, RankPassStarted,
    SameTimestampRetry, SchedulerCommand,
};

struct PendingRankPass<P> {
    dp_rank: u32,
    pending: P,
}

struct PendingGroupPass<P> {
    pass_id: PassId,
    started_at_ms: f64,
    end_ms: f64,
    by_rank: Vec<PendingRankPass<P>>,
}

/// Single-rank or attention-DP grouped generalized mock engine.
///
/// `dp_size == 1` is the single-rank layer. Larger values compose independent
/// rank cores behind one logical-worker barrier.
///
/// Multi-rank mutations are not transactional across a group. If a grouped
/// operation fails after an earlier rank may have changed state, the logical
/// engine becomes poisoned and rejects later mutations. A targeted command is
/// delegated to exactly one rank; its implementation owns command atomicity,
/// and ordinary command rejections remain recoverable.
pub struct GeneralizedMockerEngine<C: RankEngine> {
    identity: EngineIdentity,
    dp_size: NonZeroU32,
    ranks: Vec<C>,
    next_pass_id: u64,
    pending_pass: Option<PendingGroupPass<C::PendingPass>>,
    poisoned: Option<String>,
}

impl<C: RankEngine> GeneralizedMockerEngine<C> {
    /// Stable identity of this logical worker.
    pub const fn identity(&self) -> EngineIdentity {
        self.identity
    }

    /// Number of scheduler ranks composed behind the group barrier.
    pub const fn dp_size(&self) -> NonZeroU32 {
        self.dp_size
    }

    /// Stable identities of the scheduler ranks in DP-rank order.
    pub fn rank_identities(&self) -> impl ExactSizeIterator<Item = RankIdentity> + '_ {
        (0..self.dp_size.get()).map(|dp_rank| self.identity.rank(dp_rank, self.dp_size))
    }

    /// Construct every rank in a logical engine.
    pub fn new(
        identity: EngineIdentity,
        config: GeneralizedEngineConfig<C::Config>,
    ) -> Result<Self> {
        Self::new_with_rank_factory(identity, config.dp_size, |rank_identity| {
            C::new(rank_identity, &config.rank)
        })
    }

    /// Construct every rank with a caller-supplied factory.
    ///
    /// This is the runtime-provider seam for rank implementations whose
    /// serialized configuration names a provider but cannot itself contain a
    /// process-local callback, such as an AIC latency model.
    pub fn new_with_rank_factory(
        identity: EngineIdentity,
        dp_size: NonZeroU32,
        mut make_rank: impl FnMut(RankIdentity) -> Result<C>,
    ) -> Result<Self> {
        let mut ranks = Vec::with_capacity(dp_size.get() as usize);
        for dp_rank in 0..dp_size.get() {
            ranks.push(
                make_rank(identity.rank(dp_rank, dp_size))
                    .with_context(|| format!("constructing attention-DP rank {dp_rank}"))?,
            );
        }
        Ok(Self {
            identity,
            dp_size,
            ranks,
            next_pass_id: 0,
            pending_pass: None,
            poisoned: None,
        })
    }

    /// Apply a command to one rank.
    pub fn apply_command_effects(
        &mut self,
        command: SchedulerCommand<C::Command>,
        now_ms: f64,
    ) -> Result<EngineEffects<C::CommandEffects>> {
        self.ensure_healthy()?;
        validate_time(now_ms, "command time")?;
        let pass_in_flight = self.pending_pass.is_some();
        let dp_rank = command.dp_rank;
        ensure!(
            (dp_rank as usize) < self.ranks.len(),
            "attention-DP rank {dp_rank} is out of range for dp_size {}",
            self.dp_size
        );
        let effects = {
            let rank = &mut self.ranks[dp_rank as usize];
            let pending_pass = self.pending_pass.as_mut().and_then(|group| {
                group
                    .by_rank
                    .iter_mut()
                    .find(|pending| pending.dp_rank == dp_rank)
                    .map(|pending| &mut pending.pending)
            });
            rank.apply_command_effects(
                command.command,
                CommandContext {
                    now_ms,
                    pass_in_flight,
                },
                pending_pass,
            )
            .with_context(|| format!("applying command to attention-DP rank {dp_rank}"))
        };
        // A command targets exactly one rank, and schedulers use command
        // errors for recoverable admission rejections (for example, a prompt
        // larger than the destination KV pool). Poisoning the whole logical
        // worker here would convert that normal handoff failure into a replay
        // dead end. Rank implementations must therefore keep command errors
        // atomic; fail-stop poisoning is reserved for the grouped operations
        // below, where an earlier sibling may already have committed.
        let effects = effects?;
        Ok(EngineEffects::one(dp_rank, effects))
    }

    /// Whether the logical engine can commit a grouped pass.
    ///
    /// No sibling may start a new pass while an earlier grouped pass is
    /// awaiting completion. Otherwise, one ready rank that is not blocked on
    /// externally commanded ownership makes the group ready. A held source or
    /// reserved destination remains owned by the engine, but must not create
    /// effect-free passes while it waits for release/activation.
    pub fn is_ready(&self) -> bool {
        self.poisoned.is_none()
            && self.pending_pass.is_none()
            && self
                .ranks
                .iter()
                .any(|rank| rank.is_ready() && !rank.waiting_for_external_command())
    }

    /// Whether every currently ready rank is blocked on an external command.
    ///
    /// Attention-DP uses `all`, not `any`: an unrelated ready sibling must
    /// still be allowed to expose a scheduler livelock or make progress.
    pub fn waiting_for_external_command(&self) -> bool {
        if self.poisoned.is_some() || self.pending_pass.is_some() {
            return false;
        }

        let mut found_ready_rank = false;
        for rank in &self.ranks {
            if !rank.is_ready() {
                continue;
            }
            found_ready_rank = true;
            if !rank.waiting_for_external_command() {
                return false;
            }
        }
        found_ready_rank
    }

    /// Eagerly commit one pass on every currently ready rank.
    ///
    /// Returns `None` when no rank has work. Starting a second pass before
    /// completing the first is a caller error.
    pub fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<Option<EnginePassStarted<C::PassStartEffects>>> {
        self.ensure_healthy()?;
        validate_time(now_ms, "pass start time")?;
        ensure!(
            self.pending_pass.is_none(),
            "engine {} already has a pass in flight",
            self.identity.worker_id
        );

        if !self
            .ranks
            .iter()
            .any(|rank| rank.is_ready() && !rank.waiting_for_external_command())
        {
            return Ok(None);
        }

        let pass_id = PassId(self.next_pass_id);
        let next_pass_id = self
            .next_pass_id
            .checked_add(1)
            .context("generalized engine pass ID overflow")?;

        let mut end_ms = now_ms;
        let mut same_timestamp_retry = SameTimestampRetry::NotApplicable;
        let mut started = Vec::new();
        let mut pending = Vec::new();
        for dp_rank in 0..self.ranks.len() {
            if !self.ranks[dp_rank].is_ready() || self.ranks[dp_rank].waiting_for_external_command()
            {
                continue;
            }
            let pass = {
                let rank = &mut self.ranks[dp_rank];
                rank.execute_pass(now_ms)
                    .with_context(|| format!("executing attention-DP rank {dp_rank}"))
                    .and_then(|pass| {
                        validate_time(pass.end_ms, "rank pass end time")?;
                        ensure!(
                            pass.end_ms >= now_ms,
                            "attention-DP rank {dp_rank} completed before its pass started"
                        );
                        Ok(pass)
                    })
            };
            let pass = pass.map_err(|error| self.poison(error))?;
            end_ms = end_ms.max(pass.end_ms);
            same_timestamp_retry = match (same_timestamp_retry, pass.same_timestamp_retry) {
                (_, SameTimestampRetry::Retry) => SameTimestampRetry::Retry,
                (SameTimestampRetry::Retry, _) => SameTimestampRetry::Retry,
                (_, SameTimestampRetry::Exhausted) => SameTimestampRetry::Exhausted,
                (status, SameTimestampRetry::NotApplicable) => status,
            };
            started.push(RankPassStarted {
                dp_rank: dp_rank as u32,
                rank_end_ms: pass.end_ms,
                effects: pass.start_effects,
            });
            pending.push(PendingRankPass {
                dp_rank: dp_rank as u32,
                pending: pass.pending,
            });
        }

        debug_assert!(!pending.is_empty());
        self.next_pass_id = next_pass_id;
        self.pending_pass = Some(PendingGroupPass {
            pass_id,
            started_at_ms: now_ms,
            end_ms,
            by_rank: pending,
        });
        Ok(Some(EnginePassStarted {
            pass_id,
            started_at_ms: now_ms,
            end_ms,
            participating_ranks: self.dp_size,
            same_timestamp_retry,
            by_rank: started,
        }))
    }

    /// Complete the committed pass and release pass-end effects.
    ///
    /// `end_ms` may be later than the modeled boundary (for example when a
    /// wall-clock driver wakes late), but never earlier.
    pub fn complete_pass(
        &mut self,
        pass_id: PassId,
        end_ms: f64,
    ) -> Result<EnginePassCompleted<C::PassCompletionEffects>> {
        self.ensure_healthy()?;
        validate_time(end_ms, "pass completion time")?;
        let pending = self
            .pending_pass
            .as_ref()
            .context("cannot complete a pass when none is in flight")?;
        ensure!(
            pending.pass_id == pass_id,
            "pass ID mismatch: expected {}, got {}",
            pending.pass_id.get(),
            pass_id.get()
        );
        ensure!(
            end_ms >= pending.end_ms,
            "pass {} completed at {end_ms}ms before its modeled boundary {}ms",
            pass_id.get(),
            pending.end_ms
        );

        let pending = self
            .pending_pass
            .take()
            .expect("pending pass was checked immediately before take");
        let mut pending_by_rank = pending.by_rank.into_iter().peekable();
        let mut effects = Vec::with_capacity(self.ranks.len());
        for rank_index in 0..self.ranks.len() {
            let dp_rank = rank_index as u32;
            if pending_by_rank
                .peek()
                .is_some_and(|pending| pending.dp_rank == dp_rank)
            {
                let rank_pass = pending_by_rank
                    .next()
                    .expect("peeked pending rank pass must remain available");
                let rank_effects = self.ranks[rank_index]
                    // Rank-local FPM is normalized to the modeled shared
                    // barrier. A wall-clock driver's late wakeup is accepted
                    // above, but must not inflate modeled execution time.
                    .complete_pass(rank_pass.pending, pending.end_ms)
                    .with_context(|| format!("completing attention-DP rank {dp_rank}"));
                let rank_effects = rank_effects.map_err(|error| self.poison(error))?;
                effects.push(RankEffects {
                    dp_rank,
                    effects: rank_effects,
                });
                continue;
            }
            let idle_effects = self.ranks[rank_index]
                // Idle-rank FPM represents the modeled shared barrier, not a
                // live driver's scheduling delay after that barrier elapsed.
                .complete_idle_group_pass(pending.started_at_ms, pending.end_ms)
                .with_context(|| {
                    format!("completing idle attention-DP rank {dp_rank} group barrier")
                });
            let idle_effects = idle_effects.map_err(|error| self.poison(error))?;
            if let Some(rank_effects) = idle_effects {
                effects.push(RankEffects {
                    dp_rank,
                    effects: rank_effects,
                });
            }
        }
        debug_assert!(pending_by_rank.next().is_none());
        Ok(EnginePassCompleted {
            pass_id,
            effects: EngineEffects { by_rank: effects },
        })
    }

    /// Earliest valid internal-work deadline across all ranks.
    pub fn next_internal_deadline_ms(&self) -> Option<f64> {
        if self.poisoned.is_some() {
            return None;
        }
        self.ranks
            .iter()
            .filter_map(RankEngine::next_internal_deadline_ms)
            .filter(|deadline| deadline.is_finite())
            .min_by(f64::total_cmp)
    }

    /// Process internal work whose rank deadline is due.
    pub fn process_internal_work(
        &mut self,
        now_ms: f64,
    ) -> Result<EngineEffects<C::InternalEffects>> {
        self.ensure_healthy()?;
        validate_time(now_ms, "internal-work time")?;
        let pass_in_flight = self.pending_pass.is_some();
        let mut effects = Vec::new();
        for dp_rank in 0..self.ranks.len() {
            let is_due = self.ranks[dp_rank]
                .next_internal_deadline_ms()
                .is_some_and(|deadline| deadline.is_finite() && deadline <= now_ms);
            if !is_due {
                continue;
            }
            let rank_effects = self.ranks[dp_rank]
                .process_internal_work(now_ms, pass_in_flight)
                .with_context(|| {
                    format!("processing internal work for attention-DP rank {dp_rank}")
                });
            let rank_effects = rank_effects.map_err(|error| self.poison(error))?;
            effects.push(RankEffects {
                dp_rank: dp_rank as u32,
                effects: rank_effects,
            });
        }
        Ok(EngineEffects { by_rank: effects })
    }

    /// Whether every rank is drained and no grouped pass remains in flight.
    pub fn is_drained(&self) -> bool {
        self.poisoned.is_none()
            && self.pending_pass.is_none()
            && self.ranks.iter().all(RankEngine::is_drained)
    }

    fn ensure_healthy(&self) -> Result<()> {
        if let Some(reason) = &self.poisoned {
            bail!(
                "generalized engine worker {} is poisoned after a prior rank mutation failure: {reason}",
                self.identity.worker_id
            );
        }
        Ok(())
    }

    fn poison(&mut self, error: anyhow::Error) -> anyhow::Error {
        let reason = format!("{error:#}");
        self.poisoned.get_or_insert_with(|| reason.clone());
        error.context(format!(
            "generalized engine worker {} is now poisoned because a rank may have been partially mutated",
            self.identity.worker_id
        ))
    }
}

fn validate_time(time_ms: f64, label: &str) -> Result<()> {
    if !time_ms.is_finite() || time_ms < 0.0 {
        bail!("{label} must be finite and non-negative, got {time_ms}");
    }
    Ok(())
}
