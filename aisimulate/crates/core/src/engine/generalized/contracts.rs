// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use anyhow::Result;

/// Stable identity of one logical mock engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineIdentity {
    /// Stable logical worker ID.
    pub worker_id: u64,
}

impl EngineIdentity {
    /// Construct an engine identity from its stable worker ID.
    pub const fn new(worker_id: u64) -> Self {
        Self { worker_id }
    }

    /// Return the identity of one attention-DP rank.
    pub const fn rank(self, dp_rank: u32, dp_size: NonZeroU32) -> RankIdentity {
        RankIdentity {
            worker_id: self.worker_id,
            dp_rank,
            dp_size,
        }
    }
}

/// Stable identity of one scheduler core within a logical engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RankIdentity {
    /// Stable logical worker ID shared by sibling ranks.
    pub worker_id: u64,
    /// Zero-based attention-DP rank.
    pub dp_rank: u32,
    /// Total ranks in this logical engine's attention-DP group.
    pub dp_size: NonZeroU32,
}

/// Construction parameters for a generalized engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralizedEngineConfig<C> {
    /// Number of scheduler cores in the attention-DP group.
    pub dp_size: NonZeroU32,
    /// Configuration cloned into each rank core.
    pub rank: C,
}

impl<C> GeneralizedEngineConfig<C> {
    /// Construct a single-rank engine configuration.
    pub const fn single_rank(rank: C) -> Self {
        Self {
            dp_size: NonZeroU32::MIN,
            rank,
        }
    }

    /// Construct an attention-DP engine configuration.
    pub const fn attention_dp(dp_size: NonZeroU32, rank: C) -> Self {
        Self { dp_size, rank }
    }
}

/// Context supplied while a rank applies a scheduler command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommandContext {
    /// Driver clock at which the command becomes visible.
    pub now_ms: f64,
    /// Whether this logical engine has a committed pass awaiting completion.
    ///
    /// This is group-wide. It is `true` for an otherwise idle sibling rank
    /// while another rank is executing, preserving the attention-DP barrier.
    pub pass_in_flight: bool,
}

impl CommandContext {
    /// Whether a command may immediately admit work into the current pass.
    pub const fn allow_immediate_admission(self) -> bool {
        !self.pass_in_flight
    }
}

/// A command addressed to one attention-DP rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerCommand<C> {
    /// Target attention-DP rank.
    pub dp_rank: u32,
    /// Rank-engine-specific, runtime-neutral command payload.
    pub command: C,
}

impl<C> SchedulerCommand<C> {
    /// Address a command to a rank.
    pub const fn new(dp_rank: u32, command: C) -> Self {
        Self { dp_rank, command }
    }
}

/// Effects produced by one rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankEffects<T> {
    /// Rank that produced the effects.
    pub dp_rank: u32,
    /// Rank-engine-specific effects.
    pub effects: T,
}

/// Effects produced by zero or more ranks of a logical engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineEffects<T> {
    /// Effects in stable DP-rank order.
    pub by_rank: Vec<RankEffects<T>>,
}

impl<T> EngineEffects<T> {
    fn empty() -> Self {
        Self {
            by_rank: Vec::new(),
        }
    }

    pub(crate) fn one(dp_rank: u32, effects: T) -> Self {
        Self {
            by_rank: vec![RankEffects { dp_rank, effects }],
        }
    }

    /// Return `true` when no rank produced effects.
    pub fn is_empty(&self) -> bool {
        self.by_rank.is_empty()
    }

    /// Consume the wrapper and return effects in stable DP-rank order.
    pub fn into_by_rank(self) -> Vec<RankEffects<T>> {
        self.by_rank
    }
}

impl<T> Default for EngineEffects<T> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Opaque ID of a committed logical-engine pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassId(pub(crate) u64);

impl PassId {
    /// Return the monotonically increasing per-engine sequence number.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Rank-local status for a bounded same-timestamp scheduler retry.
///
/// This is a driver hint, not an observable effect or ordinary progress.
/// Drivers evaluate it only after an effect-free, zero-duration pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SameTimestampRetry {
    /// This rank has no internal same-timestamp convergence protocol.
    #[default]
    NotApplicable,
    /// Internal scheduler state changed and another pass may expose work.
    Retry,
    /// The rank's internal state no longer changes at this timestamp.
    Exhausted,
}

/// Eager execution result returned by a rank core.
///
/// `start_effects` may become visible immediately. `pending` is retained by
/// the generalized engine and handed back to the rank only at the shared
/// completion boundary.
#[derive(Debug)]
pub struct RankPass<S, P> {
    /// Modeled completion time for this rank.
    pub end_ms: f64,
    /// Rank-local bounded retry status.
    pub same_timestamp_retry: SameTimestampRetry,
    /// Effects visible at pass start.
    pub start_effects: S,
    /// Rank-private state needed to finish the pass.
    pub pending: P,
}

/// Pass-start effects from one rank.
#[derive(Debug, Clone, PartialEq)]
pub struct RankPassStarted<T> {
    /// Rank that executed work.
    pub dp_rank: u32,
    /// This rank's modeled completion time before group alignment.
    pub rank_end_ms: f64,
    /// Effects visible at pass start.
    pub effects: T,
}

/// A logical pass committed by
/// [`GeneralizedMockerEngine::execute_pass`](super::GeneralizedMockerEngine::execute_pass).
#[derive(Debug, Clone, PartialEq)]
pub struct EnginePassStarted<T> {
    /// ID supplied later to
    /// [`GeneralizedMockerEngine::complete_pass`](super::GeneralizedMockerEngine::complete_pass).
    pub pass_id: PassId,
    /// Driver time at which the pass was committed.
    pub started_at_ms: f64,
    /// Shared completion boundary, equal to the slowest executed rank.
    pub end_ms: f64,
    /// Total sibling ranks held by the barrier, including idle ranks.
    pub participating_ranks: NonZeroU32,
    /// Grouped retry status at [`Self::started_at_ms`]. A retry request from
    /// any executed rank takes precedence over an exhausted sibling.
    pub same_timestamp_retry: SameTimestampRetry,
    /// Start effects from ranks that had work, in stable rank order.
    pub by_rank: Vec<RankPassStarted<T>>,
}

/// A logical pass released at its shared completion boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct EnginePassCompleted<T> {
    /// ID of the completed pass.
    pub pass_id: PassId,
    /// Completion effects from executed ranks and idle siblings that released
    /// deferred work at the shared group boundary.
    pub effects: EngineEffects<T>,
}

/// One scheduler/KV/timing core.
///
/// Implementations own all single-rank scheduler state. The generalized layer
/// owns attention-DP grouping and never inspects the command/effect payloads.
///
/// `execute_pass` eagerly commits a non-preemptive batch. Implementations must
/// not expose pass-end effects until `complete_pass` receives the retained
/// `PendingPass`.
pub trait RankEngine: Sized {
    /// Rank construction configuration.
    type Config;
    /// Scheduler command payload.
    type Command;
    /// Effects of applying one command.
    type CommandEffects;
    /// Effects visible when a pass starts.
    type PassStartEffects;
    /// Opaque state retained between pass start and completion.
    type PendingPass;
    /// Effects visible when a pass completes.
    type PassCompletionEffects;
    /// Effects produced by deadline-driven internal work.
    type InternalEffects;

    /// Construct one rank core.
    fn new(identity: RankIdentity, config: &Self::Config) -> Result<Self>;

    /// Apply one scheduler command.
    ///
    /// `pending_pass` is the eagerly committed pass for this rank, when this
    /// rank participated in the logical engine's current in-flight pass.
    /// Commands such as cancellation may mutate it to suppress effects that
    /// were computed at pass start but must no longer become visible at pass
    /// completion. A logical attention-DP pass can be in flight while this is
    /// `None` when only sibling ranks participated; use
    /// [`CommandContext::pass_in_flight`] for the group-wide state.
    ///
    /// This operation must be error-atomic: returning `Err` must leave both
    /// the rank and `pending_pass` unchanged. Command errors are recoverable
    /// at the generalized boundary because a command targets only one rank;
    /// use a successful command effect to represent any committed mutation.
    fn apply_command_effects(
        &mut self,
        command: Self::Command,
        context: CommandContext,
        pending_pass: Option<&mut Self::PendingPass>,
    ) -> Result<Self::CommandEffects>;

    /// Whether this rank can commit a pass.
    fn is_ready(&self) -> bool;

    /// Whether a ready rank is blocked only on externally commanded state.
    ///
    /// This is narrower than having queued work. Implementations return true
    /// only when a later command can release retained ownership that prevents
    /// the ready work from advancing. Drivers use this signal to avoid
    /// repeatedly executing an effect-free, zero-duration pass while keeping
    /// genuine scheduler livelocks visible.
    fn waiting_for_external_command(&self) -> bool {
        false
    }

    /// Eagerly commit one non-preemptive pass.
    fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<RankPass<Self::PassStartEffects, Self::PendingPass>>;

    /// Release the effects of a previously committed pass.
    fn complete_pass(
        &mut self,
        pending: Self::PendingPass,
        end_ms: f64,
    ) -> Result<Self::PassCompletionEffects>;

    /// Cross the shared completion boundary without a rank-local pass.
    ///
    /// Attention-DP ranks that had no work when a sibling pass started still
    /// participate in the group barrier. Implementations may use this hook to
    /// release effects deferred while [`CommandContext::pass_in_flight`] was
    /// true. `end_ms` is the modeled shared group boundary (the maximum rank
    /// end), even when a wall-clock driver calls `complete_pass` later.
    /// Returning `None` means the idle rank has no boundary effects.
    fn complete_idle_group_pass(
        &mut self,
        _started_at_ms: f64,
        _end_ms: f64,
    ) -> Result<Option<Self::PassCompletionEffects>> {
        Ok(None)
    }

    /// Earliest deadline for independently modeled internal work.
    fn next_internal_deadline_ms(&self) -> Option<f64>;

    /// Process internal work due at `now_ms`.
    fn process_internal_work(
        &mut self,
        now_ms: f64,
        pass_in_flight: bool,
    ) -> Result<Self::InternalEffects>;

    /// Whether this rank owns no request, pass, or internal work.
    fn is_drained(&self) -> bool;
}
