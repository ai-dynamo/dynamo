// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../../docs/scheduled_sequence.md")]

use std::sync::Arc;

use derive_builder::Builder;

use crate::blocks::BlockMetadata;
use crate::manager::BlockManager;

use super::request::RequestSequence;
use dynamo_tokens::Token;

// =============================================================================
// State types
// =============================================================================

/// Current scheduling state of a [`SchedulableSequence`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Idle,
    PrefillScheduled {
        num_tokens: usize,
        blocks_allocated: usize,
    },
    DecodeScheduled {
        blocks_allocated: usize,
    },
    SpeculativeScheduled {
        num_tokens: usize,
        blocks_allocated: usize,
    },
}

/// Outcome of an [`apply_decode`](SchedulableSequence::apply_decode) or
/// [`apply_speculative`](SchedulableSequence::apply_speculative) call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeOutcome {
    Continue,
    BlockCompleted,
    MaxLength,
    BlockCompletedAndMaxLength,
}

/// Append-only event recording a lifecycle transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceEvent {
    Created {
        num_input_tokens: usize,
        max_output_tokens: usize,
        block_size: usize,
    },
    PrefixMatched {
        blocks_matched: usize,
    },
    PrefillScheduled {
        num_tokens: usize,
        blocks_allocated: usize,
    },
    PrefillApplied {
        num_tokens: usize,
        blocks_registered: usize,
        token_emitted: bool,
    },
    DecodeScheduled {
        blocks_allocated: usize,
    },
    DecodeApplied {
        token: Token,
        block_completed: bool,
    },
    SpeculativeScheduled {
        num_tokens: usize,
        blocks_allocated: usize,
    },
    SpeculativeApplied {
        accepted: usize,
        scheduled: usize,
        blocks_released: usize,
    },
    ScheduleReverted {
        blocks_released: usize,
    },
    UnassignedDropped {
        count: usize,
    },
    Released,
    Reacquired {
        prefix_matched: usize,
        success: bool,
    },
}

// =============================================================================
// Delegate
// =============================================================================

/// Callback interface for [`SchedulableSequence`] lifecycle events.
///
/// Implementations receive every [`SequenceEvent`] as it occurs, enabling
/// real-time metrics, logging, or external state updates without coupling
/// event storage to the sequence itself.
pub trait SequenceDelegate: Send + Sync {
    fn on_event(&self, event: &SequenceEvent);
}

/// No-op delegate that silently discards all events.
///
/// Used as the default when no delegate is provided to
/// [`SchedulableSequenceBuilder`].
pub struct NoopDelegate;

impl SequenceDelegate for NoopDelegate {
    fn on_event(&self, _event: &SequenceEvent) {}
}

// =============================================================================
// Builder
// =============================================================================

#[doc(hidden)]
#[derive(Builder)]
#[builder(
    name = "SchedulableSequenceBuilder",
    pattern = "owned",
    build_fn(private, name = "build_params", error = "anyhow::Error")
)]
pub struct SchedulableSequenceParams {
    tokens: Vec<Token>,
    max_output_tokens: usize,
    block_size: u32,
    #[builder(default, setter(custom))]
    delegate: Option<Arc<dyn SequenceDelegate>>,
}

impl SchedulableSequenceBuilder {
    pub fn delegate(mut self, delegate: Arc<dyn SequenceDelegate>) -> Self {
        self.delegate = Some(Some(delegate));
        self
    }

    pub fn build<T: BlockMetadata>(self) -> anyhow::Result<SchedulableSequence<T>> {
        let params = self.build_params()?;
        Ok(SchedulableSequence::new(
            params.tokens,
            params.max_output_tokens,
            params.block_size,
            params.delegate,
        ))
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Error returned by `schedule_*` methods.
#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error("expected Idle state, got {state:?}")]
    NotIdle { state: SequenceState },
    #[error("prefill overrun: position {position} + {num_tokens} > {num_input_tokens}")]
    PrefillOverrun {
        position: usize,
        num_tokens: usize,
        num_input_tokens: usize,
    },
    #[error("prefill already complete")]
    PrefillComplete,
    #[error("prefill not yet complete (position {position} < {num_input_tokens})")]
    PrefillNotComplete {
        position: usize,
        num_input_tokens: usize,
    },
    #[error("allocation failed: needed {needed} blocks")]
    AllocationFailed { needed: usize },
    #[error("generation complete: {generated} >= {max_output}")]
    GenerationComplete { generated: usize, max_output: usize },
    #[error("expected {expected} dangling token(s), got {actual}")]
    WrongDanglingCount { expected: usize, actual: usize },
}

/// Error returned by `apply_*` and `revert_schedule` methods.
#[derive(Debug, thiserror::Error)]
pub enum ApplyError {
    #[error("expected {expected}, got {actual:?}")]
    WrongState {
        expected: &'static str,
        actual: SequenceState,
    },
    #[error("token provided but prefill not completing this chunk")]
    TokenOnNonFinalChunk,
    #[error("accepted {accepted} tokens exceeds scheduled {scheduled}")]
    AcceptedExceedsScheduled { accepted: usize, scheduled: usize },
    #[error("final prefill chunk requires a generated token")]
    MissingTokenOnFinalChunk,
}

// =============================================================================
// SchedulableSequence
// =============================================================================

/// Two-phase schedule/apply wrapper over [`RequestSequence`].
///
/// Enforces a state machine protocol:
/// - `Idle` → `schedule_*` → `Scheduled` → `apply_*` or `revert_schedule` → `Idle`
///
/// Dispatches lifecycle events to a caller-provided [`SequenceDelegate`].
pub struct SchedulableSequence<T: BlockMetadata> {
    inner: RequestSequence<T>,
    state: SequenceState,
    prefill_position: usize,
    kv_position: usize,
    delegate: Arc<dyn SequenceDelegate>,
}

impl<T: BlockMetadata> SchedulableSequence<T> {
    // =====================================================================
    // Construction
    // =====================================================================

    /// Returns a builder for configuring a `SchedulableSequence`.
    pub fn builder() -> SchedulableSequenceBuilder {
        SchedulableSequenceBuilder::default()
    }

    /// Creates a new `SchedulableSequence` wrapping a fresh `RequestSequence`.
    ///
    /// If `delegate` is `None`, a [`NoopDelegate`] is used.
    pub fn new(
        tokens: Vec<Token>,
        max_output_tokens: usize,
        block_size: u32,
        delegate: Option<Arc<dyn SequenceDelegate>>,
    ) -> Self {
        let inner = RequestSequence::new(tokens, max_output_tokens, block_size);
        let delegate = delegate.unwrap_or_else(|| Arc::new(NoopDelegate));
        delegate.on_event(&SequenceEvent::Created {
            num_input_tokens: inner.num_input_tokens(),
            max_output_tokens,
            block_size: block_size as usize,
        });
        Self {
            inner,
            state: SequenceState::Idle,
            prefill_position: 0,
            kv_position: 0,
            delegate,
        }
    }

    // =====================================================================
    // Prefix matching (Idle only)
    // =====================================================================

    /// Match and add prefix blocks from the manager's pools.
    ///
    /// Advances `prefill_position` by `matched_blocks * block_size`.
    pub fn match_and_add_prefix(
        &mut self,
        manager: &BlockManager<T>,
    ) -> Result<usize, ScheduleError> {
        self.require_idle()?;

        let matched = self.inner.match_prefix(manager);
        let count = matched.len();

        if count > 0 {
            self.inner
                .add_matched_blocks(matched)
                .unwrap_or_else(|_| panic!("prefix match should not produce duplicates"));
            self.prefill_position += count * self.inner.block_size();
            self.kv_position = self.prefill_position;
        }

        self.delegate.on_event(&SequenceEvent::PrefixMatched {
            blocks_matched: count,
        });

        Ok(count)
    }

    // =====================================================================
    // Two-phase prefill
    // =====================================================================

    /// Schedule a prefill chunk of `num_tokens` tokens.
    ///
    /// Allocates the blocks needed to cover `prefill_position + num_tokens`
    /// tokens, minus already-assigned/staged/unassigned blocks. If this is
    /// the final prefill chunk and `max_output_tokens > 0`, an extra
    /// generation block is allocated.
    pub fn schedule_prefill(
        &mut self,
        num_tokens: usize,
        manager: &BlockManager<T>,
    ) -> Result<(), ScheduleError> {
        self.require_idle()?;

        if self.is_prefill_complete() {
            return Err(ScheduleError::PrefillComplete);
        }

        let num_input = self.inner.num_input_tokens();
        let new_position = self.prefill_position + num_tokens;
        if new_position > num_input {
            return Err(ScheduleError::PrefillOverrun {
                position: self.prefill_position,
                num_tokens,
                num_input_tokens: num_input,
            });
        }

        // How many total blocks are needed to cover tokens up to new_position?
        let bs = self.inner.block_size();
        let total_blocks_needed = new_position / bs;

        // How many do we already have?
        let already_have = self.inner.assigned_blocks()
            + self.inner.staged_blocks()
            + self.inner.unassigned_blocks();

        let to_allocate = total_blocks_needed.saturating_sub(already_have);

        if to_allocate > 0 && !self.inner.allocate_blocks(to_allocate, manager) {
            return Err(ScheduleError::AllocationFailed {
                needed: to_allocate,
            });
        }

        self.state = SequenceState::PrefillScheduled {
            num_tokens,
            blocks_allocated: to_allocate,
        };

        self.delegate.on_event(&SequenceEvent::PrefillScheduled {
            num_tokens,
            blocks_allocated: to_allocate,
        });

        Ok(())
    }

    /// Apply a previously scheduled prefill chunk.
    ///
    /// If `token` is `Some`, it is the first generation token emitted on the
    /// final prefill chunk. Providing a token on a non-final chunk returns an
    /// error.
    pub fn apply_prefill(
        &mut self,
        token: Option<Token>,
        manager: &BlockManager<T>,
    ) -> Result<(), ApplyError> {
        let (num_tokens, _blocks_allocated) = match self.state {
            SequenceState::PrefillScheduled {
                num_tokens,
                blocks_allocated,
            } => (num_tokens, blocks_allocated),
            other => {
                return Err(ApplyError::WrongState {
                    expected: "PrefillScheduled",
                    actual: other,
                });
            }
        };

        let new_position = self.prefill_position + num_tokens;
        let is_final = new_position >= self.inner.num_input_tokens();

        if token.is_some() && !is_final {
            return Err(ApplyError::TokenOnNonFinalChunk);
        }
        if is_final && token.is_none() && self.inner.max_output_tokens() > 0 {
            return Err(ApplyError::MissingTokenOnFinalChunk);
        }

        // Stage and register the prefill blocks
        self.inner.complete_and_register_pending(manager);

        let blocks_registered_before = self.inner.assigned_blocks();
        self.prefill_position = new_position;
        self.kv_position = self.prefill_position;

        // If a token was provided on the final chunk, append it.
        // The token is "dangling" — its KV hasn't been computed yet.
        // The block it may complete is NOT registered here; it will be
        // staged during the next apply_decode after the model forward pass.
        let token_emitted = token.is_some();
        if let Some(tok) = token {
            self.inner.append_token(tok);
        }

        let blocks_registered =
            self.inner.assigned_blocks() - blocks_registered_before + self.inner.staged_blocks();

        self.state = SequenceState::Idle;
        self.delegate.on_event(&SequenceEvent::PrefillApplied {
            num_tokens,
            blocks_registered,
            token_emitted,
        });

        Ok(())
    }

    // =====================================================================
    // Two-phase decode
    // =====================================================================

    /// Schedule a single decode step.
    ///
    /// Allocates blocks for both pending completions (blocks completed at
    /// the sequence level but not yet staged, e.g. from prefill's dangling
    /// token crossing a boundary) and the generation block.
    pub fn schedule_decode(&mut self, manager: &BlockManager<T>) -> Result<(), ScheduleError> {
        self.require_idle()?;
        self.require_prefill_complete()?;
        self.require_not_complete()?;
        self.require_one_dangling()?;

        // Pending completions: complete blocks in sequence not yet assigned/staged
        let complete_in_seq = self.inner.complete_sequence_blocks();
        let registered = self.inner.assigned_blocks() + self.inner.staged_blocks();
        let pending = complete_in_seq.saturating_sub(registered);

        // Need: pending (for staging after KV computed) + 1 (gen block)
        let need = pending + 1;
        let have = self.inner.unassigned_blocks();
        let to_allocate = need.saturating_sub(have);

        if to_allocate > 0 && !self.inner.allocate_blocks(to_allocate, manager) {
            return Err(ScheduleError::AllocationFailed {
                needed: to_allocate,
            });
        }

        self.state = SequenceState::DecodeScheduled {
            blocks_allocated: to_allocate,
        };
        self.delegate.on_event(&SequenceEvent::DecodeScheduled {
            blocks_allocated: to_allocate,
        });

        Ok(())
    }

    /// Apply a previously scheduled decode step.
    pub fn apply_decode(
        &mut self,
        token: Token,
        manager: &BlockManager<T>,
    ) -> Result<DecodeOutcome, ApplyError> {
        let _blocks_allocated = match self.state {
            SequenceState::DecodeScheduled { blocks_allocated } => blocks_allocated,
            other => {
                return Err(ApplyError::WrongState {
                    expected: "DecodeScheduled",
                    actual: other,
                });
            }
        };

        let crossed = self.inner.append_token(token);
        let block_completed = crossed.is_some();

        // Always stage pending completions — handles both:
        // 1. Blocks completed during prefill's token append (deferred staging)
        // 2. Block just completed by this decode token
        self.inner.complete_and_register_pending(manager);

        self.kv_position += 1;
        self.state = SequenceState::Idle;
        self.delegate.on_event(&SequenceEvent::DecodeApplied {
            token,
            block_completed,
        });

        let is_complete = self.inner.is_complete();
        Ok(match (block_completed, is_complete) {
            (false, false) => DecodeOutcome::Continue,
            (true, false) => DecodeOutcome::BlockCompleted,
            (false, true) => DecodeOutcome::MaxLength,
            (true, true) => DecodeOutcome::BlockCompletedAndMaxLength,
        })
    }

    // =====================================================================
    // Speculative decode
    // =====================================================================

    /// Schedule a speculative decode of `num_draft_tokens` tokens.
    ///
    /// Allocates enough blocks to accommodate the draft tokens, accounting
    /// for the current partial block state and all already-held blocks
    /// (assigned + staged + unassigned).
    pub fn schedule_speculative(
        &mut self,
        num_draft_tokens: usize,
        manager: &BlockManager<T>,
    ) -> Result<(), ScheduleError> {
        self.require_idle()?;
        self.require_prefill_complete()?;
        self.require_not_complete()?;
        self.require_one_dangling()?;

        let bs = self.inner.block_size();
        let future_total = self.inner.total_tokens() + num_draft_tokens;
        let future_blocks = future_total.div_ceil(bs);
        let have = self.inner.assigned_blocks()
            + self.inner.staged_blocks()
            + self.inner.unassigned_blocks();
        let to_allocate = future_blocks.saturating_sub(have);

        if to_allocate > 0 && !self.inner.allocate_blocks(to_allocate, manager) {
            return Err(ScheduleError::AllocationFailed {
                needed: to_allocate,
            });
        }

        self.state = SequenceState::SpeculativeScheduled {
            num_tokens: num_draft_tokens,
            blocks_allocated: to_allocate,
        };
        self.delegate.on_event(&SequenceEvent::SpeculativeScheduled {
            num_tokens: num_draft_tokens,
            blocks_allocated: to_allocate,
        });

        Ok(())
    }

    /// Apply a speculative decode with `accepted` tokens (a prefix of the
    /// scheduled draft).
    ///
    /// Excess unassigned blocks (allocated for rejected draft tokens) are
    /// LIFO-dropped, returning them to the pool via RAII.
    pub fn apply_speculative(
        &mut self,
        accepted: &[Token],
        manager: &BlockManager<T>,
    ) -> Result<DecodeOutcome, ApplyError> {
        let (scheduled_tokens, _blocks_allocated) = match self.state {
            SequenceState::SpeculativeScheduled {
                num_tokens,
                blocks_allocated,
            } => (num_tokens, blocks_allocated),
            other => {
                return Err(ApplyError::WrongState {
                    expected: "SpeculativeScheduled",
                    actual: other,
                });
            }
        };

        if accepted.len() > scheduled_tokens {
            return Err(ApplyError::AcceptedExceedsScheduled {
                accepted: accepted.len(),
                scheduled: scheduled_tokens,
            });
        }

        // Append accepted tokens one at a time, tracking boundary crossings
        let mut block_completed = false;
        for &token in accepted {
            let crossed = self.inner.append_token(token);
            if crossed.is_some() {
                block_completed = true;
            }
        }

        // Stage and register all pending completions (including any from
        // prefill's deferred staging and blocks just completed above)
        self.inner.complete_and_register_pending(manager);

        self.kv_position += accepted.len();

        // LIFO-drop excess unassigned blocks
        let blocks_actually_needed = self.blocks_needed_for_tokens(0); // 0 more beyond what was appended
        // We just need to see how many unassigned remain vs needed.
        // After appending accepted tokens, the generation block (if any) is the
        // remaining unassigned. If we over-allocated for the draft, drop excess.
        let excess = self.lifo_drop_excess_unassigned();

        self.state = SequenceState::Idle;
        self.delegate.on_event(&SequenceEvent::SpeculativeApplied {
            accepted: accepted.len(),
            scheduled: scheduled_tokens,
            blocks_released: excess,
        });

        let _ = blocks_actually_needed; // used only for the LIFO drop calculation above

        let is_complete = self.inner.is_complete();
        Ok(match (block_completed, is_complete) {
            (false, false) => DecodeOutcome::Continue,
            (true, false) => DecodeOutcome::BlockCompleted,
            (false, true) => DecodeOutcome::MaxLength,
            (true, true) => DecodeOutcome::BlockCompletedAndMaxLength,
        })
    }

    // =====================================================================
    // Revert
    // =====================================================================

    /// Revert a scheduled (but not yet applied) operation.
    ///
    /// LIFO-pops the `blocks_allocated` unassigned blocks that were
    /// pre-allocated during the schedule phase. The dropped RAII guards
    /// return blocks to the manager's pools.
    pub fn revert_schedule(&mut self) -> Result<(), ApplyError> {
        let blocks_to_release = match self.state {
            SequenceState::PrefillScheduled {
                blocks_allocated, ..
            } => blocks_allocated,
            SequenceState::DecodeScheduled { blocks_allocated } => blocks_allocated,
            SequenceState::SpeculativeScheduled {
                blocks_allocated, ..
            } => blocks_allocated,
            other => {
                return Err(ApplyError::WrongState {
                    expected: "any Scheduled state",
                    actual: other,
                });
            }
        };

        self.lifo_pop_unassigned(blocks_to_release);

        self.state = SequenceState::Idle;
        self.delegate.on_event(&SequenceEvent::ScheduleReverted {
            blocks_released: blocks_to_release,
        });

        Ok(())
    }

    // =====================================================================
    // Explicit LIFO drop of unassigned blocks
    // =====================================================================

    /// LIFO-drop up to `count` unassigned blocks. Returns the actual number
    /// dropped. Valid only in Idle state.
    pub fn drop_unassigned(&mut self, count: usize) -> usize {
        let dropped = self.lifo_pop_unassigned(count);
        if dropped > 0 {
            self.delegate
                .on_event(&SequenceEvent::UnassignedDropped { count: dropped });
        }
        dropped
    }

    // =====================================================================
    // Lifecycle
    // =====================================================================

    /// Release all block assignments (RAII returns them to pools).
    pub fn release(&mut self) -> Result<(), ApplyError> {
        self.require_idle_for_apply()?;
        self.inner.release();
        self.delegate.on_event(&SequenceEvent::Released);
        Ok(())
    }

    /// Re-acquire blocks from the manager after a release/preemption.
    pub fn reacquire(&mut self, manager: &BlockManager<T>) -> Result<bool, ApplyError> {
        self.require_idle_for_apply()?;
        let success = self.inner.reacquire(manager);
        let prefix_matched = self.inner.prefix_matched_blocks();
        self.delegate.on_event(&SequenceEvent::Reacquired {
            prefix_matched,
            success,
        });
        Ok(success)
    }

    // =====================================================================
    // Token append (no KV advancement)
    // =====================================================================

    /// Append tokens to the sequence without advancing `kv_position`.
    /// Each appended token increases the dangling count.
    /// Requires Idle state.
    pub fn append_tokens(&mut self, tokens: &[Token]) -> Result<(), ApplyError> {
        self.require_idle_for_apply()?;
        for &token in tokens {
            self.inner.append_token(token);
        }
        Ok(())
    }

    // =====================================================================
    // Accessors
    // =====================================================================

    /// Current scheduling state.
    pub fn state(&self) -> SequenceState {
        self.state
    }

    /// How many input tokens have been processed so far.
    pub fn prefill_position(&self) -> usize {
        self.prefill_position
    }

    /// Whether all input tokens have been processed.
    pub fn is_prefill_complete(&self) -> bool {
        self.prefill_position >= self.inner.num_input_tokens()
    }

    /// Number of tokens whose KV has been computed.
    pub fn kv_position(&self) -> usize {
        self.kv_position
    }

    /// Number of tokens whose KV hasn't been computed yet.
    /// After prefill: 1 (the first generated token). After decode: 1 (the new token).
    pub fn tail_tokens(&self) -> usize {
        self.inner.total_tokens().saturating_sub(self.kv_position)
    }

    /// Reference to the delegate.
    pub fn delegate(&self) -> &Arc<dyn SequenceDelegate> {
        &self.delegate
    }

    // Forwarded from RequestSequence

    pub fn generated_tokens(&self) -> usize {
        self.inner.generated_tokens()
    }

    pub fn max_output_tokens(&self) -> usize {
        self.inner.max_output_tokens()
    }

    pub fn num_input_tokens(&self) -> usize {
        self.inner.num_input_tokens()
    }

    pub fn total_tokens(&self) -> usize {
        self.inner.total_tokens()
    }

    pub fn remaining_tokens(&self) -> usize {
        self.inner.remaining_tokens()
    }

    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks()
    }

    pub fn assigned_blocks(&self) -> usize {
        self.inner.assigned_blocks()
    }

    pub fn staged_blocks(&self) -> usize {
        self.inner.staged_blocks()
    }

    pub fn unassigned_blocks(&self) -> usize {
        self.inner.unassigned_blocks()
    }

    pub fn prefix_matched_blocks(&self) -> usize {
        self.inner.prefix_matched_blocks()
    }

    pub fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Reference to the underlying `RequestSequence`.
    pub fn inner(&self) -> &RequestSequence<T> {
        &self.inner
    }

    /// Mutable reference to the underlying `RequestSequence`.
    #[allow(dead_code)]
    pub(crate) fn inner_mut(&mut self) -> &mut RequestSequence<T> {
        &mut self.inner
    }

    // =====================================================================
    // Private helpers
    // =====================================================================

    fn require_idle(&self) -> Result<(), ScheduleError> {
        if self.state != SequenceState::Idle {
            return Err(ScheduleError::NotIdle { state: self.state });
        }
        Ok(())
    }

    fn require_idle_for_apply(&self) -> Result<(), ApplyError> {
        if self.state != SequenceState::Idle {
            return Err(ApplyError::WrongState {
                expected: "Idle",
                actual: self.state,
            });
        }
        Ok(())
    }

    fn require_prefill_complete(&self) -> Result<(), ScheduleError> {
        if !self.is_prefill_complete() {
            return Err(ScheduleError::PrefillNotComplete {
                position: self.prefill_position,
                num_input_tokens: self.inner.num_input_tokens(),
            });
        }
        Ok(())
    }

    fn require_not_complete(&self) -> Result<(), ScheduleError> {
        if self.inner.is_complete() {
            return Err(ScheduleError::GenerationComplete {
                generated: self.inner.generated_tokens(),
                max_output: self.inner.max_output_tokens(),
            });
        }
        Ok(())
    }

    fn require_one_dangling(&self) -> Result<(), ScheduleError> {
        let dangling = self.tail_tokens();
        if dangling != 1 {
            return Err(ScheduleError::WrongDanglingCount {
                expected: 1,
                actual: dangling,
            });
        }
        Ok(())
    }

    /// Given `additional_tokens` more tokens to append, how many **new** blocks
    /// are needed (not counting already-unassigned blocks)?
    fn blocks_needed_for_tokens(&self, additional_tokens: usize) -> usize {
        if additional_tokens == 0 {
            return 0;
        }
        let bs = self.inner.block_size();
        if bs == 0 {
            return 0;
        }
        let total_tokens = self.inner.total_tokens();
        let current_blocks_used = total_tokens.div_ceil(bs);
        let future_total = total_tokens + additional_tokens;
        let future_blocks_needed = future_total.div_ceil(bs);
        future_blocks_needed.saturating_sub(current_blocks_used)
    }

    /// LIFO-pop up to `count` unassigned blocks. Returns the actual count dropped.
    fn lifo_pop_unassigned(&mut self, count: usize) -> usize {
        let assignments = self.inner.assignments_mut();
        let mut dropped = 0;
        for _ in 0..count {
            if assignments.pop_last_unassigned().is_some() {
                dropped += 1;
            } else {
                break;
            }
        }
        dropped
    }

    /// After speculative apply: drop any excess unassigned blocks beyond
    /// what's needed for the current partial block (at most 1 gen block).
    fn lifo_drop_excess_unassigned(&mut self) -> usize {
        let bs = self.inner.block_size();
        let total = self.inner.total_tokens();
        // We need at most 1 unassigned (gen) block if there's a partial block in progress
        // AND we haven't hit max output tokens.
        let need_gen = if self.inner.is_complete() {
            0
        } else if !total.is_multiple_of(bs) {
            // Partial block in progress — already have an unassigned block covering it
            1
        } else {
            // On exact boundary — the last block was just completed & registered.
            // We still keep 1 gen block for future decode unless complete.
            1
        };

        let current = self.inner.unassigned_blocks();
        let excess = current.saturating_sub(need_gen);
        self.lifo_pop_unassigned(excess)
    }
}

impl<T: BlockMetadata> std::fmt::Debug for SchedulableSequence<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulableSequence")
            .field("state", &self.state)
            .field("prefill_position", &self.prefill_position)
            .field("kv_position", &self.kv_position)
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestMeta, create_test_manager};
    use std::sync::Mutex;

    const BLOCK_SIZE: u32 = 4;

    // =========================================================================
    // Test delegate
    // =========================================================================

    struct CollectingDelegate {
        events: Mutex<Vec<SequenceEvent>>,
    }

    impl CollectingDelegate {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn events(&self) -> Vec<SequenceEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl SequenceDelegate for CollectingDelegate {
        fn on_event(&self, event: &SequenceEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    fn noop_delegate() -> Option<Arc<dyn SequenceDelegate>> {
        None
    }

    fn make_tokens(n: usize) -> Vec<Token> {
        (0..n as u32).collect()
    }

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn test_new_starts_idle() {
        let delegate = Arc::new(CollectingDelegate::new());
        let seq = SchedulableSequence::<TestMeta>::new(
            make_tokens(8),
            10,
            BLOCK_SIZE,
            Some(delegate.clone()),
        );

        assert_eq!(seq.state(), SequenceState::Idle);
        assert_eq!(seq.prefill_position(), 0);
        assert_eq!(seq.kv_position(), 0);
        assert_eq!(seq.tail_tokens(), 8);
        assert_eq!(seq.num_input_tokens(), 8);
        assert_eq!(seq.max_output_tokens(), 10);
        assert_eq!(seq.block_size(), BLOCK_SIZE as usize);
        assert!(!seq.is_prefill_complete());

        let events = delegate.events();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            SequenceEvent::Created {
                num_input_tokens: 8,
                max_output_tokens: 10,
                block_size: BLOCK_SIZE as usize,
            }
        );
    }

    // =========================================================================
    // State machine enforcement
    // =========================================================================

    #[test]
    fn test_schedule_prefill_requires_idle() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        let err = seq.schedule_prefill(4, &manager).unwrap_err();
        assert!(matches!(err, ScheduleError::NotIdle { .. }));
    }

    #[test]
    fn test_schedule_decode_requires_idle() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE, noop_delegate());

        // Prefill first
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();

        // Schedule decode
        seq.schedule_decode(&manager).unwrap();
        let err = seq.schedule_decode(&manager).unwrap_err();
        assert!(matches!(err, ScheduleError::NotIdle { .. }));
    }

    #[test]
    fn test_apply_prefill_requires_scheduled() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let err = seq.apply_prefill(None, &manager).unwrap_err();
        assert!(matches!(err, ApplyError::WrongState { .. }));
    }

    #[test]
    fn test_apply_decode_requires_scheduled() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE, noop_delegate());

        let err = seq.apply_decode(100, &manager).unwrap_err();
        assert!(matches!(err, ApplyError::WrongState { .. }));
    }

    #[test]
    fn test_decode_requires_prefill_complete() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let err = seq.schedule_decode(&manager).unwrap_err();
        assert!(matches!(err, ScheduleError::PrefillNotComplete { .. }));
    }

    #[test]
    fn test_speculative_requires_prefill_complete() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let err = seq.schedule_speculative(3, &manager).unwrap_err();
        assert!(matches!(err, ScheduleError::PrefillNotComplete { .. }));
    }

    // =========================================================================
    // Prefill
    // =========================================================================

    #[test]
    fn test_prefill_single_chunk() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        // Schedule and apply the full prefill (no gen block allocated)
        seq.schedule_prefill(8, &manager).unwrap();
        assert_eq!(
            seq.state(),
            SequenceState::PrefillScheduled {
                num_tokens: 8,
                blocks_allocated: 2, // 2 input blocks only
            }
        );

        seq.apply_prefill(Some(1000), &manager).unwrap();
        assert_eq!(seq.state(), SequenceState::Idle);
        assert_eq!(seq.prefill_position(), 8);
        assert_eq!(seq.kv_position(), 8);
        assert!(seq.is_prefill_complete());
        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 0); // no gen block
        assert_eq!(seq.tail_tokens(), 1); // token 1000
    }

    #[test]
    fn test_prefill_chunked() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        // Chunk 1: first 4 tokens (1 block, non-final)
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(None, &manager).unwrap();
        assert_eq!(seq.prefill_position(), 4);
        assert_eq!(seq.kv_position(), 4);
        assert!(!seq.is_prefill_complete());
        assert_eq!(seq.assigned_blocks(), 1);

        // Chunk 2: next 4 tokens (1 block, final — must provide token)
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();
        assert_eq!(seq.prefill_position(), 8);
        assert_eq!(seq.kv_position(), 8);
        assert!(seq.is_prefill_complete());
        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 0); // no gen block
        assert_eq!(seq.tail_tokens(), 1);
    }

    #[test]
    fn test_prefill_final_with_first_token() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(100), &manager).unwrap();

        assert!(seq.is_prefill_complete());
        assert_eq!(seq.generated_tokens(), 1);
        assert_eq!(seq.total_tokens(), 5);
        assert_eq!(seq.kv_position(), 4);
        assert_eq!(seq.tail_tokens(), 1);
    }

    #[test]
    fn test_prefill_token_on_non_final_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        let err = seq.apply_prefill(Some(100), &manager).unwrap_err();
        assert!(matches!(err, ApplyError::TokenOnNonFinalChunk));
    }

    #[test]
    fn test_prefill_overrun_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let err = seq.schedule_prefill(9, &manager).unwrap_err();
        assert!(matches!(err, ScheduleError::PrefillOverrun { .. }));
    }

    #[test]
    fn test_prefill_allocation_failure() {
        let manager = create_test_manager::<TestMeta>(1); // only 1 block
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let err = seq.schedule_prefill(8, &manager).unwrap_err();
        assert!(matches!(err, ScheduleError::AllocationFailed { .. }));
        // State remains idle
        assert_eq!(seq.state(), SequenceState::Idle);
    }

    #[test]
    fn test_schedule_prefill_after_complete_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();

        let err = seq.schedule_prefill(1, &manager).unwrap_err();
        assert!(matches!(err, ScheduleError::PrefillComplete));
    }

    #[test]
    fn test_apply_prefill_none_on_final_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        let err = seq.apply_prefill(None, &manager).unwrap_err();
        assert!(matches!(err, ApplyError::MissingTokenOnFinalChunk));
    }

    // =========================================================================
    // Decode
    // =========================================================================

    /// Helper: create a sequence with prefill done and first generated token.
    ///
    /// After this: `total_tokens = num_input + 1`, `generated_tokens = 1`,
    /// `kv_position = num_input`, `tail_tokens = 1`, `unassigned_blocks = 0`.
    fn prefilled_seq(
        num_input: usize,
        max_output: usize,
        manager: &BlockManager<TestMeta>,
    ) -> SchedulableSequence<TestMeta> {
        let mut seq =
            SchedulableSequence::new(make_tokens(num_input), max_output, BLOCK_SIZE, noop_delegate());
        if num_input > 0 {
            seq.schedule_prefill(num_input, manager).unwrap();
            seq.apply_prefill(Some(1000), manager).unwrap();
        }
        seq
    }

    #[test]
    fn test_decode_continue() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(5, 10, &manager);
        // After prefill: total=6, gen=1, kv=5, assigned=1, unassigned=0

        seq.schedule_decode(&manager).unwrap();
        let outcome = seq.apply_decode(100, &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::Continue);
        assert_eq!(seq.generated_tokens(), 2); // 1 from prefill + 1 decode
        assert_eq!(seq.state(), SequenceState::Idle);
    }

    #[test]
    fn test_decode_block_completed() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);
        // After prefill: total=5, kv=4, assigned=1, unassigned=0

        // 3 decodes to reach block boundary at total=8
        for _ in 0..2 {
            seq.schedule_decode(&manager).unwrap();
            let outcome = seq.apply_decode(100, &manager).unwrap();
            assert_eq!(outcome, DecodeOutcome::Continue);
        }
        seq.schedule_decode(&manager).unwrap();
        let outcome = seq.apply_decode(100, &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::BlockCompleted);
        assert_eq!(seq.assigned_blocks(), 2);
    }

    #[test]
    fn test_decode_max_length() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(5, 2, &manager);
        // After prefill: gen=1, max=2. One more decode reaches max.

        seq.schedule_decode(&manager).unwrap();
        let outcome = seq.apply_decode(100, &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::MaxLength);
        assert!(seq.is_complete());
    }

    #[test]
    fn test_decode_block_and_max() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 4, &manager);
        // After prefill: total=5, gen=1, max=4. Need 3 more decodes.
        // At decode 3: total=8 (boundary) AND gen=4=max.

        for _ in 0..2 {
            seq.schedule_decode(&manager).unwrap();
            seq.apply_decode(100, &manager).unwrap();
        }
        seq.schedule_decode(&manager).unwrap();
        let outcome = seq.apply_decode(100, &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::BlockCompletedAndMaxLength);
    }

    #[test]
    fn test_decode_allocates_gen_block() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);
        // After prefill: 1 assigned, 0 unassigned

        // First schedule_decode always allocates 1 (gen block)
        seq.schedule_decode(&manager).unwrap();
        assert_eq!(
            seq.state(),
            SequenceState::DecodeScheduled {
                blocks_allocated: 1
            }
        );
        assert_eq!(seq.unassigned_blocks(), 1);

        // Decode until block boundary: 3 decodes total reach total=8
        seq.apply_decode(100, &manager).unwrap(); // total=6
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(101, &manager).unwrap(); // total=7
        seq.schedule_decode(&manager).unwrap();
        let outcome = seq.apply_decode(102, &manager).unwrap(); // total=8, crosses boundary
        assert_eq!(outcome, DecodeOutcome::BlockCompleted);
        assert_eq!(seq.unassigned_blocks(), 0);

        // Next schedule_decode should allocate 1 gen block
        seq.schedule_decode(&manager).unwrap();
        assert_eq!(
            seq.state(),
            SequenceState::DecodeScheduled {
                blocks_allocated: 1
            }
        );
        assert_eq!(seq.unassigned_blocks(), 1);
    }

    #[test]
    fn test_decode_generation_complete_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(5, 2, &manager);
        // After prefill: gen=1, max=2

        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();
        assert!(seq.is_complete()); // gen=2=max

        let err = seq.schedule_decode(&manager).unwrap_err();
        assert!(matches!(err, ScheduleError::GenerationComplete { .. }));
    }

    // =========================================================================
    // Speculative decode
    // =========================================================================

    #[test]
    fn test_speculative_basic() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(8, 10, &manager);
        // After prefill: total=9, gen=1, kv=8, assigned=2, unassigned=0

        // Schedule 2 draft tokens (stay within block)
        seq.schedule_speculative(2, &manager).unwrap();
        assert!(matches!(
            seq.state(),
            SequenceState::SpeculativeScheduled { num_tokens: 2, .. }
        ));

        // Accept both
        let outcome = seq.apply_speculative(&[100, 101], &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::Continue);
        assert_eq!(seq.generated_tokens(), 3); // 1 from prefill + 2 speculative
        assert_eq!(seq.state(), SequenceState::Idle);
    }

    #[test]
    fn test_speculative_partial_accept() {
        let manager = create_test_manager::<TestMeta>(20);
        let delegate = Arc::new(CollectingDelegate::new());
        let mut seq =
            SchedulableSequence::new(make_tokens(4), 10, BLOCK_SIZE, Some(delegate.clone()));
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();
        // After prefill: total=5, assigned=1, unassigned=0

        let avail_before = manager.available_blocks();

        // Schedule 4 draft tokens
        seq.schedule_speculative(4, &manager).unwrap();

        // Accept only 2 → excess blocks should be released
        let outcome = seq.apply_speculative(&[100, 101], &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::Continue);
        assert_eq!(seq.generated_tokens(), 3); // 1 from prefill + 2
        assert_eq!(seq.unassigned_blocks(), 1); // keep 1 gen block

        // Check delegate records the release
        let events = delegate.events();
        let last = events.last().unwrap();
        if let SequenceEvent::SpeculativeApplied {
            accepted,
            scheduled,
            ..
        } = last
        {
            assert_eq!(*accepted, 2);
            assert_eq!(*scheduled, 4);
        } else {
            panic!("expected SpeculativeApplied");
        }

        let _ = avail_before;
    }

    #[test]
    fn test_speculative_single_accept() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);

        seq.schedule_speculative(5, &manager).unwrap();
        let outcome = seq.apply_speculative(&[100], &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::Continue);
        assert_eq!(seq.generated_tokens(), 2); // 1 from prefill + 1
    }

    #[test]
    fn test_speculative_zero_accept() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);

        let avail_before = manager.available_blocks();
        seq.schedule_speculative(3, &manager).unwrap();
        let avail_after_schedule = manager.available_blocks();

        let outcome = seq.apply_speculative(&[], &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::Continue);
        assert_eq!(seq.generated_tokens(), 1); // 1 from prefill, 0 speculative

        assert_eq!(seq.unassigned_blocks(), 1); // keep 1 gen block
        assert!(manager.available_blocks() >= avail_after_schedule);
        let _ = avail_before;
    }

    #[test]
    fn test_speculative_exceeds_scheduled_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);

        seq.schedule_speculative(2, &manager).unwrap();
        let err = seq
            .apply_speculative(&[100, 101, 102], &manager)
            .unwrap_err();
        assert!(matches!(
            err,
            ApplyError::AcceptedExceedsScheduled {
                accepted: 3,
                scheduled: 2,
            }
        ));
    }

    #[test]
    fn test_speculative_block_boundaries() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(7, 20, &manager);
        // After prefill: total=8, kv=7, assigned=1, unassigned=0
        // Block 1 complete at sequence level (4-6 + token 1000) but NOT registered

        // 5 speculative tokens: registers pending block 1, crosses another boundary
        seq.schedule_speculative(5, &manager).unwrap();
        let outcome = seq
            .apply_speculative(&[100, 101, 102, 103, 104], &manager)
            .unwrap();
        assert_eq!(outcome, DecodeOutcome::BlockCompleted);
        assert_eq!(seq.generated_tokens(), 6); // 1 from prefill + 5
        assert_eq!(seq.assigned_blocks(), 3); // block 0 + blocks 1,2 registered
    }

    // =========================================================================
    // Revert
    // =========================================================================

    #[test]
    fn test_revert_prefill() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let avail_before = manager.available_blocks();
        seq.schedule_prefill(4, &manager).unwrap();
        assert!(manager.available_blocks() < avail_before);

        seq.revert_schedule().unwrap();
        assert_eq!(seq.state(), SequenceState::Idle);
        assert_eq!(manager.available_blocks(), avail_before);
    }

    #[test]
    fn test_revert_decode() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);
        // After prefill: total=5, assigned=1, unassigned=0

        // Decode until boundary at total=8 → unassigned drops to 0
        for _ in 0..3 {
            seq.schedule_decode(&manager).unwrap();
            seq.apply_decode(100, &manager).unwrap();
        }
        assert_eq!(seq.unassigned_blocks(), 0);

        let avail_before = manager.available_blocks();
        seq.schedule_decode(&manager).unwrap();
        assert_eq!(manager.available_blocks(), avail_before - 1);

        seq.revert_schedule().unwrap();
        assert_eq!(seq.state(), SequenceState::Idle);
        assert_eq!(manager.available_blocks(), avail_before);
    }

    #[test]
    fn test_revert_speculative() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);

        let avail_before = manager.available_blocks();
        seq.schedule_speculative(4, &manager).unwrap();
        let allocated = avail_before - manager.available_blocks();

        seq.revert_schedule().unwrap();
        assert_eq!(seq.state(), SequenceState::Idle);
        // Blocks allocated during schedule should be returned
        assert_eq!(manager.available_blocks(), avail_before);
        assert!(allocated > 0 || seq.unassigned_blocks() > 0);
    }

    #[test]
    fn test_revert_returns_blocks_to_manager() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let avail_before = manager.available_blocks();
        seq.schedule_prefill(8, &manager).unwrap();
        let avail_scheduled = manager.available_blocks();
        assert!(avail_scheduled < avail_before);

        seq.revert_schedule().unwrap();
        assert_eq!(manager.available_blocks(), avail_before);
    }

    // =========================================================================
    // Drop unassigned
    // =========================================================================

    #[test]
    fn test_drop_unassigned_lifo() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);
        // After prefill: 1 assigned, 0 unassigned

        // Do one decode to get 1 unassigned (gen block)
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();
        assert_eq!(seq.unassigned_blocks(), 1);

        let dropped = seq.drop_unassigned(1);
        assert_eq!(dropped, 1);
        assert_eq!(seq.unassigned_blocks(), 0);
    }

    #[test]
    fn test_drop_unassigned_partial() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(8, 10, &manager);
        // After prefill: 2 assigned, 0 unassigned

        // Do one decode to get 1 unassigned
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();
        assert_eq!(seq.unassigned_blocks(), 1);

        // Try to drop 5, but only 1 available
        let dropped = seq.drop_unassigned(5);
        assert_eq!(dropped, 1);
        assert_eq!(seq.unassigned_blocks(), 0);
    }

    #[test]
    fn test_drop_unassigned_zero() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = prefilled_seq(4, 10, &manager);

        // Do one decode to get 1 unassigned
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();

        let dropped = seq.drop_unassigned(0);
        assert_eq!(dropped, 0);
        assert_eq!(seq.unassigned_blocks(), 1); // unchanged
    }

    // =========================================================================
    // Delegate event collection
    // =========================================================================

    #[test]
    fn test_delegate_full_lifecycle() {
        let manager = create_test_manager::<TestMeta>(20);
        let delegate = Arc::new(CollectingDelegate::new());
        // max_output=3: 1 from prefill + 2 decodes
        let mut seq = SchedulableSequence::<TestMeta>::new(
            make_tokens(4),
            3,
            BLOCK_SIZE,
            Some(delegate.clone()),
        );

        // Prefill
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();

        // Decode token 1
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();

        // Decode token 2
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(101, &manager).unwrap();

        // Release
        seq.release().unwrap();

        let h = delegate.events();
        assert_eq!(h.len(), 8);

        assert!(matches!(h[0], SequenceEvent::Created { .. }));
        assert!(matches!(h[1], SequenceEvent::PrefillScheduled { .. }));
        assert!(matches!(h[2], SequenceEvent::PrefillApplied { .. }));
        assert!(matches!(h[3], SequenceEvent::DecodeScheduled { .. }));
        assert!(matches!(h[4], SequenceEvent::DecodeApplied { .. }));
        assert!(matches!(h[5], SequenceEvent::DecodeScheduled { .. }));
        assert!(matches!(h[6], SequenceEvent::DecodeApplied { .. }));
        assert!(matches!(h[7], SequenceEvent::Released));
    }

    // =========================================================================
    // Integration: full lifecycle
    // =========================================================================

    #[test]
    fn test_full_lifecycle_prefill_decode_release() {
        let manager = create_test_manager::<TestMeta>(20);
        // max_output=7: 1 from prefill + 6 decodes
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(6), 7, BLOCK_SIZE, noop_delegate());

        // Prefill 6 tokens → 1 complete block + 2 partial
        seq.schedule_prefill(6, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();

        assert!(seq.is_prefill_complete());
        assert_eq!(seq.assigned_blocks(), 1);
        assert_eq!(seq.unassigned_blocks(), 0);

        // Decode 6 tokens
        for i in 0..6u32 {
            seq.schedule_decode(&manager).unwrap();
            let outcome = seq.apply_decode(100 + i, &manager).unwrap();
            if i < 5 {
                match outcome {
                    DecodeOutcome::Continue | DecodeOutcome::BlockCompleted => {}
                    other => panic!("unexpected outcome at token {i}: {other:?}"),
                }
            } else {
                // Last token (gen=7=max)
                assert!(
                    outcome == DecodeOutcome::MaxLength
                        || outcome == DecodeOutcome::BlockCompletedAndMaxLength,
                    "last token should hit max length, got: {outcome:?}"
                );
            }
        }

        assert!(seq.is_complete());
        assert_eq!(seq.generated_tokens(), 7);
        assert_eq!(seq.total_tokens(), 13);

        seq.release().unwrap();
        assert_eq!(seq.assigned_blocks(), 0);
    }

    #[test]
    fn test_preempt_and_reacquire() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        // Prefill
        seq.schedule_prefill(8, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();

        // Decode 2 tokens
        for _ in 0..2 {
            seq.schedule_decode(&manager).unwrap();
            seq.apply_decode(100, &manager).unwrap();
        }
        assert_eq!(seq.generated_tokens(), 3); // 1 from prefill + 2

        // Preempt
        seq.release().unwrap();
        assert_eq!(seq.assigned_blocks(), 0);

        // Reacquire (no gen block — allocated later by schedule_decode)
        let success = seq.reacquire(&manager).unwrap();
        assert!(success);
        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 0); // no gen block from reacquire
        assert_eq!(seq.generated_tokens(), 3);

        // Continue decoding
        seq.schedule_decode(&manager).unwrap();
        let outcome = seq.apply_decode(200, &manager).unwrap();
        assert_eq!(seq.generated_tokens(), 4);
        let _ = outcome;
    }

    // =========================================================================
    // Prefix matching
    // =========================================================================

    #[test]
    fn test_match_and_add_prefix() {
        let manager = create_test_manager::<TestMeta>(20);
        let tokens = make_tokens(8);

        // Populate cache
        let seq_for_populate = crate::BlockSequence::new(tokens[..4].to_vec(), BLOCK_SIZE, None);
        let mutables = manager.allocate_blocks(1).unwrap();
        let registered: Vec<_> = mutables
            .into_iter()
            .zip(seq_for_populate.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();
        drop(registered);

        let mut seq =
            SchedulableSequence::<TestMeta>::new(tokens, 10, BLOCK_SIZE, noop_delegate());
        let matched = seq.match_and_add_prefix(&manager).unwrap();
        assert_eq!(matched, 1);
        assert_eq!(seq.prefill_position(), 4); // 1 block * 4 tokens
        assert_eq!(seq.kv_position(), 4); // cache hits have KV computed
        assert_eq!(seq.assigned_blocks(), 1);

        // Should only need 1 more input block (no gen block from prefill)
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();
        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 0); // no gen block
        assert_eq!(seq.tail_tokens(), 1);
    }

    #[test]
    fn test_match_and_add_prefix_no_hits() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        let matched = seq.match_and_add_prefix(&manager).unwrap();
        assert_eq!(matched, 0);
        assert_eq!(seq.prefill_position(), 0);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_empty_tokens_prefill() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(vec![], 10, BLOCK_SIZE, noop_delegate());

        assert!(seq.is_prefill_complete());

        // Cannot schedule prefill (already complete)
        let err = seq.schedule_prefill(0, &manager).unwrap_err();
        assert!(matches!(err, ScheduleError::PrefillComplete));

        // Cannot schedule decode with 0 dangling tokens
        let err = seq.schedule_decode(&manager).unwrap_err();
        assert!(matches!(err, ScheduleError::WrongDanglingCount { .. }));

        // Append initial token to create dangling, then decode
        seq.append_tokens(&[100]).unwrap();
        assert_eq!(seq.tail_tokens(), 1);

        seq.schedule_decode(&manager).unwrap();
        let outcome = seq.apply_decode(101, &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::Continue);
    }

    #[test]
    fn test_zero_max_output_no_gen_block() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(4), 0, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(None, &manager).unwrap();

        assert_eq!(seq.assigned_blocks(), 1);
        assert_eq!(seq.unassigned_blocks(), 0); // no gen block

        // Can't decode since is_complete
        let err = seq.schedule_decode(&manager).unwrap_err();
        assert!(matches!(err, ScheduleError::GenerationComplete { .. }));
    }

    #[test]
    fn test_debug_impl() {
        let seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());
        let debug_str = format!("{seq:?}");
        assert!(debug_str.contains("SchedulableSequence"));
        assert!(debug_str.contains("Idle"));
    }

    #[test]
    fn test_revert_idle_rejected() {
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());
        let err = seq.revert_schedule().unwrap_err();
        assert!(matches!(err, ApplyError::WrongState { .. }));
    }

    #[test]
    fn test_release_while_scheduled_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        let err = seq.release().unwrap_err();
        assert!(matches!(err, ApplyError::WrongState { .. }));
    }

    #[test]
    fn test_reacquire_while_scheduled_rejected() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(4, &manager).unwrap();
        let err = seq.reacquire(&manager).unwrap_err();
        assert!(matches!(err, ApplyError::WrongState { .. }));
    }

    // =========================================================================
    // Dangling token tracking
    // =========================================================================

    #[test]
    fn test_dangling_tokens_tracking() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 20, BLOCK_SIZE, noop_delegate());

        // Before prefill: all tokens are "not yet computed"
        assert_eq!(seq.kv_position(), 0);
        assert_eq!(seq.tail_tokens(), 8);

        // After prefill with token: 1 dangling
        seq.schedule_prefill(8, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();
        assert_eq!(seq.kv_position(), 8);
        assert_eq!(seq.tail_tokens(), 1);

        // After decode: still 1 dangling
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();
        assert_eq!(seq.kv_position(), 9);
        assert_eq!(seq.tail_tokens(), 1);

        // After speculative (accept 2): still 1 dangling
        seq.schedule_speculative(3, &manager).unwrap();
        seq.apply_speculative(&[200, 201], &manager).unwrap();
        assert_eq!(seq.kv_position(), 11);
        assert_eq!(seq.tail_tokens(), 1);
    }

    #[test]
    fn test_decode_requires_one_dangling() {
        let manager = create_test_manager::<TestMeta>(20);

        // 0 dangling: empty sequence with no tokens
        let mut seq =
            SchedulableSequence::<TestMeta>::new(vec![], 10, BLOCK_SIZE, noop_delegate());
        let err = seq.schedule_decode(&manager).unwrap_err();
        assert!(matches!(
            err,
            ScheduleError::WrongDanglingCount {
                expected: 1,
                actual: 0,
            }
        ));

        // 2 dangling: append 2 tokens
        seq.append_tokens(&[100, 101]).unwrap();
        assert_eq!(seq.tail_tokens(), 2);
        let err = seq.schedule_decode(&manager).unwrap_err();
        assert!(matches!(
            err,
            ScheduleError::WrongDanglingCount {
                expected: 1,
                actual: 2,
            }
        ));
    }

    #[test]
    fn test_append_tokens_creates_dangling() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(vec![], 10, BLOCK_SIZE, noop_delegate());

        assert_eq!(seq.tail_tokens(), 0);

        seq.append_tokens(&[100]).unwrap();
        assert_eq!(seq.tail_tokens(), 1);
        assert_eq!(seq.total_tokens(), 1);
        assert_eq!(seq.kv_position(), 0);

        // Now can schedule decode
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(101, &manager).unwrap();
        assert_eq!(seq.tail_tokens(), 1);
        assert_eq!(seq.kv_position(), 1);
    }

    #[test]
    fn test_kv_position_through_lifecycle() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(8), 20, BLOCK_SIZE, noop_delegate());

        assert_eq!(seq.kv_position(), 0);

        // Chunked prefill
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(None, &manager).unwrap();
        assert_eq!(seq.kv_position(), 4);

        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();
        assert_eq!(seq.kv_position(), 8);

        // Decode
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();
        assert_eq!(seq.kv_position(), 9);

        // Speculative
        seq.schedule_speculative(3, &manager).unwrap();
        seq.apply_speculative(&[200, 201, 202], &manager).unwrap();
        assert_eq!(seq.kv_position(), 12);
    }

    #[test]
    fn test_pending_completion_staged_during_decode() {
        let manager = create_test_manager::<TestMeta>(20);
        // 7 input tokens: block 0 complete (0-3), block 1 partial (4-6)
        let mut seq =
            SchedulableSequence::<TestMeta>::new(make_tokens(7), 10, BLOCK_SIZE, noop_delegate());

        seq.schedule_prefill(7, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();

        // After prefill: token 1000 completed block 1 (4,5,6,1000) but NOT registered
        assert_eq!(seq.assigned_blocks(), 1); // only block 0
        assert_eq!(seq.unassigned_blocks(), 0);
        assert_eq!(seq.kv_position(), 7);
        assert_eq!(seq.tail_tokens(), 1); // token 1000

        // schedule_decode detects the pending completion
        seq.schedule_decode(&manager).unwrap();
        assert_eq!(
            seq.state(),
            SequenceState::DecodeScheduled {
                blocks_allocated: 2
            }
        ); // 1 pending + 1 gen

        // apply_decode stages the pending block
        let outcome = seq.apply_decode(100, &manager).unwrap();
        assert_eq!(outcome, DecodeOutcome::Continue); // no boundary from THIS token
        assert_eq!(seq.assigned_blocks(), 2); // block 1 now registered
        assert_eq!(seq.unassigned_blocks(), 1); // gen block
        assert_eq!(seq.kv_position(), 8);
        assert_eq!(seq.tail_tokens(), 1);
    }

    // =========================================================================
    // Builder
    // =========================================================================

    #[test]
    fn test_builder_basic() {
        let seq = SchedulableSequence::<TestMeta>::builder()
            .tokens(make_tokens(8))
            .max_output_tokens(10)
            .block_size(BLOCK_SIZE)
            .build::<TestMeta>()
            .unwrap();

        assert_eq!(seq.state(), SequenceState::Idle);
        assert_eq!(seq.num_input_tokens(), 8);
        assert_eq!(seq.max_output_tokens(), 10);
        assert_eq!(seq.block_size(), BLOCK_SIZE as usize);
    }

    #[test]
    fn test_builder_with_delegate() {
        let delegate = Arc::new(CollectingDelegate::new());
        let seq = SchedulableSequence::<TestMeta>::builder()
            .tokens(make_tokens(4))
            .max_output_tokens(5)
            .block_size(BLOCK_SIZE)
            .delegate(delegate.clone())
            .build::<TestMeta>()
            .unwrap();

        assert_eq!(seq.num_input_tokens(), 4);

        let events = delegate.events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SequenceEvent::Created { .. }));
    }

    #[test]
    fn test_builder_missing_required_field() {
        let result = SchedulableSequence::<TestMeta>::builder()
            .tokens(make_tokens(4))
            // missing max_output_tokens and block_size
            .build::<TestMeta>();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_default_noop_delegate() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = SchedulableSequence::<TestMeta>::builder()
            .tokens(make_tokens(4))
            .max_output_tokens(10)
            .block_size(BLOCK_SIZE)
            .build::<TestMeta>()
            .unwrap();

        // Verify the noop delegate doesn't panic — exercise the full lifecycle
        seq.schedule_prefill(4, &manager).unwrap();
        seq.apply_prefill(Some(1000), &manager).unwrap();
        seq.schedule_decode(&manager).unwrap();
        seq.apply_decode(100, &manager).unwrap();
        seq.release().unwrap();
    }
}
