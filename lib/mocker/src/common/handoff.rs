// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::protocols::{EngineType, KvTransferTimingMode};

/// Stable identifier for one prefill-to-decode handoff attempt.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HandoffId(Uuid);

impl HandoffId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for HandoffId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for HandoffId {
    fn from(value: Uuid) -> Self {
        Self(value)
    }
}

impl From<HandoffId> for Uuid {
    fn from(value: HandoffId) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HandoffOrder {
    SourceFirst,
    DestinationFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HandoffTransferTiming {
    pub mode: KvTransferTimingMode,
    pub full_prompt_tokens: usize,
    pub kv_bytes_per_token: Option<usize>,
    pub bandwidth_gb_s: Option<f64>,
}

impl HandoffTransferTiming {
    pub fn delay_ms(self, destination_missing_tokens: usize) -> Option<f64> {
        let tokens = match self.mode {
            KvTransferTimingMode::FullPrompt => self.full_prompt_tokens,
            KvTransferTimingMode::DestinationMissing => destination_missing_tokens,
        };
        let (Some(bytes_per_token), Some(bandwidth_gb_s)) =
            (self.kv_bytes_per_token, self.bandwidth_gb_s)
        else {
            return None;
        };
        if bandwidth_gb_s <= 0.0 {
            return None;
        }
        Some(tokens as f64 * bytes_per_token as f64 / (bandwidth_gb_s * 1e9) * 1000.0)
    }

    pub fn full_prompt_delay_ms(self) -> Option<f64> {
        let full_prompt = Self {
            mode: KvTransferTimingMode::FullPrompt,
            ..self
        };
        full_prompt.delay_ms(0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HandoffFact {
    SourceHeld {
        handoff_id: HandoffId,
        transfer_timing: HandoffTransferTiming,
    },
    DestinationReserved {
        handoff_id: HandoffId,
        transferable_prompt_tokens: usize,
    },
    TransferCompleted {
        handoff_id: HandoffId,
    },
    Failed {
        handoff_id: HandoffId,
    },
    TimedOut {
        handoff_id: HandoffId,
    },
    Canceled {
        handoff_id: HandoffId,
    },
}

impl HandoffFact {
    fn handoff_id(&self) -> HandoffId {
        match *self {
            Self::SourceHeld { handoff_id, .. }
            | Self::DestinationReserved { handoff_id, .. }
            | Self::TransferCompleted { handoff_id }
            | Self::Failed { handoff_id }
            | Self::TimedOut { handoff_id }
            | Self::Canceled { handoff_id } => handoff_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HandoffAction {
    SubmitPrefill {
        handoff_id: HandoffId,
    },
    ReserveDestination {
        handoff_id: HandoffId,
    },
    StartTransfer {
        handoff_id: HandoffId,
        delay_ms: f64,
    },
    ActivateDestination {
        handoff_id: HandoffId,
    },
    ReleaseSource {
        handoff_id: HandoffId,
    },
    CancelSource {
        handoff_id: HandoffId,
    },
    CancelDestination {
        handoff_id: HandoffId,
    },
    Complete {
        handoff_id: HandoffId,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct HandoffActionId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IssuedHandoffAction {
    pub id: HandoffActionId,
    pub action: HandoffAction,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HandoffActionOutcome {
    Submitted,
    Accepted,
    Scheduled,
    Applied,
    Noop,
    Failed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoordinatorMode {
    Active,
    CleaningUp,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffCompletion {
    Success,
    Canceled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NormalizedHandoffEvent {
    SourceHeld,
    DestinationAccepted,
    DestinationReserved,
    DestinationActivated,
    SourceReleased,
    Completed,
}

/// Surface-independent summary used to compare offline replay with the live
/// handoff driver. This is public only so cross-crate conformance tests can
/// exercise both implementations.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizedHandoffConformance {
    pub engine_type: EngineType,
    pub order: HandoffOrder,
    pub lifecycle: Vec<NormalizedHandoffEvent>,
    pub source_output_tokens: usize,
    pub destination_output_tokens: usize,
    pub completed_requests: usize,
    pub destination_stored: NormalizedStoredTiming,
    pub source_drained: bool,
    pub destination_drained: bool,
    pub driver_drained: bool,
}

#[doc(hidden)]
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizedStoredTiming {
    pub before_activation: usize,
    pub on_activation: usize,
    pub repeated_activation_hashes_after_activation: usize,
}

impl NormalizedHandoffConformance {
    /// Validate the deterministic one-request fixture shared by offline and
    /// live conformance tests.
    #[doc(hidden)]
    pub fn validate(&self) -> Result<()> {
        let expected_order = match self.engine_type {
            EngineType::Vllm => HandoffOrder::SourceFirst,
            EngineType::Sglang => HandoffOrder::DestinationFirst,
            EngineType::Trtllm => bail!("TRT-LLM does not support destination handoff"),
        };
        if self.order != expected_order {
            bail!(
                "normalized handoff order mismatch: expected {expected_order:?}, got {:?}",
                self.order
            );
        }
        if self.lifecycle != expected_normalized_handoff(self.order) {
            bail!(
                "normalized handoff lifecycle mismatch: expected {:?}, got {:?}",
                expected_normalized_handoff(self.order),
                self.lifecycle
            );
        }
        if self.source_output_tokens != 1 {
            bail!(
                "normalized source output count mismatch: expected 1, got {}",
                self.source_output_tokens
            );
        }
        if self.destination_output_tokens != 2 {
            bail!(
                "normalized destination output count mismatch: expected 2, got {}",
                self.destination_output_tokens
            );
        }
        if self.completed_requests != 1 {
            bail!(
                "normalized completion count mismatch: expected 1, got {}",
                self.completed_requests
            );
        }
        if self.destination_stored.before_activation != 0 {
            bail!(
                "destination published {} KV blocks before activation",
                self.destination_stored.before_activation
            );
        }
        if self.destination_stored.on_activation == 0 {
            bail!("destination activation published no KV blocks");
        }
        if self
            .destination_stored
            .repeated_activation_hashes_after_activation
            != 0
        {
            bail!(
                "destination republished {} activation KV blocks",
                self.destination_stored
                    .repeated_activation_hashes_after_activation
            );
        }
        if !self.source_drained || !self.destination_drained || !self.driver_drained {
            bail!(
                "handoff did not drain: source={}, destination={}, driver={}",
                self.source_drained,
                self.destination_drained,
                self.driver_drained
            );
        }
        Ok(())
    }
}

pub fn expected_normalized_handoff(order: HandoffOrder) -> &'static [NormalizedHandoffEvent] {
    use NormalizedHandoffEvent::*;
    match order {
        HandoffOrder::SourceFirst => &[
            SourceHeld,
            DestinationAccepted,
            DestinationReserved,
            DestinationActivated,
            SourceReleased,
            Completed,
        ],
        HandoffOrder::DestinationFirst => &[
            DestinationAccepted,
            DestinationReserved,
            SourceHeld,
            DestinationActivated,
            SourceReleased,
            Completed,
        ],
    }
}

#[derive(Default)]
struct ActionJournal {
    started: bool,
    next_id: u64,
    issued: FxHashMap<HandoffActionId, HandoffAction>,
    outcomes: FxHashMap<HandoffActionId, HandoffActionOutcome>,
}

#[derive(Default)]
struct SourceProgress {
    submit_issued: bool,
    submitted: bool,
    held: bool,
    transfer_timing: Option<HandoffTransferTiming>,
    release_issued: bool,
    cancel_issued: bool,
    cleanup_done: bool,
}

#[derive(Default)]
struct DestinationProgress {
    reserve_issued: bool,
    accepted: bool,
    reserved: bool,
    transferable_prompt_tokens: Option<usize>,
    activation_issued: bool,
    activation_applied: bool,
    cancel_issued: bool,
    cleanup_done: bool,
}

#[derive(Default)]
struct TransferProgress {
    issued: bool,
    scheduled: bool,
    completed: bool,
}

/// Pure state machine for one prefill-to-decode ownership handoff.
///
/// Drivers execute returned actions and feed action outcomes and asynchronous
/// facts back into this core. The core owns ordering only; it owns no engine or
/// transport resources.
pub struct HandoffCoordinatorCore {
    handoff_id: HandoffId,
    order: HandoffOrder,
    mode: CoordinatorMode,
    actions: ActionJournal,
    source: SourceProgress,
    destination: DestinationProgress,
    transfer: TransferProgress,
    completion: Option<HandoffCompletion>,
}

impl HandoffCoordinatorCore {
    pub fn new(handoff_id: HandoffId, order: HandoffOrder) -> Self {
        Self {
            handoff_id,
            order,
            mode: CoordinatorMode::Active,
            actions: ActionJournal::default(),
            source: SourceProgress::default(),
            destination: DestinationProgress::default(),
            transfer: TransferProgress::default(),
            completion: None,
        }
    }

    pub fn start(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.actions.started {
            return Ok(Vec::new());
        }
        self.actions.started = true;
        let action = match self.order {
            HandoffOrder::SourceFirst => self.issue_submit_prefill(),
            HandoffOrder::DestinationFirst => self.issue_reserve_destination(),
        };
        Ok(vec![action])
    }

    pub fn on_fact(&mut self, fact: HandoffFact) -> Result<Vec<IssuedHandoffAction>> {
        self.validate_handoff(fact.handoff_id())?;
        if self.mode != CoordinatorMode::Active {
            return Ok(Vec::new());
        }

        match fact {
            HandoffFact::SourceHeld {
                transfer_timing, ..
            } => {
                if self.source.held {
                    return Ok(Vec::new());
                }
                if !self.source.submitted {
                    bail!("source held before prefill submission was acknowledged");
                }
                validate_transfer_timing(transfer_timing)?;
                self.source.held = true;
                self.source.transfer_timing = Some(transfer_timing);
                self.advance_active()
            }
            HandoffFact::DestinationReserved {
                transferable_prompt_tokens,
                ..
            } => {
                if self.destination.reserved {
                    return Ok(Vec::new());
                }
                if !self.destination.accepted {
                    bail!("destination reserved before ownership was accepted");
                }
                self.destination.reserved = true;
                self.destination.transferable_prompt_tokens = Some(transferable_prompt_tokens);
                self.advance_active()
            }
            HandoffFact::TransferCompleted { .. } => {
                if self.transfer.completed {
                    return Ok(Vec::new());
                }
                if !self.transfer.scheduled {
                    bail!("transfer completed before it was scheduled");
                }
                self.transfer.completed = true;
                self.advance_active()
            }
            HandoffFact::Failed { .. }
            | HandoffFact::TimedOut { .. }
            | HandoffFact::Canceled { .. } => self.begin_cleanup(),
        }
    }

    pub fn on_action_outcome(
        &mut self,
        action_id: HandoffActionId,
        outcome: HandoffActionOutcome,
    ) -> Result<Vec<IssuedHandoffAction>> {
        if self.mode == CoordinatorMode::Complete {
            return Ok(Vec::new());
        }
        let Some(action) = self.actions.issued.get(&action_id).copied() else {
            bail!("unknown handoff action {action_id:?}");
        };
        if let Some(previous) = self.actions.outcomes.get(&action_id) {
            if previous != &outcome {
                bail!("conflicting outcome for handoff action {action_id:?}");
            }
            return Ok(Vec::new());
        }
        self.actions.outcomes.insert(action_id, outcome.clone());

        if let HandoffActionOutcome::Failed(_) = outcome {
            if matches!(
                action,
                HandoffAction::CancelSource { .. } | HandoffAction::CancelDestination { .. }
            ) {
                bail!("handoff cleanup action {action_id:?} failed");
            }
            return self.begin_cleanup();
        }

        match action {
            HandoffAction::SubmitPrefill { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Submitted])?;
                self.source.submitted = true;
            }
            HandoffAction::ReserveDestination { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Accepted])?;
                self.destination.accepted = true;
            }
            HandoffAction::StartTransfer { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Scheduled])?;
                self.transfer.scheduled = true;
            }
            HandoffAction::ActivateDestination { .. } => {
                require_outcome(&outcome, &[HandoffActionOutcome::Applied])?;
                self.destination.activation_applied = true;
            }
            HandoffAction::ReleaseSource { .. } => {
                require_outcome(
                    &outcome,
                    &[HandoffActionOutcome::Applied, HandoffActionOutcome::Noop],
                )?;
                self.source.cleanup_done = true;
            }
            HandoffAction::CancelSource { .. } => {
                require_outcome(
                    &outcome,
                    &[HandoffActionOutcome::Applied, HandoffActionOutcome::Noop],
                )?;
                self.source.cleanup_done = true;
            }
            HandoffAction::CancelDestination { .. } => {
                require_outcome(
                    &outcome,
                    &[HandoffActionOutcome::Applied, HandoffActionOutcome::Noop],
                )?;
                self.destination.cleanup_done = true;
            }
            HandoffAction::Complete { .. } => return Ok(Vec::new()),
        }

        match self.mode {
            CoordinatorMode::Active => self.advance_active(),
            CoordinatorMode::CleaningUp => self.advance_cleanup(),
            CoordinatorMode::Complete => Ok(Vec::new()),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.mode == CoordinatorMode::Complete
    }

    pub fn completion(&self) -> Option<HandoffCompletion> {
        self.completion
    }

    fn advance_active(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.order == HandoffOrder::SourceFirst
            && self.source.held
            && !self.destination.reserve_issued
        {
            return Ok(vec![self.issue_reserve_destination()]);
        }
        if self.order == HandoffOrder::DestinationFirst
            && self.destination.reserved
            && !self.source.submit_issued
        {
            return Ok(vec![self.issue_submit_prefill()]);
        }
        if self.source.held && self.destination.reserved && !self.transfer.issued {
            self.transfer.issued = true;
            let transfer_timing = self
                .source
                .transfer_timing
                .expect("held source must retain transfer timing");
            let transferable_prompt_tokens = self
                .destination
                .transferable_prompt_tokens
                .expect("reserved destination must report its transferable footprint");
            return Ok(vec![
                self.issue(HandoffAction::StartTransfer {
                    handoff_id: self.handoff_id,
                    delay_ms: transfer_timing
                        .delay_ms(transferable_prompt_tokens)
                        .unwrap_or_default(),
                }),
            ]);
        }
        if self.transfer.completed && !self.destination.activation_issued {
            self.destination.activation_issued = true;
            return Ok(vec![self.issue(HandoffAction::ActivateDestination {
                handoff_id: self.handoff_id,
            })]);
        }
        if self.destination.activation_applied && !self.source.release_issued {
            self.source.release_issued = true;
            return Ok(vec![self.issue(HandoffAction::ReleaseSource {
                handoff_id: self.handoff_id,
            })]);
        }
        if self.source.cleanup_done {
            return Ok(vec![self.complete()]);
        }
        Ok(Vec::new())
    }

    fn begin_cleanup(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        if self.mode == CoordinatorMode::Complete {
            return Ok(Vec::new());
        }
        self.mode = CoordinatorMode::CleaningUp;
        self.advance_cleanup()
    }

    fn advance_cleanup(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        let mut actions = Vec::new();
        if self.source.submit_issued && !self.source.cancel_issued && !self.source.cleanup_done {
            self.source.cancel_issued = true;
            actions.push(self.issue(HandoffAction::CancelSource {
                handoff_id: self.handoff_id,
            }));
        }
        if self.destination.reserve_issued
            && !self.destination.cancel_issued
            && !self.destination.cleanup_done
        {
            self.destination.cancel_issued = true;
            actions.push(self.issue(HandoffAction::CancelDestination {
                handoff_id: self.handoff_id,
            }));
        }
        if actions.is_empty()
            && (!self.source.submit_issued || self.source.cleanup_done)
            && (!self.destination.reserve_issued || self.destination.cleanup_done)
        {
            actions.push(self.complete());
        }
        Ok(actions)
    }

    fn issue_submit_prefill(&mut self) -> IssuedHandoffAction {
        self.source.submit_issued = true;
        self.issue(HandoffAction::SubmitPrefill {
            handoff_id: self.handoff_id,
        })
    }

    fn issue_reserve_destination(&mut self) -> IssuedHandoffAction {
        self.destination.reserve_issued = true;
        self.issue(HandoffAction::ReserveDestination {
            handoff_id: self.handoff_id,
        })
    }

    fn complete(&mut self) -> IssuedHandoffAction {
        self.completion = Some(if self.mode == CoordinatorMode::CleaningUp {
            HandoffCompletion::Canceled
        } else {
            HandoffCompletion::Success
        });
        self.mode = CoordinatorMode::Complete;
        let action = self.issue(HandoffAction::Complete {
            handoff_id: self.handoff_id,
        });
        self.actions.issued = FxHashMap::default();
        self.actions.outcomes = FxHashMap::default();
        action
    }

    fn issue(&mut self, action: HandoffAction) -> IssuedHandoffAction {
        let id = HandoffActionId(self.actions.next_id);
        self.actions.next_id = self
            .actions
            .next_id
            .checked_add(1)
            .expect("handoff action ID overflow");
        let previous = self.actions.issued.insert(id, action);
        debug_assert!(previous.is_none());
        IssuedHandoffAction { id, action }
    }

    fn validate_handoff(&self, handoff_id: HandoffId) -> Result<()> {
        if handoff_id != self.handoff_id {
            bail!("fact belongs to a different handoff");
        }
        Ok(())
    }
}

pub fn validate_transfer_delay_ms(transfer_delay_ms: Option<f64>) -> Result<()> {
    let Some(delay_ms) = transfer_delay_ms else {
        return Ok(());
    };
    if !delay_ms.is_finite() || delay_ms < 0.0 {
        bail!("invalid handoff transfer delay {delay_ms}");
    }
    Ok(())
}

pub fn validate_transfer_timing(transfer_timing: HandoffTransferTiming) -> Result<()> {
    if let Some(bandwidth_gb_s) = transfer_timing.bandwidth_gb_s
        && (!bandwidth_gb_s.is_finite() || bandwidth_gb_s <= 0.0)
    {
        bail!("invalid handoff transfer bandwidth {bandwidth_gb_s}");
    }
    validate_transfer_delay_ms(transfer_timing.full_prompt_delay_ms())
}

fn require_outcome(outcome: &HandoffActionOutcome, allowed: &[HandoffActionOutcome]) -> Result<()> {
    if allowed.contains(outcome) {
        return Ok(());
    }
    bail!("invalid handoff action outcome {outcome:?}")
}

#[cfg(test)]
#[path = "handoff_tests.rs"]
mod coordinator_tests;
