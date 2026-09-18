// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo compatibility types for the shared Replay handoff coordinator.
//!
//! Dynamo retains its UUID transport DTOs at this boundary. Ordering and
//! cleanup are owned by `aisimulate_core::replay`; this module only converts between
//! the public Dynamo surface and Replay's runtime-neutral value types.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::protocols::{EngineType, KvTransferTimingMode};

pub use aisimulate_core::replay::{
    HandoffActionId, HandoffActionOutcome, HandoffCompletion, HandoffOrder, NormalizedHandoffEvent,
    NormalizedStoredTiming, expected_normalized_handoff, validate_transfer_delay_ms,
};

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

impl From<HandoffId> for aisimulate_core::replay::HandoffId {
    fn from(value: HandoffId) -> Self {
        Self::from(value.0)
    }
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
        replay_timing(self).delay_ms(destination_missing_tokens)
    }

    pub fn full_prompt_delay_ms(self) -> Option<f64> {
        replay_timing(self).full_prompt_delay_ms()
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IssuedHandoffAction {
    pub id: HandoffActionId,
    pub action: HandoffAction,
}

/// Compatibility summary used by Dynamo's live/offline conformance tests.
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

impl NormalizedHandoffConformance {
    #[doc(hidden)]
    pub fn validate(&self) -> Result<()> {
        aisimulate_core::replay::NormalizedHandoffConformance {
            engine_type: match self.engine_type {
                EngineType::Vllm => aisimulate_core::engine::Backend::Vllm,
                EngineType::Sglang => aisimulate_core::engine::Backend::Sglang,
                EngineType::Trtllm => aisimulate_core::engine::Backend::Trtllm,
            },
            order: self.order,
            lifecycle: self.lifecycle.clone(),
            source_output_tokens: self.source_output_tokens,
            destination_output_tokens: self.destination_output_tokens,
            completed_requests: self.completed_requests,
            destination_stored: self.destination_stored.clone(),
            source_drained: self.source_drained,
            destination_drained: self.destination_drained,
            driver_drained: self.driver_drained,
        }
        .validate()
    }
}

impl From<aisimulate_core::replay::NormalizedHandoffConformance> for NormalizedHandoffConformance {
    fn from(value: aisimulate_core::replay::NormalizedHandoffConformance) -> Self {
        Self {
            engine_type: match value.engine_type {
                aisimulate_core::engine::Backend::Vllm => EngineType::Vllm,
                aisimulate_core::engine::Backend::Sglang => EngineType::Sglang,
                aisimulate_core::engine::Backend::Trtllm => EngineType::Trtllm,
            },
            order: value.order,
            lifecycle: value.lifecycle,
            source_output_tokens: value.source_output_tokens,
            destination_output_tokens: value.destination_output_tokens,
            completed_requests: value.completed_requests,
            destination_stored: value.destination_stored,
            source_drained: value.source_drained,
            destination_drained: value.destination_drained,
            driver_drained: value.driver_drained,
        }
    }
}

/// Thin UUID-compatibility wrapper around Replay's single handoff state machine.
pub struct HandoffCoordinatorCore {
    inner: aisimulate_core::replay::HandoffCoordinatorCore,
}

impl HandoffCoordinatorCore {
    pub fn new(handoff_id: HandoffId, order: HandoffOrder) -> Self {
        Self {
            inner: aisimulate_core::replay::HandoffCoordinatorCore::new(handoff_id.into(), order),
        }
    }

    pub fn start(&mut self) -> Result<Vec<IssuedHandoffAction>> {
        Ok(self
            .inner
            .start()?
            .into_iter()
            .map(convert_action)
            .collect())
    }

    pub fn on_fact(&mut self, fact: HandoffFact) -> Result<Vec<IssuedHandoffAction>> {
        Ok(self
            .inner
            .on_fact(convert_fact(fact))?
            .into_iter()
            .map(convert_action)
            .collect())
    }

    pub fn on_action_outcome(
        &mut self,
        action_id: HandoffActionId,
        outcome: HandoffActionOutcome,
    ) -> Result<Vec<IssuedHandoffAction>> {
        Ok(self
            .inner
            .on_action_outcome(action_id, outcome)?
            .into_iter()
            .map(convert_action)
            .collect())
    }

    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    pub fn completion(&self) -> Option<HandoffCompletion> {
        self.inner.completion()
    }
}

pub fn validate_transfer_timing(transfer_timing: HandoffTransferTiming) -> Result<()> {
    aisimulate_core::replay::validate_transfer_timing(replay_timing(transfer_timing))
}

fn replay_timing(timing: HandoffTransferTiming) -> aisimulate_core::replay::HandoffTransferTiming {
    aisimulate_core::replay::HandoffTransferTiming {
        mode: match timing.mode {
            KvTransferTimingMode::FullPrompt => {
                aisimulate_core::engine::TransferTimingMode::FullPrompt
            }
            KvTransferTimingMode::DestinationMissing => {
                aisimulate_core::engine::TransferTimingMode::DestinationMissing
            }
        },
        full_prompt_tokens: timing.full_prompt_tokens,
        kv_bytes_per_token: timing.kv_bytes_per_token,
        bandwidth_gb_s: timing.bandwidth_gb_s,
    }
}

fn convert_fact(fact: HandoffFact) -> aisimulate_core::replay::HandoffFact {
    match fact {
        HandoffFact::SourceHeld {
            handoff_id,
            transfer_timing,
        } => aisimulate_core::replay::HandoffFact::SourceHeld {
            handoff_id: handoff_id.into(),
            transfer_timing: replay_timing(transfer_timing),
        },
        HandoffFact::DestinationReserved {
            handoff_id,
            transferable_prompt_tokens,
        } => aisimulate_core::replay::HandoffFact::DestinationReserved {
            handoff_id: handoff_id.into(),
            transferable_prompt_tokens,
        },
        HandoffFact::TransferCompleted { handoff_id } => {
            aisimulate_core::replay::HandoffFact::TransferCompleted {
                handoff_id: handoff_id.into(),
            }
        }
        HandoffFact::Failed { handoff_id } => aisimulate_core::replay::HandoffFact::Failed {
            handoff_id: handoff_id.into(),
        },
        HandoffFact::TimedOut { handoff_id } => aisimulate_core::replay::HandoffFact::TimedOut {
            handoff_id: handoff_id.into(),
        },
        HandoffFact::Canceled { handoff_id } => aisimulate_core::replay::HandoffFact::Canceled {
            handoff_id: handoff_id.into(),
        },
    }
}

fn convert_action(action: aisimulate_core::replay::IssuedHandoffAction) -> IssuedHandoffAction {
    let aisimulate_core::replay::IssuedHandoffAction { id, action } = action;
    let action = match action {
        aisimulate_core::replay::HandoffAction::SubmitPrefill { handoff_id } => {
            HandoffAction::SubmitPrefill {
                handoff_id: HandoffId::from(handoff_id.get()),
            }
        }
        aisimulate_core::replay::HandoffAction::ReserveDestination { handoff_id } => {
            HandoffAction::ReserveDestination {
                handoff_id: HandoffId::from(handoff_id.get()),
            }
        }
        aisimulate_core::replay::HandoffAction::StartTransfer {
            handoff_id,
            delay_ms,
        } => HandoffAction::StartTransfer {
            handoff_id: HandoffId::from(handoff_id.get()),
            delay_ms,
        },
        aisimulate_core::replay::HandoffAction::ActivateDestination { handoff_id } => {
            HandoffAction::ActivateDestination {
                handoff_id: HandoffId::from(handoff_id.get()),
            }
        }
        aisimulate_core::replay::HandoffAction::ReleaseSource { handoff_id } => {
            HandoffAction::ReleaseSource {
                handoff_id: HandoffId::from(handoff_id.get()),
            }
        }
        aisimulate_core::replay::HandoffAction::CancelSource { handoff_id } => {
            HandoffAction::CancelSource {
                handoff_id: HandoffId::from(handoff_id.get()),
            }
        }
        aisimulate_core::replay::HandoffAction::CancelDestination { handoff_id } => {
            HandoffAction::CancelDestination {
                handoff_id: HandoffId::from(handoff_id.get()),
            }
        }
        aisimulate_core::replay::HandoffAction::Complete { handoff_id } => {
            HandoffAction::Complete {
                handoff_id: HandoffId::from(handoff_id.get()),
            }
        }
    };
    IssuedHandoffAction { id, action }
}

#[cfg(test)]
#[path = "handoff_tests.rs"]
mod coordinator_tests;
