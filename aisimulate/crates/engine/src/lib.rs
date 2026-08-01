// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral mock inference schedulers and attention-DP composition.

mod cache;
mod common;
mod config;
mod engine;
pub mod generalized;
mod handoff;
mod kv_manager;
mod protocol;
mod scheduler;
mod timing;
mod trace;

pub use common::running_mean::RunningMean;
pub use common::speculative::normalize_conditional_accept_rates;
pub use config::{
    Backend, EngineConfig, PreemptionMode, SglangConfig, SglangSchedulePolicy,
    TrtllmCapacityPolicy, TrtllmConfig, WorkerType,
};
pub use engine::{Engine, EngineFactory};
pub use handoff::{HandoffId, HandoffTransferTiming, TransferTimingMode, prefill_handoff_delay_ms};
pub use protocol::{
    Admission, Command, CommandEffects, CommandResult, ForwardPassMetrics, KvBlock, KvEvent,
    KvEventData, LifecycleEvent, Metrics, Output, PassCompletionEffects, PassStartEffects,
    PressureEvent, PressureKind, PressureState, Request, StoredBlocks,
};
pub use scheduler::SchedulerRank;
pub use timing::{TimingModel, TimingModelConfig};

#[doc(hidden)]
pub use protocol::PendingPass;
pub(crate) use timing::modeled_duration_ms;
#[doc(hidden)]
pub use trace::g1_parent_chain_events;
