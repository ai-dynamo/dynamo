// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-neutral inference simulation and deterministic replay.
//!
//! [`engine`] owns scheduling, native GPU KV accounting, preemption, timing,
//! and attention-DP composition. [`replay`] owns virtual time, logical-worker
//! lifecycle, placement/scaling composition, and report collection.
//!
//! The crate root exposes the stable, commonly used configuration and replay
//! surface. Advanced engine and adapter contracts remain available through
//! their explicit module paths.

pub mod engine;
pub mod replay;

pub use engine::{EngineConfig, TimingModel, TimingModelConfig};
pub use replay::{ReplayReport, ReplaySpec, Replayer};
