// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral mock-engine composition.
//!
//! [`RankEngine`] is the boundary implemented by one scheduler/KV/timing core.
//! [`GeneralizedMockerEngine`] composes one or more of those cores into one
//! logical worker. A one-rank engine and an attention-DP engine therefore have
//! the same caller-facing contract.
//!
//! Starting a pass eagerly commits every ready rank's work. Completion effects
//! remain hidden until [`GeneralizedMockerEngine::complete_pass`] is called.
//! With attention DP, all sibling ranks share the slowest rank's completion
//! boundary, including ranks that had no work when the pass started.

mod contracts;
mod engine;

pub use contracts::*;
pub use engine::*;

#[cfg(test)]
mod tests;
