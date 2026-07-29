// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared vLLM/TRT-LLM scheduler simulation around a unified request model.
//!
//! vLLM and TRT-LLM share the queue, allocation, and lifecycle core. Their
//! admission and preemption differences live in [`policy`].

mod core;
mod live;
mod policy;

pub(crate) use core::VllmCore;
pub use live::{MockerMetrics, Scheduler};

/// Re-exported for policy tests that assert on request status through
/// [`VllmCore::state`]; only needed in test builds.
#[cfg(test)]
pub(crate) use core::RequestStatus;

#[cfg(test)]
mod tests;
