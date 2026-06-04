// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM scheduler simulation around a unified waiting/running request model.
//!
//! Reference: vllm/vllm/v1/core/sched/scheduler.py

mod core;
mod live;

pub(crate) use core::VllmCore;
pub use live::{MockerMetrics, Scheduler};

/// Re-exported for the sibling `crate::scheduler::trtllm` tests, which assert on
/// request status through [`VllmCore::state`]; only needed in test builds.
#[cfg(test)]
pub(crate) use core::RequestStatus;

#[cfg(test)]
mod tests;
