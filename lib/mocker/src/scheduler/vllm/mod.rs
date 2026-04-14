// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM scheduler simulation around a unified waiting/running request model.
//!
//! Reference: vllm/vllm/v1/core/sched/scheduler.py

mod core;
mod live;

pub(crate) use core::VllmCore;
pub use live::{MockerMetrics, Scheduler};

#[cfg(test)]
mod tests;
