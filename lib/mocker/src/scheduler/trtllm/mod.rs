// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TRT-LLM scheduler simulation.
//!
//! TRT-LLM is not a separate engine in the mocker: `EngineType::Trtllm` routes
//! to the vLLM `Scheduler` / `VllmCore` (see [`crate::scheduler::vllm`]), which
//! switches behavior on `MockEngineArgs::scheduling_policy`. This module owns the
//! pieces that differ from vLLM.
//!
//! - **Reservation-based admission** — [`blocks_needed_to_finish`] and
//!   [`available_blocks`]: a waiting request is admitted only if its
//!   `prompt + max_output` footprint still fits after every running request's
//!   to-completion reservation.
//! - **No-preemption invariant** — [`is_no_evict`] gates the policy and
//!   [`report_no_evict_violation`] fails loudly if a preemption is ever required.

mod policy;

pub(crate) use policy::{
    available_blocks, blocks_needed_to_finish, is_no_evict, normalize_max_output_tokens,
    report_no_evict_violation,
};

#[cfg(test)]
mod tests;
