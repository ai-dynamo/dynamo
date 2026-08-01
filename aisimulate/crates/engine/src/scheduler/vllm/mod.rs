// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared vLLM/TRT-LLM scheduler simulation around a unified request model.
//!
//! vLLM and TRT-LLM share the queue, allocation, and lifecycle core. Their
//! admission and preemption differences live in [`policy`].

mod core;
mod policy;
mod request;

pub(crate) use core::VllmCore;

#[cfg(test)]
pub(crate) use core::RequestStatus;

#[cfg(test)]
mod tests;
