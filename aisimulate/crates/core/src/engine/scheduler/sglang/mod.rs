// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang scheduler simulation with adaptive admission control.
//!
//! Reference: sglang/python/sglang/srt/managers/scheduler.py

mod config;
mod core;
mod decode;
mod policy;
mod prefill;
mod request;

pub(crate) use core::SglangCore;

#[cfg(test)]
mod tests;
