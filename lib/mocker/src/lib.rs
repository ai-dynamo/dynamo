// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock LLM scheduler and KV manager for testing.
//!
//! This crate provides a mock implementation of an LLM scheduler that simulates
//! KV cache management, request scheduling, and token generation timing without
//! requiring actual GPU resources or a full distributed runtime.

pub mod bootstrap;
pub mod evictor;
pub mod kv_manager;
pub mod perf_model;
pub mod protocols;
pub mod running_mean;
pub mod scheduler;
pub mod sequence;

// Re-export commonly used types
pub use protocols::{DirectRequest, KvCacheEventSink, MockEngineArgs, MockEngineArgsBuilder};
pub use scheduler::Scheduler;
