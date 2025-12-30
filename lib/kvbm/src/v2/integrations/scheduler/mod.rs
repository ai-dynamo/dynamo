// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust scheduler for G1 (GPU) block management.
//!
//! This module provides a modular scheduler that manages KV cache blocks on
//! GPU memory (G1 tier). It is designed to be a simplified, flexible implementation
//! inspired by vLLM's scheduler.
//!
//! # Architecture
//!
//! The scheduler is composed of several modular components:
//!
//! - **KVCacheManager**: Wraps BlockManager<G1> to provide block allocation/deallocation
//! - **RequestQueues**: Manages waiting and running request queues
//! - **SchedulingPolicy**: Trait for pluggable scheduling algorithms (FCFS by default)
//! - **Scheduler**: The main scheduler that orchestrates scheduling decisions
//!
//! # Optional Shared State
//!
//! The scheduler can optionally integrate with the ConnectorLeader via shared state.
//! When `shared_state` is Some, the scheduler can communicate request lifecycle
//! events to the connector. When None, the scheduler operates independently.

mod config;
mod core;
mod kv_cache;
mod policy;
mod queues;
mod request;

#[cfg(test)]
mod tests;

pub use config::{SchedulerConfig, SchedulerConfigBuilder, SchedulerConfigBuilderError};
pub use core::Scheduler;
pub use kv_cache::{AllocatedBlocks, KVCacheManager, RequestBlockState};
pub use policy::{FCFSPolicy, SchedulingPolicy};
pub use queues::{RunningRequests, WaitingQueue};
pub use request::{RequestStatus, SchedulerRequest};

