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
//! # Connector Integration
//!
//! The scheduler can optionally integrate with a [`ConnectorLeader`] to enable
//! intelligent eviction and KV cache offloading. When a connector is attached
//! via [`SchedulerBuilder::connector`], the scheduler gains access to:
//!
//! - **Inflight transfer awareness**: The connector tracks active G1→G2 offload
//!   operations. Requests with inflight offloads cannot be evicted (their source
//!   blocks are being read by RDMA transfers).
//!
//! - **G2 block availability**: The connector knows which blocks exist in G2
//!   (host memory). Requests with more G2 coverage are better eviction candidates
//!   because they require less or no prefill computation when resumed.
//!
//! - **Request lifecycle coordination**: On request completion, the scheduler
//!   checks with the connector whether to delay block freeing (for offload
//!   completion).
//!
//! ## Eviction Criteria
//!
//! When memory pressure requires preemption, the scheduler considers three factors:
//!
//! ```text
//!                     ┌─────────────────────────────────────────┐
//!                     │         Eviction Candidate Selection     │
//!                     └─────────────────────────────────────────┘
//!                                        │
//!               ┌────────────────────────┼────────────────────────┐
//!               │                        │                        │
//!               ▼                        ▼                        ▼
//!     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//!     │ 1. Can Evict?   │     │ 2. G2 Coverage  │     │ 3. Block Align  │
//!     │                 │     │                 │     │                 │
//!     │ No inflight     │     │ Prefer requests │     │ Prefer requests │
//!     │ offloads        │     │ with more G2    │     │ at block        │
//!     │                 │     │ blocks          │     │ boundaries      │
//!     └─────────────────┘     └─────────────────┘     └─────────────────┘
//!               │                        │                        │
//!               └────────────────────────┼────────────────────────┘
//!                                        │
//!                                        ▼
//!                             ┌─────────────────┐
//!                             │ Selected Victim │
//!                             └─────────────────┘
//! ```
//!
//! ### 1. Inflight Offload Protection
//!
//! A request **cannot be evicted** if it has active G1→G2 transfers in progress.
//! Evicting would free G1 blocks that are being read by RDMA, causing data
//! corruption or undefined behavior.
//!
//! The connector's [`can_evict()`] method checks [`RequestSlot::has_inflight_offloads()`]
//! to determine if a request is safe to evict.
//!
//! ### 2. G2 Block Coverage Scoring
//!
//! Requests with more blocks already offloaded to G2 are preferred for eviction
//! because:
//!
//! - They can be resumed with minimal prefill (onboarding from G2 is fast)
//! - The work invested in offloading is preserved
//! - Memory is freed without losing computation
//!
//! The connector's [`get_eviction_score()`] returns coverage information used
//! by the scheduling policy to rank candidates.
//!
//! ### 3. Block Boundary Alignment (Future)
//!
//! Evicting at a block boundary is optimal because:
//!
//! - No partial block is wasted (current block is full)
//! - Continuing generation until block boundary costs zero extra resources
//! - On resume, we can prefill just the known next token for the new block
//!
//! This optimization requires preserving the last complete block's state and
//! the predicted first token of the next block for fast resumption.
//!
//! ## Connector API (vLLM Compatible)
//!
//! The scheduler's connector integration mirrors vLLM's `KVConnector` API:
//!
//! | vLLM Method | Our Method | When Called |
//! |-------------|------------|-------------|
//! | `get_num_new_matched_tokens()` | Same | New request scheduling |
//! | `update_state_after_alloc()` | Same | After block allocation |
//! | `request_finished()` | Same | Request completion |
//! | `build_connector_meta()` | Same | End of schedule() |
//! | (N/A) | `can_evict()` | **Before preemption** |
//! | (N/A) | `get_eviction_score()` | **Victim selection** |
//!
//! The new methods (`can_evict`, `get_eviction_score`) extend vLLM's API to
//! support intelligent eviction decisions.
//!
//! ## Future: Shared State Coordination
//!
//! For advanced eviction strategies, the scheduler and connector can share
//! state to coordinate:
//!
//! - **Proactive offloading**: Connector pre-offloads blocks for likely eviction
//!   candidates based on scheduling policy hints
//! - **G2 block reservation**: Connector reserves G2 space for eviction candidates
//! - **Resume prioritization**: Evicted requests with full G2 coverage get
//!   scheduling priority
//!
//! This coordination happens via the `SchedulerConnectorState` trait, which
//! provides a shared view of request state across both components.
//!
//! [`ConnectorLeader`]: crate::v2::integrations::connector::leader::ConnectorLeader
//! [`can_evict()`]: crate::v2::integrations::connector::leader::ConnectorLeader::can_evict
//! [`get_eviction_score()`]: crate::v2::integrations::connector::leader::ConnectorLeader::get_eviction_score
//! [`RequestSlot::has_inflight_offloads()`]: crate::v2::integrations::connector::leader::slot::RequestSlot::has_inflight_offloads

mod config;
mod core;
mod kv_cache;
mod policy;
mod projection;
mod queues;
mod request;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod trace_tests;


pub use config::{SchedulerConfig, SchedulerConfigBuilder, SchedulerConfigBuilderError};
pub use core::{Scheduler, SchedulerBuilder, SchedulerBuilderError};
pub use kv_cache::{AllocatedBlocks, KVCacheManager, RequestBlockState};
pub use policy::{FCFSPolicy, SchedulingPolicy};
pub use projection::{
    AggregateDemandEvent, BlockEvent, ChokePoint, FinishEntry, GlobalProjectionState, NextFinish,
    PlannedEviction, PlannedEvictionTracker, ProjectionState, RequestBlockSchedule, RequestPhase,
};
pub use queues::{PausedRequests, RunningRequests, WaitingQueue};
pub use request::{RequestStatus, SchedulerRequest};
