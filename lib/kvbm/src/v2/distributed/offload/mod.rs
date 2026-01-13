// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offload Engine for asynchronous block transfers between storage tiers.
//!
//! The offload engine provides a policy-based, cancellable pipeline for moving
//! blocks from higher-performance tiers (G1/G2) to lower-cost tiers (G3/G4).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        OffloadEngine                            │
//! │                                                                 │
//! │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    │
//! │  │G1→G2 Pipeline │────│ G2→G3 Pipeline│    │ G2→G4 Pipeline│    │
//! │  └───────────────┘    └───────────────┘    └───────────────┘    │
//! │         │                     │                     │           │
//! │         └─────────auto_chain──┘---------------------┘           │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! Pipeline stages:
//! ┌─────────────┐    ┌────────────────┐    ┌──────────────────┐
//! │   Policy    │───▶│     Batch      │───▶│    Transfer      │
//! │  Evaluator  │    │   Collector    │    │    Executor      │
//! └─────────────┘    └────────────────┘    └──────────────────┘
//!       │                   │                      │
//!       ▼                   ▼                      ▼
//!   cancel check       cancel check          wait for in-flight
//! ```
//!
//! # Features
//!
//! - **Policy-based filtering**: Blocks pass through configurable policies
//!   (presence checks, LFU thresholds) before transfer
//! - **Batched transfers**: Blocks are accumulated into batches for efficient
//!   bulk transfers
//! - **Cancellation**: Clean cancellation with confirmation that all blocks
//!   are released and no outstanding operations remain
//! - **Pipeline chaining**: G1→G2 completions can automatically feed G2→G3
//!
//! # Example
//!
//! ```ignore
//! use kvbm::v2::distributed::offload::{
//!     OffloadEngine, PipelineBuilder, PresenceFilter, PresenceAndLFUFilter,
//! };
//!
//! // Build engine with pipelines
//! let engine = OffloadEngine::builder(leader.clone())
//!     .with_registry(registry.clone())
//!     .with_g2_manager(g2_manager.clone())
//!     .with_g3_manager(g3_manager.clone())
//!     .with_g2_to_g3_pipeline(
//!         PipelineBuilder::<G2, G3>::new()
//!             .policy(Arc::new(PresenceAndLFUFilter::with_default_threshold(registry.clone())))
//!             .batch_size(64)
//!             .build()
//!     )
//!     .build()?;
//!
//! // Enqueue blocks for offload
//! let handle = engine.enqueue_g2_to_g3(blocks)?;
//!
//! // Wait for completion or cancel
//! tokio::select! {
//!     result = handle.wait() => {
//!         println!("Completed: {:?}", result?.completed_blocks);
//!     }
//!     _ = shutdown_signal => {
//!         handle.cancel().await;
//!         println!("Cancelled");
//!     }
//! }
//! ```

mod batch;
mod cancel;
mod engine;
mod handle;
mod pending;
mod pipeline;
mod policy;
mod queue;
mod source;

#[cfg(test)]
mod cancel_tests;

// Re-export public API
pub use cancel::{CancelConfirmation, CancelState, CancellationToken};
pub use engine::{OffloadEngine, OffloadEngineBuilder};
pub use handle::{TransferHandle, TransferId, TransferResult, TransferStatus};
pub use pending::{PendingGuard, PendingTracker};
pub use pipeline::{
    ObjectPipeline, ObjectPipelineBuilder, ObjectPipelineConfig, Pipeline, PipelineBuilder,
    PipelineConfig, ResolvedBatch, ResolvedBlock, upgrade_batch,
};
pub use policy::{
    AllOfPolicy, AnyOfPolicy, BoxFuture, EvalContext, ObjectLockPresenceFilter,
    ObjectPresenceFilter, OffloadPolicy, PassAllPolicy, PolicyBatchFuture, PolicyFuture,
    PresenceAndLFUFilter, PresenceChecker, PresenceFilter, S3PresenceChecker, async_batch_result,
    async_result, create_policy_from_config, sync_batch_result, sync_result,
};
pub use queue::CancellableQueue;
pub use source::{ExternalBlock, SourceBlock, SourceBlocks};

// Re-export batch config for advanced users
pub use batch::{BatchConfig, TimingTrace};
