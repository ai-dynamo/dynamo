// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock engine core for CPU-only scheduler testing.
//!
//! This module provides a mock engine that drives the real Scheduler without GPU.
//! It generates deterministic "model outputs" using seeded random tokens, enabling
//! fast, reproducible tests for scheduler state evaluation.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                   MockEngineCore (Rust)                     │
//! ├────────────────────────────────────────────────────────────┤
//! │  ┌──────────────────┐   ┌─────────────────────────────┐   │
//! │  │ MockModelRunner  │   │   Real Scheduler (core.rs)  │   │
//! │  │ (seeded random)  │──▶│   + Real KVCacheManager     │   │
//! │  └──────────────────┘   └─────────────────────────────┘   │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use dynamo_kvbm::v2::testing::scheduler::mock::{MockEngineCore, TestRequest};
//!
//! let config = MockEngineCoreConfig::default();
//! let mut engine = MockEngineCore::new(config).unwrap();
//!
//! engine.add_request(TestRequest {
//!     request_id: "test-1".into(),
//!     prompt_tokens: (0..100).collect(),
//!     max_tokens: 50,
//! });
//!
//! let outputs = engine.run_to_completion(1000);
//! ```

mod engine;
mod model;

pub use engine::{MockEngineCore, MockEngineCoreConfig, StepOutput, TestRequest};
pub use model::MockModelRunner;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod abort_tests;
