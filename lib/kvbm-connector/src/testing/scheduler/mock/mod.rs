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
//! use kvbm_connector::testing::scheduler::mock::{MockEngineCore, TestRequest};
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

// Source: mock/mod.rs
// TODO: Disabled — engine.rs depends on v2::integrations::scheduler::{KVCacheManager, Scheduler, SchedulerConfig}
// which has no workspace equivalent. Re-enable when integrations/scheduler is ported.
#[cfg(TODO)]
mod engine;

mod model;

// Source: pub use engine::{MockEngineCore, ...}
// TODO: Disabled — engine types depend on Scheduler
#[cfg(TODO)]
pub use engine::{MockEngineCore, MockEngineCoreConfig, StepOutput, TestRequest};

pub use model::MockModelRunner;

// Source: mod tests, mod abort_tests, mod connector_e2e_tests
// TODO: Disabled — all test files use Scheduler from v2::integrations::scheduler
#[cfg(TODO)]
mod tests;

#[cfg(TODO)]
mod abort_tests;

#[cfg(TODO)]
mod connector_e2e_tests;
