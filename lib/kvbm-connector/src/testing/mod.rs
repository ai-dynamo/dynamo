// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-specific testing utilities.
//!
//! Provides test infrastructure for the connector layer:
//! - `connector`: ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster, TestConnectorWorker
//! - `e2e`: End-to-end cluster tests
//! - `recorder`: Method-level event recording for connector operations (snapshot testing with insta)
//!
//! Scheduler integration tests live in `kvbm-scheduler/tests/` to avoid circular dev-dependencies.
//!
//! Sub-crate testing modules are NOT re-exported here — use them directly:
//! - `kvbm_engine::testing::*` for managers, token_blocks, physical, distributed, events, messenger, offloading
//! - `kvbm_logical::testing::*` for blocks, sequences, pools, config
//! - `kvbm_physical::testing::*` for TestAgent, physical layouts

pub mod connector;

pub use connector::{
    ConnectorTestConfig, TestConnectorCluster, TestConnectorInstance, TestConnectorWorker,
};

pub mod e2e;

pub mod recorder;

// Scheduler integration tests live in kvbm-scheduler to avoid circular dev-dependency.
// See lib/kvbm-scheduler/tests/ for connector+scheduler integration tests.
