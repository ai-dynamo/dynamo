// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-specific testing utilities.
//!
//! Provides test infrastructure for the connector layer:
//! - `connector`: ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster, TestConnectorWorker
//! - `e2e`: End-to-end cluster tests (TODO: Plan 03)
//! - `scheduler`: Scheduler integration tests and mock types (TODO: Plan 03)
//!
//! Sub-crate testing modules are NOT re-exported here — use them directly:
//! - `kvbm_engine::testing::*` for managers, token_blocks, physical, distributed, events, messenger, offloading
//! - `kvbm_logical::testing::*` for blocks, sequences, pools, config
//! - `kvbm_physical::testing::*` for TestAgent, physical layouts

pub mod connector;

pub use connector::{
    ConnectorTestConfig,
    TestConnectorInstance,
    TestConnectorCluster,
    TestConnectorWorker,
};

// TODO(Phase 4 Plan 03): Uncomment after e2e/ and scheduler/ are ported
// pub mod e2e;
// pub mod scheduler;
