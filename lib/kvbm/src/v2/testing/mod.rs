// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testing utilities for KVBM v2.
//!
//! This module provides reusable test infrastructure for:
//! - Token block generation
//! - BlockManager setup and configuration
//! - Nova instance creation and pairing
//! - Distributed leader test scenarios
//! - Physical layout and transfer testing
//! - Connector worker and client testing
//! - End-to-end multi-instance tests

pub mod connector;
pub mod distributed;
pub mod e2e;
pub mod events;
pub mod managers;
pub mod nova;
pub mod offloading;
pub mod physical;
pub mod scheduler;
pub mod token_blocks;

// Re-export commonly used testing utilities

// Physical layer utilities
pub use physical::{TestAgent, TestAgentBuilder, TransferChecksums};

// Manager utilities
pub use managers::{
    MultiInstancePopulator, MultiInstancePopulatorBuilder, PopulatedInstances,
    InstancePopulationResult, InstancePopulationSpec,
};

// Distributed testing utilities
pub use distributed::TestSession;

// Events pipeline utilities
pub use events::{EventsPipelineConfig, EventsPipelineConfigBuilder, EventsPipelineFixture};
