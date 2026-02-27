// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testing utilities for KVBM distributed engine.
//!
//! Core block and token utilities are re-exported from `kvbm_logical::testing`
//! and `kvbm_physical::testing`. This module adds engine-specific test
//! infrastructure: messenger pairing, events pipeline, distributed sessions,
//! and multi-instance population helpers.

pub mod distributed;
pub mod events;
pub mod managers;
pub mod messenger;
pub mod offloading;
pub mod physical;
pub mod token_blocks;

// Re-export commonly used testing utilities
pub use physical::{TestAgent, TestAgentBuilder, TransferChecksums};
pub use managers::{
    InstancePopulationResult, InstancePopulationSpec, MultiInstancePopulator,
    MultiInstancePopulatorBuilder, PopulatedInstances, TestManagerBuilder, TestRegistryBuilder,
    populate_manager_with_blocks, create_and_populate_manager,
};
pub use distributed::TestSession;
pub use events::{EventsPipelineConfig, EventsPipelineConfigBuilder, EventsPipelineFixture};
pub use messenger::{MessengerPair, create_messenger_tcp, create_messenger_pair_tcp};
pub use token_blocks::*;
