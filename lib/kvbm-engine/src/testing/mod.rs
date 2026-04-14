// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../../docs/testing.md")]

pub mod distributed;
pub mod events;
pub mod managers;
pub mod messenger;
pub mod offloading;
pub mod physical;
pub mod token_blocks;

// Re-export commonly used testing utilities
pub use distributed::TestSession;
pub use events::{EventsPipelineConfig, EventsPipelineConfigBuilder, EventsPipelineFixture};
pub use managers::{
    InstancePopulationResult, InstancePopulationSpec, MultiInstancePopulator,
    MultiInstancePopulatorBuilder, PopulatedInstances, TestManagerBuilder, TestRegistryBuilder,
    create_and_populate_manager, populate_manager_with_blocks,
};
pub use messenger::{MessengerPair, create_messenger_pair_tcp, create_messenger_tcp};
pub use physical::{TestAgent, TestAgentBuilder, TransferChecksums};
pub use token_blocks::*;
