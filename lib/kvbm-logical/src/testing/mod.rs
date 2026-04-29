// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test fixtures for `v2/logical/` module tests.
//!
//! This module consolidates duplicated test helpers across registry, pools, and manager tests.
//! It uses internal types like `Block<T, Staged>` so it must be behind `#[cfg(test)]`.
//!
//! Note: `v2/testing/` uses only the public API and remains separate.

pub mod config;

#[allow(dead_code)]
mod blocks;
mod managers;
#[allow(dead_code)]
mod pools;
#[allow(dead_code)]
mod sequences;
mod token_blocks;

// ============================================================================
// Canonical test metadata types
// ============================================================================

/// Primary test metadata type used across all v2 tests.
/// Replaces: `TestMetadata` (registry), `TestData` (pools), `TestBlockData` (manager)
#[derive(Debug, Clone, PartialEq)]
pub struct TestMeta {
    pub value: u64,
}

/// Multi-type registry test metadata A
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataA(pub u32);

/// Multi-type registry test metadata B
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataB(pub String);

/// Multi-type registry test metadata C (unit struct)
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataC;

// ============================================================================
// Constants
// ============================================================================

/// Standard salt value for test token blocks.
/// Standardizes mixed salt values (42 in pools/manager, 1337 in registry).
pub const TEST_SALT: u64 = 42;

// ============================================================================
// Re-exports from submodules
// ============================================================================

// pub items — usable by downstream crates via the testing feature
pub use managers::{create_test_manager, create_test_manager_with_block_size};
pub use sequences::BlockSequenceBuilder;
pub use token_blocks::{
    create_iota_token_block, create_test_token_block, sequential_tokens, tokens_for_id,
};

// pub(crate) items — internal helpers using internal types
#[cfg(test)]
pub(crate) use blocks::{
    TestBlockBuilder, create_registered_block, create_reset_block, create_reset_blocks,
    create_staged_block,
};
#[cfg(test)]
pub(crate) use pools::TestPoolSetupBuilder;
