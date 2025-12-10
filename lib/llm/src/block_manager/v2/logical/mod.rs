// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical block management for v2 KV cache.
//!
//! This module handles the logical layer of block management:
//! - Object storage registry (sequence hash → object key mapping)
//! - Distributed registry for multi-node coordination
//! - Block location tracking across storage tiers
//!
//! The logical layer sits above the physical layer (layouts, transfers)
//! and provides semantic block operations.
//!
//! # Modules
//!
//! - [`registry`]: Local object registry (single-node, Moka-based)
//! - [`distributed`]: Distributed registry for multi-node deployments

pub mod distributed;
pub mod registry;

pub use registry::ObjectRegistry;

// Re-export distributed types for convenience
pub use distributed::{DistributedRegistry, HubStats, MatchResult, ObjectKey, OffloadResult, RegistryHub};

