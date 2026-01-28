// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Router - Radix tree data structures for LLM KV cache routing.
//!
//! This crate provides the core radix tree implementation and protocols for
//! efficient KV cache lookup and routing in distributed LLM inference systems.

pub mod approx;
pub mod indexer;
pub mod protocols;

// Re-export key types for convenience
pub use indexer::{MaybeError, RadixTree, RouterEvent};
pub use protocols::{LocalBlockHash, WorkerId, compute_block_hash_for_seq};
