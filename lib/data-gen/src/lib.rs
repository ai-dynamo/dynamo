// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared schemas and primitives for Dynamo data generation.
//!
//! The crate hosts the Mooncake replay JSONL schema and helpers, plus the
//! Dynamo request-trace loader and transient lowering used to build mocker
//! replay models in memory. Direct Dynamo request-trace replay does not write
//! an intermediate Mooncake file. Shared row types keep trace producers and
//! consumers in lockstep.

pub mod mooncake;
pub mod request_trace;

pub use mooncake::{
    AgenticMooncakeRow, AgenticToolEvent, MooncakeJsonlWriter, MooncakeRow, RollingHashIdMapper,
    WriterStats, hash_token_blocks, ids_for_sequence_hashes, require_positive,
    sequence_hashes_for_tokens, try_hash_token_blocks, write_empty_files,
};
