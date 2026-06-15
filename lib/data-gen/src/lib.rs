// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared schemas and primitives for Dynamo data generation.
//!
//! Today the crate hosts the Mooncake replay JSONL row schema, the rolling
//! block-hash-to-id mapper, and the JSONL writer used by trace producers
//! (e.g. the Claude exporter in `dynamo-bench`) and trace consumers
//! (e.g. `dynamo-mocker`). A single `MooncakeRow` type with both `Serialize`
//! and `Deserialize` derives keeps the producer and consumer in lockstep and
//! eliminates the schema drift that previously existed between two private
//! copies of the schema.

pub mod mooncake;

pub use mooncake::{
    AgenticMooncakeRow, AgenticToolEvent, MooncakeJsonlWriter, MooncakeRow, RollingHashIdMapper,
    WriterStats, hash_token_blocks, ids_for_sequence_hashes, require_positive,
    try_hash_token_blocks, write_empty_files,
};
