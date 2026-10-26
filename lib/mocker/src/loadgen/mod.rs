// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo composition adapters for the AISimulate workload driver.

mod router;

pub use aisimulate_core::replay::loadgen::*;
pub(crate) use router::local_block_hashes;
pub use router::{DynamoTraceRouterExt, RouterSequence, SequenceHashMode, to_router_sequences};

#[cfg(test)]
mod tests;
