// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Facade re-exporting from `dynamo-kvbm-logical`.

pub use dynamo_kvbm_logical::{
    BlockError, BlockMetadata, BlockRegistry, CompleteBlock, ImmutableBlock, LogicalLayoutHandle,
    MutableBlock, WeakBlock, blocks, events, manager, pools, registry,
};

pub use super::BlockId;
pub use super::SequenceHash;

pub use dynamo_kvbm_logical::testing;
