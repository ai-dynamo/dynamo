// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Facade re-exporting from `dynamo-kvbm-logical`.

pub use dynamo_kvbm_logical::{
    blocks, events, manager, pools, registry,
    BlockRegistry, BlockError, BlockMetadata, CompleteBlock, ImmutableBlock,
    MutableBlock, WeakBlock, LogicalLayoutHandle,
};

pub use super::BlockId;
pub use super::SequenceHash;

pub use dynamo_kvbm_logical::testing;
