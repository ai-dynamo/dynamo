// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension points for custom block lifecycle strategies.
//!
//! - Implement [`BlockAllocator`] to plug in a custom allocation strategy
//!   (e.g., remote/networked block allocation). Use [`FifoBlockAllocator`]
//!   as a reference implementation.
//! - Implement [`InactivePoolBackend`] to customize inactive pool storage,
//!   eviction, and the [`should_reset`](InactivePoolBackend::should_reset)
//!   hook. Use [`ResetInactiveBlocksBackend`] to disable the inactive cache.

pub use crate::BlockId;
pub use crate::SequenceHash;
pub use crate::blocks::Block;
pub use crate::blocks::state::{Registered, Reset};
pub use crate::blocks::{BlockError, BlockMetadata};
pub use crate::pools::BlockAllocator;
pub use crate::pools::FifoBlockAllocator;
pub use crate::pools::InactivePoolBackend;
pub use crate::pools::ResetInactiveBlocksBackend;
pub use crate::registry::{PresenceDelegate, typed_presence_delegate};
