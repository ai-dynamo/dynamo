// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extension point for custom block allocation strategies.
//!
//! Implement [`BlockAllocator`] to plug in a custom allocation strategy
//! (e.g., remote/networked block allocation). Use [`FifoBlockAllocator`]
//! as a reference implementation.

pub use crate::blocks::Block;
pub use crate::blocks::state::Reset;
pub use crate::blocks::{BlockError, BlockMetadata};
pub use crate::pools::BlockAllocator;
pub use crate::pools::FifoBlockAllocator;
pub use crate::registry::{PresenceDelegate, typed_presence_delegate};
pub use crate::BlockId;
