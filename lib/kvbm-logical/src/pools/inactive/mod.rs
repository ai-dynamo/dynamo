// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inactive-pool eviction backends. The pool itself is part of the unified
//! [`BlockStore`](super::store::BlockStore); this module only houses the
//! pluggable [`InactiveIndex`](super::store::InactiveIndex) implementations.
//!
//! Note: *whether* a released primary block enters the inactive index at
//! all is decided at the [`BlockStore`](super::store::BlockStore) level
//! (per-block flag on [`ImmutableBlockInner`](crate::blocks::ImmutableBlockInner)
//! plus a store-wide default), not by the backend. A backend cannot see
//! slot state, the free list, or registration handles — see
//! [`BlockStore::release_primary`](super::store::BlockStore).

pub mod backends;

use super::InactiveBlock;
use crate::BlockId;
