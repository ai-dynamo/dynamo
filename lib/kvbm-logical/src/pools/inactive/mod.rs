// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inactive-pool eviction backends. The pool itself is part of the unified
//! [`BlockStore`](super::store::BlockStore); this module only houses the
//! pluggable [`InactiveIndex`](super::store::InactiveIndex) implementations.

pub mod backends;

use super::InactiveBlock;
use crate::BlockId;
