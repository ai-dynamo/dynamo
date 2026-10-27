// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Implementations of [`crate::pool_discovery`], one module per
//! store.

pub mod kubernetes;
pub mod memory;
mod stream;
