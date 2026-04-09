// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical layout and transfer testing utilities.
//!
//! Note: These are local implementations using workspace-local types.
//! When kvbm-physical moves to a workspace path dependency, these can
//! be replaced with re-exports from `kvbm_physical::testing`.

pub use kvbm_physical::testing::*;
