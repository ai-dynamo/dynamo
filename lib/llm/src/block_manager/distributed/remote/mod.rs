// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote registry handle module.
mod handle;

pub use handle::{
    CanOffloadResult, PositionalRemoteHandle, RemoteHandle, RemoteHashOperations,
    RemoteHashOperationsSync, RemoteOperation,
};
