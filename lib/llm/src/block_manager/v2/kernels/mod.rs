// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe-ish wrappers around the CUDA block/universal packing kernels.
//!
//! The core ideas:
//! * A “block” represents the stack of `nl * no` tensors arranged either as NHD
//!   (inner axes `[nt, nh, hd]`) or HND (inner axes `[nh, nt, hd]`).
//! * A “universal” tensor is `[nh, nl, no, nt, hd]` stored contiguously.
//! * An “operational” tensor is `[nl, no, inner]` with `inner = nt * nh * hd`.
//!
//! Host code calls these helpers with flattened pointer tables so a single
//! launch can move many logical blocks in one go.

#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

/// Numeric tags passed across the FFI boundary to select the CUDA template.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDataType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F64 = 3,
}

/// Identifies how each `[nt, nh, hd]` chunk is laid out in device memory.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockLayout {
    NHD = 0,
    HND = 1,
}

/// Direction flag for copying between block stacks and operational buffers.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationalCopyDirection {
    BlockToOperational = 0,
    OperationalToBlock = 1,
}

/// Selects how the operational copy should move data.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationalCopyBackend {
    /// Try cudaMemcpyBatchAsync, fall back to cudaMemcpyAsync, then the kernel.
    Auto = 0,
    /// Force the custom CUDA kernel path.
    KernelOnly = 1,
    /// Issue one cudaMemcpyAsync per chunk.
    MemcpyAsync = 2,
    /// Invoke cudaMemcpyBatchAsync directly.
    MemcpyBatch = 3,
}
