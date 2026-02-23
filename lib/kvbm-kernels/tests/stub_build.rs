// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests that only compile when stub kernels are in use (no CUDA available).
//!
//! Complementary to the `memcpy_batch::stubs_not_active` test which asserts
//! the opposite under `not(stub_kernels)`.

#![cfg(stub_kernels)]

use kvbm_kernels::{is_memcpy_batch_available, is_using_stubs};

#[test]
fn stubs_active() {
    assert!(
        is_using_stubs(),
        "expected is_using_stubs() == true under stub build"
    );
}

#[test]
fn memcpy_batch_unavailable_under_stubs() {
    assert!(
        !is_memcpy_batch_available(),
        "expected is_memcpy_batch_available() == false under stub build"
    );
}
