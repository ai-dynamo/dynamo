// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3-compatible object storage implementations.
//!
//! This module contains all S3-specific implementations:
//! - [`S3ObjectBlockClient`] - Implements [`ObjectBlockOps`] for S3/MinIO
//! - [`S3LockManager`] - Implements [`ObjectLockManager`] for distributed locking
//!
//! All types in this module are feature-gated behind `s3`. Consumers should use
//! the factory functions in the parent [`object`](super) module to create trait
//! objects without needing to depend on the `s3` feature.

mod client;
mod lock;

pub use client::{S3Config, S3ObjectBlockClient};
pub use lock::S3LockManager;


