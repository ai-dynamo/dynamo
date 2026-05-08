// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error type re-exports.
//!
//! Backends use [`DynamoError`] (the workspace-wide standardized error) with
//! [`ErrorType`] categorization. For engine-originated failures, use
//! `ErrorType::Backend(BackendError::X)`.
//!
//! Example:
//!
//! ```ignore
//! use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
//!
//! return Err(DynamoError::builder()
//!     .error_type(ErrorType::Backend(BackendError::InvalidArgument))
//!     .message(format!("bad param: {reason}"))
//!     .build());
//! ```

pub use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
