// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
pub mod draft;
mod metrics;
pub mod protocol;
pub mod queue;
pub mod target;
pub mod transport;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};

pub const PROTOCOL: &str = "mock-specdec-zmq-v1";
pub const DP_RANK: u32 = 0;

pub(crate) fn backend_error(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}
