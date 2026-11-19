// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};

/// Construct a typed invalid-argument error for validation performed at an
/// HTTP protocol adapter boundary.
pub(crate) fn invalid_argument(message: impl Into<String>) -> DynamoError {
    let message = message.into();
    DynamoError::builder()
        .error_type(DynamoErrorType::InvalidArgument)
        .message(message.clone())
        .public_message(message)
        .build()
}
