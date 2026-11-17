// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};

/// For the kinds `dynamo_sidecar_common` has no constructor for.
pub(crate) fn backend_error(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub(crate) fn is_kind(error: &DynamoError, kind: BackendError) -> bool {
    error.error_type() == ErrorType::Backend(kind)
}

/// The peer did not answer, as opposed to answering with a refusal.
pub(crate) fn is_transport_failure(error: &DynamoError) -> bool {
    [
        BackendError::CannotConnect,
        BackendError::Disconnected,
        BackendError::ResponseTimeout,
    ]
    .into_iter()
    .any(|kind| is_kind(error, kind))
}
