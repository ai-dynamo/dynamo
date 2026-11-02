// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::ErrorClass;

use super::{
    super::{decision_lifecycle, error, metrics},
    DispatchError,
};

pub(super) fn lifecycle_error(cause: decision_lifecycle::OperationError) -> DispatchError {
    match cause {
        decision_lifecycle::OperationError::Cancelled => DispatchError::cancelled(),
        decision_lifecycle::OperationError::Backend(cause) => classify(cause.as_ref()),
    }
}

pub(super) fn classify(cause: &(dyn std::error::Error + 'static)) -> DispatchError {
    if metrics::request_was_cancelled(cause) {
        return DispatchError::cancelled();
    }
    if metrics::request_was_rejected(cause) {
        return DispatchError {
            code: None,
            status: error::overload_status_code(),
            public_message: "decision scoring worker capacity is exhausted",
            detail: "backend capacity exhausted".to_string(),
            retry_after: true,
            is_capacity_exhausted: true,
        };
    }
    if let Some(error) = error::find_canonical_error_in_chain(cause) {
        if error.class().normalized() == ErrorClass::BackendProtocol {
            return DispatchError::invalid_executor("invalid decision executor result");
        }
        if error.reason().as_str() == "runtime.unclassified" {
            return DispatchError::unavailable("native scoring transport failed");
        }
        if let error::ClientErrorAction::Respond {
            status,
            public_message,
        } = error::http_action_for_error(error)
        {
            let class = error.class().normalized();
            return DispatchError {
                code: None,
                status: if class == ErrorClass::InvalidRequest {
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY
                } else {
                    status
                },
                public_message,
                detail: error.reason().as_str().to_string(),
                retry_after: class == ErrorClass::CapacityExhausted,
                is_capacity_exhausted: class == ErrorClass::CapacityExhausted,
            };
        }
    }
    DispatchError::unavailable("native scoring engine failed")
}
